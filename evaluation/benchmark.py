"""Paper-style paired offline CAT benchmark_v2."""
from __future__ import annotations

import csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

from evaluation.offline_eval import CATOfflineEvaluator, MissingAssetsError, StudentSequence
from evaluation.metrics import metric_bundle, nanmean, gini, nll_score
from evaluation.protocol import StudentSplit, make_student_split, save_manifest, valid_item_count
from evaluation.policies import RandomPolicy, HeuristicMIRTPolicy, DDPGPolicy, OneStepOraclePolicy
from models.ncdm import OfficialNCDM, fit_student_alpha, safe_load_ncdm_checkpoint
from models.actor import LSTMActor


class BenchmarkV2Evaluator:
    def __init__(self, config: Mapping, *, debug=False, ddpg_checkpoint="outputs/ddpg_actor.pt", seeds=None, max_students=None, steps=None, output_dir=None):
        self.config = dict(config)
        b = dict((config.get("benchmark") or {}))
        self.debug = debug
        self.seeds = [int(x) for x in (seeds if seeds is not None else b.get("seeds", [42]))]
        self.steps = [int(x) for x in (steps if steps is not None else b.get("steps", [0,1,3,5,10,20]))]
        self.max_students = int(max_students if max_students is not None else b.get("max_students", 20 if debug else 300))
        self.query_ratio = float(b.get("query_ratio", 0.2)); self.min_query_items = int(b.get("min_query_items", 5))
        self.candidate_size = b.get("candidate_size", None)
        self.save_predictions = bool(b.get("save_predictions", True)); self.save_traces = bool(b.get("save_traces", True))
        self.output_dir = Path(output_dir or b.get("output_dir", "results/benchmark_v2"))
        self.ddpg_checkpoint = Path(ddpg_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") != "cpu" else "cpu")

    def _load_or_synthetic(self):
        legacy = CATOfflineEvaluator(self.config, debug=self.debug, ddpg_checkpoint=str(self.ddpg_checkpoint))
        missing = legacy.missing_required_assets()
        if missing and self.debug:
            q = np.eye(6, dtype=np.float32)[np.arange(30) % 6]
            item_bank = np.random.default_rng(0).normal(size=(30,128)).astype(np.float32)
            seqs=[]
            for s in range(25):
                items=list(range(30)); rng=np.random.default_rng(s); rng.shuffle(items)
                res=[float(((i+s)%3)!=0) for i in items]
                seqs.append(StudentSequence(str(s), items, res))
            class TinyNCDM(torch.nn.Module):
                def __init__(self, exer_n, knowledge_n):
                    super().__init__(); self.knowledge_dim=knowledge_n
                    self.k_difficulty=torch.nn.Embedding(exer_n, knowledge_n)
                    self.e_discrimination=torch.nn.Embedding(exer_n, 1)
                def predict_with_alpha(self, alpha, exer_id, q_matrix):
                    diff=torch.sigmoid(self.k_difficulty(exer_id)); disc=torch.sigmoid(self.e_discrimination(exer_id))*3.0
                    return torch.sigmoid((disc*(torch.sigmoid(alpha)-diff)*q_matrix[exer_id]).sum(dim=-1))
            ncdm = TinyNCDM(len(q), q.shape[1]).to(self.device).eval()
            return q, item_bank, seqs[:self.max_students], ncdm, True
        q, item_bank, seqs = legacy.load()
        ncdm = OfficialNCDM(1, q.shape[0], q.shape[1]).to(self.device)
        safe_load_ncdm_checkpoint(ncdm, legacy.paths["ncdm_checkpoint"], self.device)
        ncdm.eval()
        return q, item_bank, seqs[:self.max_students], ncdm, False

    def _predict(self, ncdm, q_tensor, hist_i, hist_r, target_i):
        if not target_i: return []
        alpha = fit_student_alpha(ncdm, q_tensor, hist_i, hist_r, steps=(2 if self.debug else 8), device=self.device)
        ncdm.eval()
        with torch.no_grad():
            out = ncdm.predict_with_alpha(alpha, torch.tensor(target_i, dtype=torch.long, device=self.device), q_tensor)
        return [float(x) for x in out.detach().cpu().tolist()]

    def _policies(self, q, item_bank, ncdm, synthetic):
        policies=[RandomPolicy(), HeuristicMIRTPolicy("MIRT-MFI"), HeuristicMIRTPolicy("MIRT-KLI"), OneStepOraclePolicy()]
        try:
            actor=None; q_t=torch.tensor(q,dtype=torch.float32,device=self.device); ib_t=torch.tensor(item_bank,dtype=torch.float32,device=self.device)
            if self.ddpg_checkpoint.exists():
                actor=LSTMActor(semantic_dim=item_bank.shape[1], q_dim=q.shape[1]).to(self.device); actor.load_state_dict(torch.load(self.ddpg_checkpoint,map_location=self.device)); actor.eval()
            policies.insert(3, DDPGPolicy(self.ddpg_checkpoint, actor=actor, q_matrix=q_t, item_bank=ib_t, ncdm=ncdm, device=self.device, allow_debug_fallback=self.debug or synthetic))
        except FileNotFoundError as e:
            raise
        return policies

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        q, item_bank, seqs, ncdm, synthetic = self._load_or_synthetic()
        q_tensor=torch.tensor(q,dtype=torch.float32,device=self.device)
        vcount=valid_item_count(q, item_bank, ncdm)
        all_rows=[]; pred_rows=[]; student_rows=[]; traces=[]; metadata={}
        for seed in self.seeds:
            splits=[]; skipped={}
            for seq in seqs:
                sp, reason=make_student_split(seq.student_id, seq.item_ids, seq.responses, seed=seed, valid_count=vcount, query_ratio=self.query_ratio, min_query_items=self.min_query_items)
                if sp: splits.append(sp)
                else: skipped[seq.student_id]=reason or "invalid"
            save_manifest(self.output_dir/f"splits_seed{seed}.json", splits, skipped, {"seed":seed,"query_ratio":self.query_ratio,"min_query_items":self.min_query_items})
            policies=self._policies(q, item_bank, ncdm, synthetic)
            for pol in policies: metadata[pol.name]=pol.metadata.__dict__
            for pol in policies:
                step_pred=defaultdict(lambda: ([],[])); step_stu=defaultdict(list); exposures=defaultdict(Counter); q_inter=defaultdict(int)
                for sp in splits:
                    pol.reset(sp.student_id, seed, {"query_item_ids":sp.query_item_ids})
                    cand=[i for i in sp.support_item_ids if i != sp.warm_start_item]
                    cresp={int(i):float(r) for i,r in zip(sp.support_item_ids, sp.support_responses)}
                    hist_i=[sp.warm_start_item]; hist_r=[sp.warm_start_response]; selected=[]; selected_r=[]
                    max_extra=max(self.steps)
                    checkpoints=set(self.steps)
                    def predict_history(hi, hr, targets): return self._predict(ncdm, q_tensor, hi, hr, list(targets))
                    def query_nll_after(hi, hr): return nll_score(sp.query_responses, self._predict(ncdm, q_tensor, hi, hr, sp.query_item_ids))
                    for t in range(0, max_extra+1):
                        if t in checkpoints:
                            scores=self._predict(ncdm, q_tensor, hist_i, hist_r, sp.query_item_ids)
                            step_pred[t][0].extend(sp.query_responses); step_pred[t][1].extend(scores); q_inter[t]+=len(sp.query_item_ids)
                            mb=metric_bundle(sp.query_responses, scores); step_stu[t].append(mb)
                            for qi,yt,ys in zip(sp.query_item_ids, sp.query_responses, scores):
                                pred_rows.append({"seed":seed,"policy":pol.name,"student_id":sp.student_id,"step":t,"query_item_id":qi,"y_true":yt,"y_score":ys})
                            student_rows.append({"seed":seed,"policy":pol.name,"student_id":sp.student_id,"step":t,**{k:v for k,v in mb.items()}})
                        if t == max_extra or not cand: break
                        avail=cand if self.candidate_size is None else cand[:int(self.candidate_size)]
                        ctx={"query_item_ids":sp.query_item_ids,"predict_history":predict_history,"candidate_response_lookup":cresp,"query_nll_after_history":query_nll_after}
                        item=pol.select(avail, hist_i, hist_r, ctx)
                        if item not in avail: raise RuntimeError(f"{pol.name} selected item outside candidate pool")
                        cand.remove(item); resp=cresp[item]; hist_i.append(item); hist_r.append(resp); selected.append(item); selected_r.append(resp); exposures[t+1][item]+=1
                    traces.append({"student_id":sp.student_id,"policy":pol.name,"seed":seed,"warm_start_item":sp.warm_start_item,"selected_items":selected,"selected_responses":selected_r,"query_items":sp.query_item_ids})
                for step in self.steps:
                    yt,ys=step_pred[step]; micro=metric_bundle(yt,ys); macros={k:nanmean([m[k] for m in step_stu[step]]) for k in ["accuracy","auc","nll","brier"]}
                    cnt=Counter(); [cnt.update(c) for s,c in exposures.items() if s<=step]
                    concepts=set();
                    for item in cnt: concepts.update(np.nonzero(q[item])[0].tolist())
                    all_rows.append({"policy":pol.name,"seed":seed,"step":step,"students":len(splits),"valid_students":len(splits),"skipped_students":len(skipped),"query_interactions":q_inter[step],"selected_items":sum(cnt.values()),"average_test_length":(sum(cnt.values())/len(splits) if splits else 0),"accuracy_micro":micro["accuracy"],"auc_micro":micro["auc"],"nll_micro":micro["nll"],"brier_micro":micro["brier"],"accuracy_macro":macros["accuracy"],"auc_macro":macros["auc"],"nll_macro":macros["nll"],"brier_macro":macros["brier"],"concept_coverage":len(concepts)/q.shape[1],"unique_item_count":len(cnt),"item_exposure_max":max(cnt.values()) if cnt else 0,"item_exposure_gini":gini(cnt.values())})
        self._write_csv(self.output_dir/"per_seed.csv", all_rows)
        agg=[]
        for key in sorted({(r['policy'],r['step']) for r in all_rows}):
            rows=[r for r in all_rows if (r['policy'],r['step'])==key]; out={"policy":key[0],"step":key[1]}
            for col in [c for c in all_rows[0] if c not in ('policy','seed','step')]:
                vals=[float(r[col]) for r in rows if not math.isnan(float(r[col]))]
                out[col+"_mean"]=sum(vals)/len(vals) if vals else float('nan'); out[col+"_std"]=statistics.pstdev(vals) if len(vals)>1 else 0.0
            agg.append(out)
        self._write_csv(self.output_dir/"aggregate.csv", agg)
        if self.save_predictions: self._write_csv(self.output_dir/"predictions.csv", pred_rows)
        self._write_csv(self.output_dir/"per_student.csv", student_rows)
        if self.save_traces:
            with (self.output_dir/"traces.jsonl").open('w') as f:
                for tr in traces: f.write(json.dumps(tr)+"\n")
        (self.output_dir/"policy_metadata.json").write_text(json.dumps(metadata,indent=2))
        (self.output_dir/"run_config.yaml").write_text(yaml.safe_dump(self.config))
        return all_rows

    @staticmethod
    def _write_csv(path, rows):
        if not rows: return
        with Path(path).open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
