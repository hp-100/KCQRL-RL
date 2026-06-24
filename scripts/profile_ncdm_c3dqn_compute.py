from __future__ import annotations
import argparse, time, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.ncdm import OfficialNCDM
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork

CASES = [
    ("Base C3DQN, Top-K=128", "isab", 0, 128),
    ("Set-C3DQN ISAB M=8, Top-K=128", "isab", 8, 128),
    ("Set-C3DQN ISAB M=16, Top-K=256", "isab", 16, 256),
    ("Set-C3DQN Full Attention, Top-K=128", "full_self_attention", 16, 128),
]

def main():
    ap=argparse.ArgumentParser(description="Profile Efficient Set-C3DQN-NCDM compute cost")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--knowledge-dim", type=int, default=36)
    ap.add_argument("--history-len", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args=ap.parse_args(); dev=torch.device(args.device)
    rows=[]
    for name, enc, m, topk in CASES:
        ncdm=OfficialNCDM(1, max(topk+8, 300), args.knowledge_dim).to(dev).eval()
        q=torch.randint(0,2,(max(topk+8,300),args.knowledge_dim),device=dev).float(); q[:,0]=1
        cache=NCDMItemFeatureCache(ncdm,q,dev)
        samples=[]
        for b in range(args.batch_size):
            samples.append({"history_item_ids":list(range(args.history_len)),"history_responses":[float(i%2) for i in range(args.history_len)],"candidate_item_ids":list(range(args.history_len, args.history_len+topk)),"mastery":[0.5]*args.knowledge_dim,"coverage":[0.0]*args.knowledge_dim,"policy_step":1,"selected_item_id":args.history_len})
        batch=pad_c3dqn_batch(samples,cache,args.history_len+1)
        net=CandidateConditionedNCDMQNetwork(args.knowledge_dim,d_model=64,n_heads=4,num_history_layers=1,candidate_set_encoder=enc,num_inducing_points=max(1,m),num_set_layers=1,full_attention_max_candidates=128).to(dev)
        params=sum(p.numel() for p in net.parameters())
        if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
        for _ in range(2): net(**{k:v for k,v in batch.items() if k!='action_index'})
        if dev.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter(); qv,_=net(**{k:v for k,v in batch.items() if k!='action_index'}); loss=qv[qv>-1e8].mean(); loss.backward()
        if dev.type=='cuda': torch.cuda.synchronize()
        step_ms=(time.perf_counter()-t0)*1000
        peak=torch.cuda.max_memory_allocated(dev)/1024/1024 if dev.type=='cuda' else 0.0
        rows.append((name, step_ms, params, topk, peak))
    print("method,training_step_ms,parameters,candidate_count,peak_memory_mb")
    for r in rows: print(f"{r[0]},{r[1]:.3f},{r[2]},{r[3]},{r[4]:.1f}")
if __name__=='__main__': main()
