"""Independent MIRT-native DDPG training pipeline."""
from __future__ import annotations
import csv, json, random, subprocess
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from agents.replay_buffer import ReplayBuffer
from core.mirt_state_builder import build_mirt_state, compute_action_normalizer, nearest_item
from evaluation.offline_eval import StudentSequence
from evaluation.metrics import metric_bundle, nll_score
from models.mirt import load_mirt_checkpoint, fit_student_theta, predict_with_theta
from models.mirt_actor import MIRTActor, STATE_DEFINITION_VERSION, ACTOR_ARCHITECTURE
from models.mirt_critic import MIRTCritic
from reward.mirt_reward import query_nll, nll_drop_reward

class MIRTDDPGTrainer:
    def __init__(self, config, device=None):
        self.config=dict(config); self.device=torch.device(device or ("cuda" if torch.cuda.is_available() and config.get("device")!="cpu" else "cpu"))
        t=config.get('training',{}); self.seed=int(t.get('seed',42)); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.out=Path(t.get('output_dir','outputs/mirt_ddpg')); self.out.mkdir(parents=True,exist_ok=True)
        self.max_steps=int(t.get('max_steps',20)); self.epochs=int(t.get('epochs',1)); self.batch_size=int(t.get('batch_size',32))
        self.gamma=float(t.get('gamma',.99)); self.tau=float(t.get('tau',.005)); self.policy_delay=int(t.get('policy_delay',2)); self.q_clip=float(t.get('q_clip',20.0))
        self.theta_cfg=dict(config.get('theta_fit',{})) or {'steps':5 if t.get('smoke') else 30}
        self.mirt=load_mirt_checkpoint(config['assets']['mirt_checkpoint'], self.device); self._assert_frozen()
    def _assert_frozen(self):
        self.mirt.eval()
        for p in self.mirt.parameters():
            if p.requires_grad or p.grad is not None: raise RuntimeError('MIRT parameters must be frozen and grad-free')
    def _load_sequences(self,path):
        rows=[]; import pandas as pd
        df=pd.read_csv(path); sid=df.columns[0]; iid='item_id' if 'item_id' in df.columns else df.columns[1]; resp='response' if 'response' in df.columns else df.columns[2]
        for s,g in df.groupby(sid): rows.append(StudentSequence(str(s), [int(x) for x in g[iid]], [float(x) for x in g[resp]]))
        return rows
    def _split(self, seqs):
        rng=random.Random(self.seed); ids=sorted([s.student_id for s in seqs]); rng.shuffle(ids); cut=max(1,int(.8*len(ids)))
        train=set(ids[:cut]); valid=set(ids[cut:]); manifest={'policy_train_students':sorted(train),'policy_validation_students':sorted(valid),'students':{}}
        out=[]
        for seq in seqs:
            seen={};
            for i,r in zip(seq.item_ids,seq.responses):
                if 0<=int(i)<self.mirt.n_items and int(i) not in seen: seen[int(i)]=float(r)
            items=list(seen); rng2=random.Random(f'{self.seed}:{seq.student_id}'); rng2.shuffle(items); qn=max(5,int(.2*len(items)))
            if len(items)<qn+1: continue
            query=items[:qn]; support=items[qn:]
            manifest['students'][seq.student_id]={'split':'train' if seq.student_id in train else 'validation','support':support,'query':query}
            out.append((seq.student_id, seq.student_id in train, support, [seen[i] for i in support], query, [seen[i] for i in query]))
        (self.out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
        return out
    def _soft(self, src, dst):
        for s,d in zip(src.parameters(),dst.parameters()): d.data.mul_(1-self.tau).add_(s.data, alpha=self.tau)
    def _validate(self, actor, rows, normalizer):
        vals=[]; auc=[]
        for sid,is_train,supp,sr,query,qr in rows:
            if is_train: continue
            hist_i=[]; hist_r=[]; cand=list(supp)
            for step in range(min(10,len(cand))):
                st=build_mirt_state(self.mirt,hist_i,hist_r,step,self.max_steps,self.theta_cfg,self.device)
                with torch.no_grad(): a=actor(st)
                it=nearest_item(a.squeeze(0),cand,self.mirt,normalizer,self.device); j=cand.index(it); cand.pop(j); hist_i.append(it); hist_r.append(sr[supp.index(it)])
            th=fit_student_theta(self.mirt,hist_i,hist_r,device=self.device,**self.theta_cfg)
            p=predict_with_theta(self.mirt,th,query).detach().cpu().tolist(); mb=metric_bundle(qr,p); vals.append(mb['nll']); auc.append(mb['auc'])
        return {'validation_query_nll':float(np.nanmean(vals)) if vals else float('nan'), 'validation_query_auc':float(np.nanmean(auc)) if auc else float('nan')}
    def train(self):
        path=self.config['assets'].get('train_valid_sequences','kc_level/train_valid_sequences.csv')
        rows=self._split(self._load_sequences(path)); valid_items=sorted({i for _,_,s,_,_,_ in rows for i in s})
        normalizer=compute_action_normalizer(self.mirt, valid_items, self.device)
        actor=MIRTActor(hidden_dim=int(self.config.get('model',{}).get('hidden_dim',128))).to(self.device); targ_a=MIRTActor(hidden_dim=actor.hidden_dim).to(self.device); targ_a.load_state_dict(actor.state_dict())
        critic=MIRTCritic(q_clip=self.q_clip).to(self.device); targ_c=MIRTCritic(q_clip=self.q_clip).to(self.device); targ_c.load_state_dict(critic.state_dict())
        ao=torch.optim.Adam(actor.parameters(), lr=float(self.config.get('training',{}).get('actor_lr',3e-5))); co=torch.optim.Adam(critic.parameters(), lr=float(self.config.get('training',{}).get('critic_lr',1e-5)))
        rb=ReplayBuffer(seed=self.seed); hist=[]; best=float('inf')
        for ep in range(1,self.epochs+1):
            rewards=[]; cl=al=mq=tq=qfrac=float('nan'); uniq=set(); updates=0
            for sid,is_train,supp,sr,query,qr in rows:
                if not is_train: continue
                cand=list(supp); hi=[]; hr=[]; prev=None
                for step in range(min(self.max_steps,len(cand))):
                    st=build_mirt_state(self.mirt,hi,hr,step,self.max_steps,self.theta_cfg,self.device)
                    th=fit_student_theta(self.mirt,hi,hr,device=self.device,**self.theta_cfg); prev=query_nll(self.mirt,th,query,qr) if prev is None else prev
                    with torch.no_grad(): act=actor(st); act=act.squeeze(0)+torch.randn(37,device=self.device)*0.1
                    it=nearest_item(act,cand,self.mirt,normalizer,self.device); uniq.add(it); cand.remove(it); hi2=hi+[it]; hr2=hr+[sr[supp.index(it)]]
                    th2=fit_student_theta(self.mirt,hi2,hr2,device=self.device,**self.theta_cfg); cur=query_nll(self.mirt,th2,query,qr); r=nll_drop_reward(prev,cur,self.config.get('training',{}).get('reward_scale',10.0),self.config.get('training',{}).get('reward_clip',5.0)); rewards.append(r)
                    ns=build_mirt_state(self.mirt,hi2,hr2,step+1,self.max_steps,self.theta_cfg,self.device); rb.push(st.cpu().numpy(), act.detach().cpu().numpy(), r, ns.cpu().numpy(), float(step+1>=self.max_steps or not cand)); hi,hr,prev=hi2,hr2,cur
                    if len(rb)>=self.batch_size:
                        s,a,rw,nsb,d=rb.sample(self.batch_size,self.device); rw=rw.view(-1,1); d=d.view(-1,1)
                        with torch.no_grad(): na=targ_a(nsb); y=rw+self.gamma*(1-d)*targ_c(nsb,na)
                        q=critic(s,a); loss=F.huber_loss(q,y.clamp(-self.q_clip,self.q_clip)); co.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),1.0); co.step(); cl=loss.detach().item(); mq=q.detach().mean().item(); tq=y.detach().mean().item(); qfrac=(q.detach().abs()>self.q_clip*.99).float().mean().item()
                        if updates%self.policy_delay==0:
                            pa=actor(s); lossa=-critic(s,pa).mean(); ao.zero_grad(); lossa.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); ao.step(); al=lossa.detach().item()
                            self._soft(actor,targ_a); self._soft(critic,targ_c)
                        updates+=1; self._assert_frozen()
            vm=self._validate(actor,rows,normalizer); rec={'epoch':ep,'mean_reward':float(np.mean(rewards)) if rewards else 0.0,'critic_loss':cl,'actor_loss':al,'mean_q':mq,'target_q_mean':tq,'q_clip_fraction':qfrac,'selected_unique_items':len(uniq),**vm}; hist.append(rec)
            if qfrac==qfrac and qfrac>.5: print('WARNING: critic Q values are saturating near q_clip boundary')
            if rec['validation_query_nll']<best: best=rec['validation_query_nll']; self._save(actor,normalizer,ep,rec,'ddpg_mirt_actor_best.pt')
        self._save(actor,normalizer,self.epochs,hist[-1] if hist else {},'ddpg_mirt_actor_final.pt'); torch.save({'training_history':hist}, self.out/'ddpg_mirt_training_state.pt')
        with (self.out/'training_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
        return hist
    def _save(self,actor,norm,epoch,metrics,name):
        try: git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        except Exception: git=None
        torch.save({'actor_state_dict':actor.state_dict(),'action_mean':norm.mean,'action_std':norm.std,'mirt_dim':self.mirt.n_dims,'hidden_dim':actor.hidden_dim,'state_definition_version':STATE_DEFINITION_VERSION,'actor_architecture':ACTOR_ARCHITECTURE,'theta_fit':self.theta_cfg,'training_config':self.config,'epoch':epoch,'validation_metrics':metrics,'git_commit':git}, self.out/name)
