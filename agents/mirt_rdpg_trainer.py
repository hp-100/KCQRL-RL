"""Strict RDPG-MIRT trainer using episode replay, burn-in, and BPTT."""
from __future__ import annotations
import csv, json, random, subprocess, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from agents.sequence_replay_buffer import SequenceReplayBuffer
from core.mirt_state_builder import build_mirt_state, compute_action_normalizer, nearest_item
from evaluation.metrics import metric_bundle
from evaluation.offline_eval import StudentSequence
from models.mirt import load_mirt_checkpoint, fit_student_theta, predict_with_theta
from models.mirt_actor import STATE_DEFINITION_VERSION
from models.mirt_recurrent_actor import MIRTRecurrentActor, ACTOR_ARCHITECTURE
from models.mirt_recurrent_critic import MIRTRecurrentCritic
from reward.mirt_reward import query_nll, nll_drop_reward

class MIRTRDPGTrainer:
    def __init__(self, config, device=None):
        self.config=dict(config); t=self.config.get('training',{}); self.device=torch.device(device or ('cuda' if torch.cuda.is_available() and self.config.get('device')!='cpu' else 'cpu'))
        self.seed=int(t.get('seed',42)); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.out=Path(t.get('output_dir','outputs/mirt_rdpg')); self.out.mkdir(parents=True,exist_ok=True)
        self.max_steps=int(t.get('max_steps',10)); self.epochs=int(t.get('epochs',2)); self.gamma=float(t.get('gamma',.99)); self.tau=float(t.get('tau',.005)); self.policy_delay=int(t.get('policy_delay',2)); self.grad_clip=float(t.get('gradient_clip',1.0)); self.q_clip=float(t.get('q_clip',20.0))
        sr=self.config.get('sequence_replay',{}); self.burn=int(sr.get('burn_in_length',3)); self.unroll=int(sr.get('unroll_length',5)); self.batch=int(sr.get('batch_sequences',8)); self.min_eps=int(sr.get('min_episodes_before_update',8)); self.updates_per_episode=int(t.get('updates_per_episode',1))
        self.theta_cfg=dict(self.config.get('theta_fit',{})) or {'steps':5 if t.get('smoke') else 30}
        self.mirt=load_mirt_checkpoint(self.config['assets']['mirt_checkpoint'], self.device); self._assert_frozen()
    def _assert_frozen(self):
        self.mirt.eval()
        for p in self.mirt.parameters():
            if p.requires_grad or p.grad is not None: raise RuntimeError('MIRT parameters must be frozen and grad-free')
    def _load_sequences(self,path):
        import pandas as pd
        df=pd.read_csv(path); sid=df.columns[0]; iid='item_id' if 'item_id' in df.columns else df.columns[1]; resp='response' if 'response' in df.columns else df.columns[2]
        return [StudentSequence(str(s), [int(x) for x in g[iid]], [float(x) for x in g[resp]]) for s,g in df.groupby(sid)]
    def _split(self,seqs):
        rng=random.Random(self.seed); ids=sorted(s.student_id for s in seqs); rng.shuffle(ids); cut=max(1,int(.8*len(ids))); train=set(ids[:cut]); rows=[]; manifest={'policy_train_students':sorted(train),'policy_validation_students':sorted(set(ids)-train),'students':{}}
        for seq in seqs:
            seen={}
            for i,r in zip(seq.item_ids,seq.responses):
                if 0<=int(i)<self.mirt.n_items and int(i) not in seen: seen[int(i)]=float(r)
            items=list(seen); random.Random(f'{self.seed}:{seq.student_id}').shuffle(items); qn=max(5,int(.2*len(items)))
            if len(items)<qn+1: continue
            query=items[:qn]; supp=items[qn:]; manifest['students'][seq.student_id]={'split':'train' if seq.student_id in train else 'validation','support':supp,'query':query}; rows.append((seq.student_id,seq.student_id in train,supp,[seen[i] for i in supp],query,[seen[i] for i in query]))
        (self.out/'split_manifest.json').write_text(json.dumps(manifest,indent=2)); return rows
    def _soft(self,src,dst):
        for s,d in zip(src.parameters(),dst.parameters()): d.data.mul_(1-self.tau).add_(s.data,alpha=self.tau)
    def _optimize(self, rb, actor, critic, ta, tc, ao, co, stepno):
        b=rb.sample(self.batch,self.burn,self.unroll,self.device); s,a,r,ns,d,m=[b[k] for k in ('states','actions','rewards','next_states','dones','valid_mask')]
        bs=s[:,:self.burn]; us=s[:,self.burn:]; ba=a[:,:self.burn]; ua=a[:,self.burn:]; br=ns[:,:self.burn]; uns=ns[:,self.burn:]; um=m[:,self.burn:]
        with torch.no_grad():
            _, ch=critic.forward_sequence(bs,ba); _, tch=tc.forward_sequence(br,ba); _, tah=ta.forward_sequence(br)
            na,_=ta.forward_sequence(uns,tah); yq,_=tc.forward_sequence(uns,na,tch); y=(r[:,self.burn:]+self.gamma*(1-d[:,self.burn:])*yq).clamp(-self.q_clip,self.q_clip)
        q,_=critic.forward_sequence(us,ua,ch); el=F.huber_loss(q,y,reduction='none'); cl=(el*um).sum()/um.sum().clamp_min(1); co.zero_grad(); cl.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),self.grad_clip); co.step()
        al=torch.tensor(float('nan'),device=self.device)
        if stepno%self.policy_delay==0:
            with torch.no_grad(): _, ah=actor.forward_sequence(bs); _, ach=critic.forward_sequence(bs,ba)
            pa,_=actor.forward_sequence(us,ah); pq,_=critic.forward_sequence(us,pa,ach); al=-(pq*um).sum()/um.sum().clamp_min(1); ao.zero_grad(); al.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),self.grad_clip); ao.step(); self._soft(actor,ta); self._soft(critic,tc)
        return {'critic_loss':float(cl.detach()), 'actor_loss':float(al.detach()), 'mean_q':float(q.detach().mean()), 'target_q_mean':float(y.detach().mean()), 'q_clip_fraction':float((q.detach().abs()>self.q_clip*.99).float().mean()), 'sampled_valid_timesteps':int(um.sum().item()), 'shapes':{k:tuple(v.shape) for k,v in b.items()}}
    def _validate(self,actor,rows,norm):
        vals=[]; auc=[]
        for _,is_train,supp,sr,query,qr in rows:
            if is_train: continue
            hi=[]; hr=[]; cand=list(supp); h=actor.init_hidden(1,self.device)
            for step in range(min(self.max_steps,len(cand))):
                st=build_mirt_state(self.mirt,hi,hr,step,self.max_steps,self.theta_cfg,self.device)
                with torch.no_grad(): a,h=actor.forward_step(st,h)
                it=nearest_item(a.squeeze(0),cand,self.mirt,norm,self.device); cand.remove(it); hi.append(it); hr.append(sr[supp.index(it)])
            th=fit_student_theta(self.mirt,hi,hr,device=self.device,**self.theta_cfg); p=predict_with_theta(self.mirt,th,query).detach().cpu().tolist(); mb=metric_bundle(qr,p); vals.append(mb['nll']); auc.append(mb['auc'])
        return {'validation_query_nll':float(np.nanmean(vals)) if vals else float('nan'),'validation_query_auc':float(np.nanmean(auc)) if auc else float('nan')}
    def train(self):
        start=time.time(); rows=self._split(self._load_sequences(self.config['assets'].get('train_valid_sequences','kc_level/train_valid_sequences.csv'))); norm=compute_action_normalizer(self.mirt, sorted({i for _,_,s,_,_,_ in rows for i in s}), self.device)
        actor=MIRTRecurrentActor(**self.config.get('model',{})).to(self.device); ta=MIRTRecurrentActor(**self.config.get('model',{})).to(self.device); ta.load_state_dict(actor.state_dict())
        critic=MIRTRecurrentCritic(**self.config.get('model',{}), q_clip=self.q_clip).to(self.device); tc=MIRTRecurrentCritic(**self.config.get('model',{}), q_clip=self.q_clip).to(self.device); tc.load_state_dict(critic.state_dict())
        ao=torch.optim.Adam(actor.parameters(),lr=float(self.config['training'].get('actor_lr',3e-5))); co=torch.optim.Adam(critic.parameters(),lr=float(self.config['training'].get('critic_lr',1e-5))); rb=SequenceReplayBuffer(self.config.get('sequence_replay',{}).get('capacity_episodes',5000),self.seed)
        hist=[]; best=float('inf'); optn=0; last={}
        for ep in range(1,self.epochs+1):
            rewards=[]; uniq=set()
            for _,is_train,supp,sr,query,qr in rows:
                if not is_train: continue
                hi=[]; hr=[]; cand=list(supp); h=actor.init_hidden(1,self.device); S=[];A=[];R=[];NS=[];D=[]; prev=None
                for step in range(min(self.max_steps,len(cand))):
                    st=build_mirt_state(self.mirt,hi,hr,step,self.max_steps,self.theta_cfg,self.device); prev=query_nll(self.mirt,fit_student_theta(self.mirt,hi,hr,device=self.device,**self.theta_cfg),query,qr) if prev is None else prev
                    with torch.no_grad(): act,h=actor.forward_step(st,h); act=act.squeeze(0)+torch.randn(37,device=self.device)*0.1
                    it=nearest_item(act,cand,self.mirt,norm,self.device); uniq.add(it); cand.remove(it); hi2=hi+[it]; hr2=hr+[sr[supp.index(it)]]; cur=query_nll(self.mirt,fit_student_theta(self.mirt,hi2,hr2,device=self.device,**self.theta_cfg),query,qr); rw=nll_drop_reward(prev,cur,self.config['training'].get('reward_scale',10.0),self.config['training'].get('reward_clip',5.0)); ns=build_mirt_state(self.mirt,hi2,hr2,step+1,self.max_steps,self.theta_cfg,self.device); done=float(step+1>=self.max_steps or not cand)
                    S.append(st.cpu().numpy()); A.append(act.detach().cpu().numpy()); R.append(rw); NS.append(ns.cpu().numpy()); D.append(done); rewards.append(rw); hi,hr,prev=hi2,hr2,cur
                if S: rb.add_episode(S,A,R,NS,D)
                if len(rb)>=self.min_eps:
                    for _ in range(self.updates_per_episode): last=self._optimize(rb,actor,critic,ta,tc,ao,co,optn); optn+=1; self._assert_frozen()
            vm=self._validate(actor,rows,norm); rec={'epoch':ep,'mean_reward':float(np.mean(rewards)) if rewards else 0.0,'selected_unique_items':len(uniq),'replay_episode_count':len(rb),**{k:v for k,v in last.items() if k!='shapes'},**vm}; hist.append(rec)
            if rec['validation_query_nll']<best: best=rec['validation_query_nll']; self._save(actor,norm,ep,rec,'rdpg_mirt_actor_best.pt')
        self._save(actor,norm,self.epochs,hist[-1] if hist else {},'rdpg_mirt_actor_final.pt'); torch.save({'training_history':hist,'last_batch_shapes':last.get('shapes'),'smoke_seconds':time.time()-start},self.out/'rdpg_mirt_training_state.pt')
        if hist:
            with (self.out/'training_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
        return hist
    def _save(self,actor,norm,epoch,metrics,name):
        try: git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        except Exception: git=None
        torch.save({'actor_state_dict':actor.state_dict(),'actor_architecture':ACTOR_ARCHITECTURE,'state_definition_version':STATE_DEFINITION_VERSION,'hidden_dim':actor.hidden_dim,'burn_in_length':self.burn,'unroll_length':self.unroll,'theta_fit':self.theta_cfg,'action_mean':norm.mean,'action_std':norm.std,'training_config':self.config,'validation_metrics':metrics,'epoch':epoch,'git_commit':git}, self.out/name)
