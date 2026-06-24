"""Small profiler for Base and Set C3DQN variants."""
from __future__ import annotations
import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork

def _batch(k=16,c=64,device='cpu'):
    h=torch.rand(4,8,2*k+3,device=device); hm=torch.ones(4,8,dtype=torch.bool,device=device)
    cf=torch.rand(4,c,2*k+1,device=device); cm=torch.ones(4,c,dtype=torch.bool,device=device); g=torch.rand(4,2*k+1,device=device); cov=torch.zeros(4,k,device=device)
    return dict(history_features=h,history_mask=hm,candidate_features=cf,candidate_mask=cm,global_features=g,coverage_count=cov)

def _sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()

def measure(model,b,training=False,repeats=30,warmup=10):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3) if training else None
    for _ in range(warmup):
        if training: opt.zero_grad(); q,_=model(**b); loss=q[q>-1e8].mean(); loss.backward(); opt.step()
        else:
            with torch.no_grad(): model(**b)
    _sync(next(model.parameters()).device); t=time.perf_counter()
    for _ in range(repeats):
        if training: opt.zero_grad(); q,_=model(**b); loss=q[q>-1e8].mean(); loss.backward(); opt.step()
        else:
            with torch.no_grad(): model(**b)
    _sync(next(model.parameters()).device); return (time.perf_counter()-t)/repeats

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'; b=_batch(device=device); k=16
    variants=[('Base',CandidateConditionedNCDMQNetwork(k,dropout=0).to(device),{x:y for x,y in b.items() if x!='coverage_count'}),('Set-none',SetConditionedNCDMQNetwork(k,dropout=0,candidate_set_encoder='none').to(device),b),('Set-ISAB-M8',SetConditionedNCDMQNetwork(k,dropout=0,candidate_set_encoder='isab',num_inducing_points=8).to(device),b),('Set-ISAB-M16',SetConditionedNCDMQNetwork(k,dropout=0,candidate_set_encoder='isab',num_inducing_points=16).to(device),b),('Full Attention',SetConditionedNCDMQNetwork(k,dropout=0,candidate_set_encoder='full_self_attention').to(device),b)]
    for name,m,bb in variants:
        print(name, 'forward_s', measure(m.eval(),bb,False), 'training_step_s', measure(m.train(),bb,True), 'memory', 'N/A' if device=='cpu' else torch.cuda.max_memory_allocated())
    s=variants[2][1].eval(); q,_=s(**b); qc,_=s.forward_chunked(**b,chunk_size=16); torch.testing.assert_close(q,qc); print('chunked equivalence ok')
if __name__=='__main__': main()
