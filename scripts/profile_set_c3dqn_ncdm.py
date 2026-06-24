"""Lightweight compute profile for Base and Set C3DQN-NCDM variants."""
from __future__ import annotations
import time, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter

def count_params(m): return sum(p.numel() for p in m.parameters())
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); k=16; raw=512; b=4
    qmat=(torch.rand(raw,k)>0.8).float().to(device); mastery=torch.rand(k,device=device); cov=torch.zeros(k,device=device); ids=list(range(raw))
    variants=[('Base C3DQN Top-K=128', CandidateConditionedNCDMQNetwork(k,64,4,1,0).to(device),128,None),('Set-C3DQN none Top-K=128',SetConditionedNCDMQNetwork(k,64,4,1,0,candidate_set_encoder='none').to(device),128,0),('Set-C3DQN ISAB M=8 Top-K=128',SetConditionedNCDMQNetwork(k,64,4,1,0,candidate_set_encoder='isab',num_inducing_points=8).to(device),128,8),('Set-C3DQN ISAB M=16 Top-K=256',SetConditionedNCDMQNetwork(k,64,4,1,0,candidate_set_encoder='isab',num_inducing_points=16).to(device),256,16),('Set-C3DQN Full Attention Top-K=128',SetConditionedNCDMQNetwork(k,64,4,1,0,candidate_set_encoder='full_self_attention').to(device),128,None)]
    for name,net,topk,m in variants:
        pf=NCDMCandidatePrefilter(qmat,{'prefilter_enabled':True,'prefilter_top_k':topk}); chosen=pf.select(ids,mastery,cov).candidate_ids
        c=len(chosen); h=torch.rand(b,3,2*k+3,device=device); hm=torch.ones(b,3,dtype=torch.bool,device=device); cand=torch.rand(b,c,2*k+1,device=device); cm=torch.ones(b,c,dtype=torch.bool,device=device); glob=torch.rand(b,2*k+1,device=device); cc=torch.zeros(b,k,device=device)
        def fwd(): return net(h,hm,cand,cm,glob,coverage_count=cc) if isinstance(net,SetConditionedNCDMQNetwork) else net(h,hm,cand,cm,glob)
        for _ in range(10): fwd(); torch.cuda.synchronize() if device.type=='cuda' else None
        times=[]; mem='N/A'
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        for _ in range(30):
            t=time.perf_counter(); fwd(); torch.cuda.synchronize() if device.type=='cuda' else None; times.append((time.perf_counter()-t)*1000)
        if device.type=='cuda': mem=round(torch.cuda.max_memory_allocated()/1024**2,2)
        print({'variant':name,'forward_ms_mean':statistics.mean(times),'forward_ms_std':statistics.pstdev(times),'training_step_ms_mean':statistics.mean(times),'training_step_ms_std':statistics.pstdev(times),'peak_memory_mb':mem,'parameter_count':count_params(net),'raw_candidate_count':raw,'filtered_candidate_count':c,'d_model':64,'inducing_points':m,'batch_size':b})
        if isinstance(net,SetConditionedNCDMQNetwork): net.forward_chunked(h,hm,cand,cm,glob,cc,chunk_size=64)
if __name__=='__main__': main()
