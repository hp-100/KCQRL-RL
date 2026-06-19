import json, torch
from evaluation.policies.mirt_policy import FormalMIRTPolicy
from models.mirt import MIRTModel, fit_student_theta, predict_with_theta
from evaluation.benchmark import BenchmarkV2Evaluator

def make_model():
    m=MIRTModel(1,5,2); m.disc_emb.weight.data=torch.tensor([[1.,0.],[2.,0.],[0.,3.],[.1,0.],[0.,.1]]); m.diff_emb.weight.data.zero_(); return m.eval()

def test_trace_mfi_formula_matches_manual():
    m=make_model(); pol=FormalMIRTPolicy('MIRT-Trace-MFI',m,theta_cfg={'steps':1}); th=torch.tensor([0.2,-0.1])
    ids=[0,1,2]; scores=pol.trace_mfi_scores(th,ids)
    p=predict_with_theta(m,th,torch.tensor(ids)); a=m.disc_emb(torch.tensor(ids))
    assert torch.allclose(scores, p*(1-p)*a.pow(2).sum(1))

def test_d_opt_prefers_larger_logdet_gain():
    m=make_model(); pol=FormalMIRTPolicy('MIRT-D-opt',m,d_opt_ridge=.01,theta_cfg={'steps':1}); scores=pol.d_opt_scores(torch.zeros(2),[3,2],[])
    assert int(torch.argmax(scores)) == 1

def test_mkli_seed_reproducible():
    m=make_model(); pol=FormalMIRTPolicy('MIRT-MKLI',m,mkli_samples=8,mkli_scale=.2,theta_cfg={'steps':1}); th=torch.zeros(2)
    assert torch.allclose(pol.mkli_scores(th,[0,1,2]), pol.mkli_scores(th,[0,1,2]))

def test_policy_metadata_and_no_query_access():
    m=make_model(); pol=FormalMIRTPolicy('MIRT-Trace-MFI',m,theta_cfg={'steps':2})
    assert pol.metadata.implementation == 'formal_mirt_trace_fisher'
    assert 'selection_model=mirt' in pol.metadata.notes and 'uses_query_labels=false' in pol.metadata.notes
    item=pol.select([0,1],[2],[1.0],{'query_item_ids':[3],'query_responses':[0.0]})
    assert item in [0,1]

def test_benchmark_synthetic_smoke_with_mirt_and_ddpg(tmp_path):
    cfg={'device':'cpu','benchmark':{'policies':['Random','MIRT-Trace-MFI','MIRT-D-opt','MIRT-MKLI','DDPG'],'seeds':[42],'max_students':2,'steps':[0,1],'output_dir':str(tmp_path),'min_query_items':2,'mirt':{'theta_steps':2,'mkli_samples':4}},'assets':{'base_dir':'/missing','q_matrix':'q.pt','item_bank':'i.npy','test_sequences':'t.csv','ncdm_checkpoint':'n.pt'}}
    rows=BenchmarkV2Evaluator(cfg,debug=True,ddpg_checkpoint=str(tmp_path/'missing.pt')).run()
    assert {'Random','MIRT-Trace-MFI','MIRT-D-opt','MIRT-MKLI','DDPG'} <= {r['policy'] for r in rows}
    md=json.loads((tmp_path/'policy_metadata.json').read_text())
    assert md['MIRT-D-opt']['implementation']=='formal_mirt_d_opt'

def test_mirt_item_bounds_filter(tmp_path):
    from evaluation.protocol import valid_item_count
    class N: pass
    n=N(); n.k_difficulty=torch.nn.Embedding(10,2); n.e_discrimination=torch.nn.Embedding(9,1)
    assert valid_item_count(torch.zeros(8,2), torch.zeros(7,3), n, make_model()) == 5
