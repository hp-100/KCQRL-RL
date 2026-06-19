from evaluation.benchmark import BenchmarkV2Evaluator

def test_synthetic_smoke(tmp_path):
    cfg={'benchmark': {'protocol':'benchmark_v2','seeds':[42], 'steps':[0,1], 'max_students':3, 'output_dir': str(tmp_path)}, 'assets': {'base_dir': '/missing', 'q_matrix':'q.pt','item_bank':'i.npy','test_sequences':'t.csv','ncdm_checkpoint':'n.pt'}}
    rows=BenchmarkV2Evaluator(cfg, debug=True, ddpg_checkpoint=str(tmp_path/'missing.pt')).run()
    assert rows
    counts={(r['policy'], r['step']): (r['students'], r['query_interactions']) for r in rows}
    per_step={}
    for (pol,step), val in counts.items(): per_step.setdefault(step,set()).add(val)
    assert all(len(v)==1 for v in per_step.values())
