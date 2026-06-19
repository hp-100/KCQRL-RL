import csv
import numpy as np
import torch
import pytest

from evaluation.benchmark import BenchmarkV2Evaluator
from evaluation.protocol import valid_item_count
from models.mirt import MIRTModel


def _save_mirt(path, n_items=12, dim=3):
    m=MIRTModel(2,n_items,dim)
    torch.save(m.state_dict(), path)
    return m


def _write_sequences(path, n_items=12):
    with path.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['student_id','item_id','response'])
        for s in range(3):
            for i in range(n_items):
                w.writerow([s, i, (i+s)%2])


def test_mirt_native_valid_count_ignores_item_bank_and_ncdm():
    m=MIRTModel(1,7652,2)
    assert valid_item_count(np.zeros((8000,2)), np.zeros((7052,128)), None, m, track='mirt_native') == 7652


def test_mirt_native_does_not_require_or_load_ncdm_or_item_bank(tmp_path, monkeypatch):
    mirt_path=tmp_path/'mirt.pt'; seq_path=tmp_path/'test.csv'; q_path=tmp_path/'q.npy'
    _save_mirt(mirt_path, n_items=12, dim=3); _write_sequences(seq_path, 12); np.save(q_path, np.zeros((12,3), dtype=np.float32))
    import evaluation.benchmark as bench
    def boom(*args, **kwargs):
        raise AssertionError('NCDM checkpoint should not be loaded for mirt_native')
    monkeypatch.setattr(bench, 'safe_load_ncdm_checkpoint', boom)
    cfg={'device':'cpu','benchmark':{'track':'mirt_native','policies':['Random-MIRT'],'seeds':[1],'steps':[0],'max_students':2,'min_query_items':2,'output_dir':str(tmp_path/'out'),'mirt':{'theta_steps':3}},'assets':{'base_dir':str(tmp_path),'mirt_checkpoint':'mirt.pt','test_sequences':'test.csv','q_matrix':'q.npy','item_bank':'missing.npy','ncdm_checkpoint':'missing.pt'}}
    rows=BenchmarkV2Evaluator(cfg, track='mirt_native').run()
    assert {r['policy'] for r in rows} == {'Random-MIRT'}
