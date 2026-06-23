from __future__ import annotations
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspect_dataset import inspect
from sample_items import sample

def make_raw(tmp_path: Path) -> Path:
    raw=tmp_path/'raw'; raw.mkdir()
    rows=[]
    for i in range(120):
        rows.append({'text':f'题干 $x_{i}$  保持LaTeX','process':'解析步骤 '+('很长'* (i%9+1)),'answer_expressions':[str(i)],'fact_expressions':['f'], 'query_expressions':['q'], 'fact_spans':[], 'query_spans':[]})
    with (raw/'train.jsonl').open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return raw

def test_inspect_required_fields_and_load(tmp_path):
    report=inspect(make_raw(tmp_path), tmp_path/'out')
    assert report['total_rows']==120
    assert all(report['required_fields'].values())
    assert (tmp_path/'out/conic10k_audit.json').exists()

def test_sampling_reproducible_outputs_and_preserves_latex(tmp_path):
    raw=make_raw(tmp_path); out1=tmp_path/'s1'; out2=tmp_path/'s2'
    a=sample(raw,out1,100,20260623); b=sample(raw,out2,100,20260623)
    assert [r['item_id'] for r in a] == [r['item_id'] for r in b]
    assert (out1/'conic10k_sample_100.csv').exists() and (out1/'conic10k_sample_100.jsonl').exists() and (out1/'conic10k_sample_100.html').exists()
    with (out1/'conic10k_sample_100.csv').open(encoding='utf-8-sig') as f:
        first=next(csv.DictReader(f))
    assert '$x_' in first['text'] and '  ' in first['text']
    for col in ['valid_high_school_item','solution_correct','review_notes']:
        assert first[col] == ''

def test_raw_gitignore_present():
    gi=Path(__file__).resolve().parents[1]/'.gitignore'
    assert 'data/raw/' in gi.read_text(encoding='utf-8')
