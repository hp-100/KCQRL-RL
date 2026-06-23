#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, html, json, logging, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw/conic10k'; OUT=ROOT/'data/samples'; SEED=20260623
BASE=['item_id','split','text','process','answer_expressions','fact_expressions','query_expressions','text_length','process_length','has_complete_process']
REVIEW=['valid_high_school_item','solution_correct','solution_complete','core_knowledge_identifiable','core_strategy_identifiable','multiple_solution_paths','requires_missing_figure','possible_surface_similar_pair','possible_cognitive_similar_pair','review_notes']
def load(raw:Path):
    out=[]
    for p in sorted(raw.glob('*.jsonl')):
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines()):
            if line.strip():
                r=json.loads(line); r['split']=p.stem; r['_idx']=i; out.append(r)
    return out
def stable_id(r):
    return 'conic10k_'+hashlib.sha1(f"{r.get('split')}\0{r.get('_idx')}\0{r.get('text','')}".encode('utf-8')).hexdigest()[:12]
def bucket(v, qs): return sum(v>q for q in qs)
def sample(raw:Path=RAW,out:Path=OUT,n:int=100,seed:int=SEED):
    data=load(raw)
    if len(data)<n: raise RuntimeError(f'need at least {n} rows, found {len(data)}')
    tl=sorted(len(str(r.get('text',''))) for r in data); pl=sorted(len(str(r.get('process',''))) for r in data)
    qs_t=[tl[int(len(tl)*q)] for q in (.25,.5,.75)]; qs_p=[pl[int(len(pl)*q)] for q in (.25,.5,.75)]
    strata={}
    for r in data: strata.setdefault((bucket(len(str(r.get('text',''))),qs_t), bucket(len(str(r.get('process',''))),qs_p)), []).append(r)
    rng=random.Random(seed); chosen=[]
    keys=sorted(strata)
    while len(chosen)<n and keys:
        for k in list(keys):
            pool=strata[k]
            if pool: chosen.append(pool.pop(rng.randrange(len(pool))))
            else: keys.remove(k)
            if len(chosen)==n: break
    rng.shuffle(chosen)
    rows=[]
    for r in chosen:
        row={k:'' for k in BASE+REVIEW}; row.update({
            'item_id':stable_id(r),'split':r.get('split',''),'text':r.get('text',''),'process':r.get('process',''),
            'answer_expressions':json.dumps(r.get('answer_expressions',''),ensure_ascii=False),
            'fact_expressions':json.dumps(r.get('fact_expressions',''),ensure_ascii=False),
            'query_expressions':json.dumps(r.get('query_expressions',''),ensure_ascii=False),
            'text_length':len(str(r.get('text',''))),'process_length':len(str(r.get('process',''))),
            'has_complete_process':bool(str(r.get('process','')).strip())})
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True); fields=BASE+REVIEW
    with (out/'conic10k_sample_100.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    with (out/'conic10k_sample_100.jsonl').open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    parts=['<html><head><meta charset="utf-8"><style>body{font-family:system-ui,"Noto Sans CJK SC",sans-serif} table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}.tex{white-space:pre-wrap}</style></head><body><h1>Conic10K 100-item review sample</h1><table><tr>'+''.join(f'<th>{html.escape(f)}</th>' for f in fields)+'</tr>']
    for r in rows: parts.append('<tr>'+''.join(f'<td class="tex">{html.escape(str(r[f]))}</td>' for f in fields)+'</tr>')
    parts.append('</table></body></html>'); (out/'conic10k_sample_100.html').write_text('\n'.join(parts),encoding='utf-8')
    return rows
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=RAW); p.add_argument('--output-dir',type=Path,default=OUT); p.add_argument('--seed',type=int,default=SEED); p.add_argument('--n',type=int,default=100)
    a=p.parse_args(); logging.basicConfig(level='INFO')
    try: print(f'wrote {len(sample(a.input_dir,a.output_dir,a.n,a.seed))} rows to {a.output_dir}')
    except Exception as e: logging.exception('sampling failed'); sys.exit(1)
if __name__=='__main__': main()
