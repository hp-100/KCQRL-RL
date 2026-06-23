#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, logging, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
ROOT = Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw/conic10k'; OUT=ROOT/'data/interim'; DOC=ROOT/'docs/DATA_AUDIT_REPORT.md'
REQ=['text','process','answer_expressions','fact_expressions','query_expressions','fact_spans','query_spans']
log=logging.getLogger('inspect_dataset')
def rows(raw:Path):
    for p in sorted(raw.glob('*.jsonl')):
        if p.name=='metadata.json': continue
        split=p.stem
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines()):
            if line.strip():
                r=json.loads(line); r['_split']=split; r['_row']=i; yield r
def norm_ws(s:str)->str: return re.sub(r'[ \t\r\n\f\v]+',' ',s).strip()
def dist(xs):
    xs=sorted(xs); n=len(xs)
    if not xs: return {'count':0}
    def q(p): return xs[min(n-1, int((n-1)*p))]
    return {'count':n,'min':xs[0],'p25':q(.25),'median':q(.5),'p75':q(.75),'max':xs[-1],'mean':round(mean(xs),2)}
def inspect(raw:Path=RAW,out:Path=OUT)->dict[str,Any]:
    data=list(rows(raw)); total=len(data)
    if not total: raise RuntimeError(f'no JSONL rows found in {raw}')
    fields=sorted({k for r in data for k in r if not k.startswith('_')})
    missing={f:sum(1 for r in data if r.get(f) in (None,'',[])) for f in fields}
    types={f:sorted({type(r.get(f)).__name__ for r in data if f in r}) for f in fields}
    texts=[str(r.get('text','')) for r in data]; procs=[str(r.get('process','')) for r in data]
    exact=sum(c-1 for c in Counter(texts).values() if c>1); norm=sum(c-1 for c in Counter(norm_ws(t) for t in texts).values() if c>1)
    bad=sum(1 for t in texts+procs if '\ufffd' in t)
    report={'total_rows':total,'splits':dict(Counter(r['_split'] for r in data)),'fields':fields,'field_types':types,
            'missing':{f:{'count':c,'ratio':round(c/total,6)} for f,c in missing.items()},
            'required_fields':{f:f in fields for f in REQ},'text_length':dist([len(t) for t in texts]),
            'process_length':dist([len(p) for p in procs]),'duplicate_text_exact_extra_rows':exact,
            'duplicate_text_normalized_whitespace_extra_rows':norm,'empty_answer':sum(1 for r in data if not r.get('answer_expressions')),
            'empty_process':sum(1 for p in procs if not p.strip()),'short_process_lt_20':sum(1 for p in procs if len(p.strip())<20),
            'replacement_character_rows':bad}
    out.mkdir(parents=True,exist_ok=True)
    (out/'conic10k_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'conic10k_missing.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['field','missing_count','missing_ratio','types'])
        for fld in fields: w.writerow([fld,report['missing'][fld]['count'],report['missing'][fld]['ratio'],';'.join(types[fld])])
    md=['# Conic10K Data Audit Report','',f'- Total rows: {total}',f'- Splits: {report["splits"]}',f'- Required fields present: {report["required_fields"]}',f'- Text length: {report["text_length"]}',f'- Process length: {report["process_length"]}',f'- Exact duplicate text extra rows: {exact}',f'- Whitespace-normalized duplicate text extra rows: {norm}',f'- Empty answer rows: {report["empty_answer"]}',f'- Empty process rows: {report["empty_process"]}',f'- Short process (<20 chars): {report["short_process_lt_20"]}',f'- Replacement-character rows: {bad}']
    (out/'conic10k_audit.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); DOC.write_text('\n'.join(md)+'\n',encoding='utf-8')
    return report
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=RAW); p.add_argument('--output-dir',type=Path,default=OUT); p.add_argument('--log-level',default='INFO')
    a=p.parse_args(); logging.basicConfig(level=a.log_level)
    try: print(json.dumps(inspect(a.input_dir,a.output_dir),ensure_ascii=False,indent=2))
    except Exception: log.exception('inspection failed'); sys.exit(1)
if __name__=='__main__': main()
