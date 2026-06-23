"""Conic10K dataset download, audit, and review-sample generation utilities."""
from __future__ import annotations

import argparse, csv, hashlib, html, json, math, random, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

REQUIRED_FIELDS = ["text","process","answer_expressions","fact_expressions","query_expressions","fact_spans","query_spans"]
REVIEW_COLUMNS = ["valid_high_school_item","solution_correct","solution_complete","core_knowledge_identifiable","core_strategy_identifiable","multiple_solution_paths","requires_missing_figure","possible_surface_similar_pair","possible_cognitive_similar_pair","review_notes"]
SAMPLE_COLUMNS = ["item_id","split","text","process","answer_expressions","fact_expressions","query_expressions","text_length","process_length","has_complete_process",*REVIEW_COLUMNS]
DEFAULT_SEED = 20260623
REPLACEMENT = "\ufffd"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip()=="") or (isinstance(v,(list,dict)) and len(v)==0)

def value_type(v: Any) -> str:
    if v is None: return "null"
    if isinstance(v,bool): return "bool"
    if isinstance(v,int) and not isinstance(v,bool): return "int"
    if isinstance(v,float): return "float"
    if isinstance(v,str): return "str"
    if isinstance(v,list): return "list"
    if isinstance(v,dict): return "dict"
    return type(v).__name__

def load_jsonl(path: Path, split: str|None=None) -> list[dict[str,Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            item=json.loads(line)
            item.setdefault("split", split or path.stem)
            item.setdefault("item_id", item.get("id", f"{item['split']}-{i}"))
            rows.append(item)
    return rows

def load_json(path: Path, split: str|None=None) -> list[dict[str,Any]]:
    data=json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        rows=[]
        for sp, items in data.items():
            for i,item in enumerate(items):
                item=dict(item); item.setdefault("split", sp); item.setdefault("item_id", item.get("id", f"{sp}-{i}")); rows.append(item)
        return rows
    if isinstance(data, dict): data=data.get("data", data.get("items", []))
    rows=[]
    for i,item in enumerate(data):
        item=dict(item); item.setdefault("split", split or path.stem); item.setdefault("item_id", item.get("id", f"{item['split']}-{i}")); rows.append(item)
    return rows

def discover_dataset_files(root: Path) -> list[Path]:
    pats=["*.jsonl","*.json"]
    files=[]
    for p in pats: files.extend(root.rglob(p))
    return [f for f in files if not any(part.startswith(".") or part in {"artifacts","reports"} for part in f.parts)]

def load_dataset(data_dir: Path) -> list[dict[str,Any]]:
    rows=[]
    for path in discover_dataset_files(data_dir):
        if path.suffix==".jsonl": rows.extend(load_jsonl(path))
        elif path.suffix==".json": rows.extend(load_json(path))
    if not rows: raise FileNotFoundError(f"No JSON/JSONL dataset files found under {data_dir}")
    return rows

def quantiles(vals: list[int]) -> dict[str, float|int]:
    if not vals: return {k:0 for k in ["min","p25","median","p75","p90","p95","p99","max","mean"]}
    s=sorted(vals)
    def q(p): return s[min(len(s)-1, max(0, math.ceil(p*len(s))-1))]
    return {"min":s[0],"p25":q(.25),"median":q(.5),"p75":q(.75),"p90":q(.9),"p95":q(.95),"p99":q(.99),"max":s[-1],"mean":round(statistics.fmean(s),2)}

def bucket(value:int, cuts:list[int]) -> int:
    return sum(value>c for c in cuts)

def audit_rows(rows: list[dict[str,Any]]) -> dict[str,Any]:
    n=len(rows); fields=sorted({k for r in rows for k in r})
    split_counts=Counter(str(r.get("split","unknown")) for r in rows)
    field_types={f:dict(Counter(value_type(r.get(f)) for r in rows if f in r)) for f in fields}
    missing={f:{"count":sum(1 for r in rows if is_missing(r.get(f))),"ratio":(sum(1 for r in rows if is_missing(r.get(f)))/n if n else 0)} for f in fields}
    for f in REQUIRED_FIELDS:
        missing.setdefault(f,{"count":n,"ratio":1.0}); field_types.setdefault(f,{})
    text_l=[len(str(r.get("text") or "")) for r in rows]; proc_l=[len(str(r.get("process") or "")) for r in rows]
    exact=Counter(str(r.get("text") or "") for r in rows); ws=Counter(normalize_ws(str(r.get("text") or "")) for r in rows)
    return {"total_items":n,"split_counts":dict(split_counts),"fields":fields,"field_types":field_types,"missing":missing,
            "lengths":{"text":quantiles(text_l),"process":quantiles(proc_l)},
            "quality":{"empty_answers":sum(is_missing(r.get("answer_expressions")) for r in rows),"empty_process":sum(is_missing(r.get("process")) for r in rows),"short_process_lt_20":sum(len(str(r.get("process") or ""))<20 for r in rows),"replacement_character_items":sum(REPLACEMENT in str(r.get("text",""))+str(r.get("process","")) for r in rows)},
            "duplicates":{"exact_text_duplicate_groups":sum(1 for c in exact.values() if c>1),"exact_text_duplicate_items":sum(c for c in exact.values() if c>1),"whitespace_normalized_duplicate_groups":sum(1 for c in ws.values() if c>1),"whitespace_normalized_duplicate_items":sum(c for c in ws.values() if c>1)},
            "required_fields_present":{f: all(f in r for r in rows) for f in REQUIRED_FIELDS}}

def stratified_sample(rows: list[dict[str,Any]], k:int=100, seed:int=DEFAULT_SEED) -> list[dict[str,Any]]:
    text_lengths=[len(str(r.get("text") or "")) for r in rows]; proc_lengths=[len(str(r.get("process") or "")) for r in rows]
    tc=[quantiles(text_lengths)[x] for x in ["p25","median","p75"]]; pc=[quantiles(proc_lengths)[x] for x in ["p25","median","p75"]]
    strata=defaultdict(list)
    for r,tl,pl in zip(rows,text_lengths,proc_lengths): strata[(bucket(tl,tc),bucket(pl,pc))].append(r)
    rng=random.Random(seed); sample=[]; keys=sorted(strata)
    for key in keys:
        if len(sample)<min(k,len(rows)):
            sample.append(rng.choice(strata[key]))
    remaining=[r for key in keys for r in strata[key] if r not in sample]
    rng.shuffle(remaining); sample.extend(remaining[:max(0,min(k,len(rows))-len(sample))])
    out=[]
    for r in sample:
        proc=str(r.get("process") or "")
        item={"item_id":r.get("item_id") or hashlib.sha1(str(r.get("text","")).encode()).hexdigest()[:12],"split":r.get("split","unknown"),"text":r.get("text",""),"process":proc,"answer_expressions":r.get("answer_expressions",""),"fact_expressions":r.get("fact_expressions",""),"query_expressions":r.get("query_expressions",""),"text_length":len(str(r.get("text") or "")),"process_length":len(proc),"has_complete_process":bool(proc.strip()) and len(proc)>=20}
        item.update({c:"" for c in REVIEW_COLUMNS}); out.append(item)
    return out

def write_outputs(rows, out_dir:Path, sample_size:int, seed:int):
    out_dir.mkdir(parents=True, exist_ok=True); report=audit_rows(rows); sample=stratified_sample(rows,sample_size,seed)
    (out_dir/"conic10k_audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out_dir/"conic10k_review_sample_100.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=SAMPLE_COLUMNS); w.writeheader(); w.writerows(sample)
    with (out_dir/"conic10k_review_sample_100.jsonl").open("w",encoding="utf-8") as f:
        for r in sample: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    html_rows="\n".join("<tr>"+"".join(f"<td>{html.escape(str(r.get(c,'')))}</td>" for c in SAMPLE_COLUMNS)+"</tr>" for r in sample)
    (out_dir/"conic10k_review_sample_100.html").write_text("<html><meta charset='utf-8'><body><h1>Conic10K Review Sample</h1><table border='1'><thead><tr>"+"".join(f"<th>{c}</th>" for c in SAMPLE_COLUMNS)+"</tr></thead><tbody>"+html_rows+"</tbody></table></body></html>",encoding="utf-8")
    return report

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",type=Path,required=True); ap.add_argument("--out-dir",type=Path,default=Path("artifacts/conic10k")); ap.add_argument("--sample-size",type=int,default=100); ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    args=ap.parse_args(argv); rows=load_dataset(args.data_dir); report=write_outputs(rows,args.out_dir,args.sample_size,args.seed)
    print(json.dumps({"total_items":report["total_items"],"split_counts":report["split_counts"],"out_dir":str(args.out_dir)},ensure_ascii=False))
if __name__ == "__main__": main()
