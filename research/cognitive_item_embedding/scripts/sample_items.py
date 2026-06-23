#!/usr/bin/env python3
"""Create a fixed 100-item review sample from Conic10K."""
from __future__ import annotations
import argparse, csv, json, random, html
from pathlib import Path
from inspect_dataset import audit, load_hf_dataset, DATASET_ID
REVIEW_COLUMNS = ["review_keep", "review_issue", "review_notes"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-id", default=DATASET_ID); ap.add_argument("--output-dir", type=Path, default=Path("artifacts/conic10k_sample")); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=20240623); args=ap.parse_args()
    stats, rows = audit(load_hf_dataset(args.dataset_id), args.dataset_id)
    if stats["total_items"] < args.n: raise SystemExit("dataset smaller than sample")
    sample = random.Random(args.seed).sample(rows, args.n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields=["sample_index","id","split","text","answer","formal",*REVIEW_COLUMNS]
    csv_path=args.output_dir/"conic10k_100_item_review_sample.csv"; jsonl_path=args.output_dir/"conic10k_100_item_review_sample.jsonl"; html_path=args.output_dir/"conic10k_100_item_review_sample.html"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fields); w.writeheader();
        for i,r in enumerate(sample,1): w.writerow({**{k:r.get(k,"") for k in ["id","split","text","answer","formal"]}, "sample_index":i, **{c:"" for c in REVIEW_COLUMNS}})
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i,r in enumerate(sample,1): f.write(json.dumps({**{k:r.get(k,"") for k in ["id","split","text","answer","formal"]}, "sample_index":i, **{c:"" for c in REVIEW_COLUMNS}}, ensure_ascii=False)+"\n")
    trs=[]
    for i,r in enumerate(sample,1):
        tds="".join(f"<td>{html.escape(str(v))}</td>" for v in [i,r['id'],r['split'],r['text'],r['answer'],r['formal'],"","",""])
        trs.append(f"<tr>{tds}</tr>")
    html_path.write_text("<html><meta charset='utf-8'><body><table border='1'><tr>"+"".join(f"<th>{c}</th>" for c in fields)+"</tr>"+"\n".join(trs)+"</table></body></html>", encoding="utf-8")
    print(json.dumps({"stats":stats,"files":[str(csv_path),str(jsonl_path),str(html_path)]}, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
