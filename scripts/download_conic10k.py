"""Download Conic10K without committing raw data."""
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',type=Path,default=Path('data/conic10k'))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    try:
        from datasets import load_dataset
        ds=load_dataset('WenyangHui/Conic10K')
        for split in ds:
            ds[split].to_json(str(args.out_dir/f'{split}.jsonl'), force_ascii=False)
        print(f'Downloaded Conic10K via Hugging Face to {args.out_dir}'); return
    except Exception as exc:
        print(f'Hugging Face download failed, falling back to git archive: {exc}', file=sys.stderr)
    tmp=args.out_dir.parent/'_conic10k_repo'
    if tmp.exists(): shutil.rmtree(tmp)
    subprocess.run(['git','clone','--depth','1','https://github.com/whyNLP/Conic10K.git',str(tmp)],check=True)
    src=tmp/'conic10k'
    if not src.exists(): raise SystemExit('Cloned repository did not contain conic10k/')
    if args.out_dir.exists(): shutil.rmtree(args.out_dir)
    shutil.copytree(src,args.out_dir)
    shutil.rmtree(tmp)
    print(f'Downloaded Conic10K repository data to {args.out_dir}')
if __name__=='__main__': main()
