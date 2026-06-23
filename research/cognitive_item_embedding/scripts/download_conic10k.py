#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'data/raw/conic10k'
DATASET = 'WenyangHui/Conic10K'
REQUIRED = ['text','process','answer_expressions','fact_expressions','query_expressions','fact_spans','query_spans']
log = logging.getLogger('download_conic10k')

def _write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def _load_hf() -> tuple[dict[str, list[dict]], dict]:
    import importlib
    dsmod = importlib.import_module('datasets')
    if not hasattr(dsmod, 'load_dataset'):
        raise RuntimeError('installed/imported datasets module does not provide load_dataset')
    data = dsmod.load_dataset(DATASET)
    info = getattr(data, 'info', None)
    meta = {'source': f'huggingface:{DATASET}', 'dataset_version': str(getattr(info, 'version', 'unknown')),
            'license': str(getattr(info, 'license', 'MIT (per official repository)'))}
    return {k: [dict(x) for x in v] for k, v in data.items()}, meta

def _load_github() -> tuple[dict[str, list[dict]], dict]:
    # Official repo is the documented fallback. The exact file names have changed historically,
    # so try a small set of common JSON/JSONL locations.
    base = 'https://raw.githubusercontent.com/whyNLP/Conic10K/main/conic10k'
    splits: dict[str, list[dict]] = {}
    tried = []
    for split in ('train','dev','validation','valid','test'):
        for ext in ('jsonl','json'):
            url = f'{base}/{split}.{ext}'; tried.append(url)
            try:
                text = urlopen(url, timeout=30).read().decode('utf-8')
            except Exception:
                continue
            if ext == 'jsonl':
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                obj = json.loads(text); rows = obj if isinstance(obj, list) else obj.get('data', [])
            splits['validation' if split in ('valid','dev') else split] = rows
            break
    if not splits:
        raise RuntimeError('could not fetch Conic10K from Hugging Face or official GitHub; tried ' + ', '.join(tried))
    return splits, {'source': 'github:whyNLP/Conic10K', 'dataset_version': 'main branch (commit unavailable via raw fallback)', 'license': 'MIT'}

def download(out: Path = DEFAULT_OUT, force: bool = False) -> dict:
    if out.exists() and any(out.glob('*.jsonl')) and not force:
        log.info('data already exists at %s; use --force to refresh', out)
        return json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    if force and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        splits, meta = _load_hf()
    except Exception as e:
        log.warning('Hugging Face load failed: %s; trying official GitHub fallback', e)
        splits, meta = _load_github()
    counts = {k: len(v) for k, v in splits.items()}
    for split, rows in splits.items(): _write_jsonl(rows, out / f'{split}.jsonl')
    meta.update({'download_time_utc': datetime.now(timezone.utc).isoformat(), 'splits': counts,
                 'total_rows': sum(counts.values()), 'required_fields': REQUIRED})
    (out/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    for k, n in counts.items(): print(f'{k}\t{n}')
    return meta

def main():
    p=argparse.ArgumentParser(description='Download Conic10K into JSONL split files.')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT); p.add_argument('--force', action='store_true')
    p.add_argument('--log-level', default='INFO')
    a=p.parse_args(); logging.basicConfig(level=a.log_level, format='%(levelname)s:%(message)s')
    try: download(a.output_dir, a.force)
    except Exception as e: log.exception('download failed'); sys.exit(1)
if __name__ == '__main__': main()
