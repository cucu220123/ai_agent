"""Download a pinned public dataset, deduplicate, and freeze a stratified split."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

URL = 'https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip'
ARCHIVE_SHA256 = 'afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343'
FILES = ('amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt')


def prepare(output: Path, archive_path: Path | None = None) -> dict:
    raw = archive_path.read_bytes() if archive_path else urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('dataset archive hash mismatch; inspect upstream change before accepting')
    records = []
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / 'raw'
    raw_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in FILES:
            content = archive.read('sentiment labelled sentences/' + name)
            (raw_dir / name).write_bytes(content)
            # Split at the final TAB. CSV quote parsing loses IMDB rows.
            # Only LF is a record delimiter; IMDB contains Unicode line-separator
            # characters within sentences which str.splitlines() wrongly splits.
            for number, line in enumerate(content.decode('utf-8').rstrip('\n').split('\n'), 1):
                line = line.rstrip('\r')
                text, label = line.rsplit('\t', 1)
                if not text.strip() or label not in {'0', '1'}:
                    raise ValueError(f'invalid record {name}:{number}')
                key = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
                records.append({'text': text, 'label': int(label), 'source': name, 'line': number, 'group': key})
    frame = pd.DataFrame(records)
    conflicts = frame.groupby('group').label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    clean = frame[~frame.group.isin(conflicting)].drop_duplicates('group').copy()
    development, final = train_test_split(clean, test_size=.25, random_state=20260908, stratify=clean.source + ':' + clean.label.astype(str))
    assert set(development.group).isdisjoint(final.group)
    manifest = {'source_url': URL, 'source_doi': 'https://doi.org/10.24432/C57604', 'creator': 'Dimitrios Kotzias (2015)', 'license': 'CC BY 4.0', 'archive_sha256': ARCHIVE_SHA256, 'original_rows': len(frame), 'unique_rows': len(clean), 'conflicting_groups_removed': len(conflicting), 'split_seed': 20260908, 'policy': 'NFKC/casefold/whitespace duplicate groups removed before source+label stratification; final 25% withheld from all agents', 'splits': {}}
    for name, part in [('development', development), ('final_test', final)]:
        path = output / f'{name}.csv'
        part[['text', 'label']].to_csv(path, index=False, lineterminator='\n')
        part[['source', 'line', 'label']].to_json(output / f'{name}_provenance.json', orient='records', indent=2)
        manifest['splits'][name] = {'rows': len(part), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'class_counts': {str(k): int(v) for k, v in part.label.value_counts().items()}, 'source_counts': part.source.value_counts().to_dict()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/uci_sentiment'))
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.archive), indent=2))
