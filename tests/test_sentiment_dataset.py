import hashlib
import json
from pathlib import Path
import pandas as pd
from app.validation.evaluation import feature_fingerprints


def test_public_sentiment_split_preserves_source_records_and_has_no_feature_overlap():
    root = Path(__file__).parents[1] / 'data/uci_sentiment'
    manifest = json.loads((root / 'manifest.json').read_text())
    raw_count = sum(len(p.read_text().rstrip('\n').split('\n')) for p in (root / 'raw').glob('*_labelled.txt'))
    assert raw_count == manifest['original_rows'] == 3000
    frames = {}
    for split in ('development', 'final_test'):
        path = root / f'{split}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
        frame = pd.read_csv(path); frames[split] = frame
        assert len(frame) == manifest['splits'][split]['rows']
        provenance = json.loads((root / f'{split}_provenance.json').read_text())
        assert len(provenance) == len(frame)
        for i, record in enumerate(provenance):
            original = (root / 'raw' / record['source']).read_text().split('\n')[record['line'] - 1]
            text, label = original.rsplit('\t', 1)
            assert frame.iloc[i]['text'] == text and frame.iloc[i]['label'] == int(label)
    assert not (feature_fingerprints(frames['development'], 'label') & feature_fingerprints(frames['final_test'], 'label'))
    assert sum(len(v) for v in frames.values()) == manifest['unique_rows']
