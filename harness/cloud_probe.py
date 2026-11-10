"""Portable selection and verification for remote probes, never candidate promotion."""
import hashlib
import json
from pathlib import Path
import re


def probe_selection(*, debug=False, fold=None, probe_samples=None, config=None):
    configured = ((config or {}).get('validation') or {}).get('sample_limit')
    if probe_samples is not None and (type(probe_samples) is not int or probe_samples <= 0):
        raise ValueError('Probe samples must be a positive integer')
    if fold is not None and (type(fold) is not int or fold < 0):
        raise ValueError('Probe fold must be a nonnegative integer')
    if configured is not None and (type(configured) is not int or configured <= 0):
        raise ValueError('Configured probe sample limit must be a positive integer')
    if not (debug or fold is not None or probe_samples is not None or configured is not None):
        return None
    limit = probe_samples if probe_samples is not None else 50 if debug else configured
    if limit is None or fold is None:
        raise ValueError('Remote probe requires both an explicit fold and a sample limit')
    return {'fold': fold, 'sample_limit': limit}


def probe_arguments(selection):
    selected = probe_selection(fold=selection['fold'], probe_samples=selection['sample_limit'])
    return ['--fold', str(selected['fold']), '--probe-samples', str(selected['sample_limit']), '--skip-leaderboard']


def probe_relative_path(exp_id, run_id):
    for value in (exp_id, run_id):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise ValueError('Unsafe remote probe identity')
    return Path('probes') / exp_id / run_id


def verify_probe_artifacts(directory, *, expected, selection):
    """Verify hashes, run kind, sample bounds and exact OOF coverage after download."""
    import csv
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'artifact_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('run_kind') != 'probe' or manifest.get('status') != 'COMPLETED':
        raise ValueError('Remote output is not a completed probe')
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Remote probe identity mismatch')
    files = manifest.get('files') or {}
    required = {'validation_partitions.json', 'oof_predictions.csv', 'metrics.json', 'run_config.yaml'}
    if not required <= set(files):
        raise ValueError('Probe lacks required evidence')
    for name, digest in files.items():
        relative = Path(name)
        target = (directory/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(directory):
            raise ValueError('Unsafe probe artifact path')
        if not target.is_file():
            raise ValueError('Missing probe artifact: ' + name)
        checksum = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError('Probe artifact SHA-256 mismatch: ' + name)
    partition = json.loads((directory/'validation_partitions.json').read_text(encoding='utf-8'))
    parts = partition.get('partitions', [])
    if (partition.get('run_kind') != 'probe' or partition.get('sample_limit') != selection['sample_limit']
            or len(parts) != 1 or parts[0].get('fold') != selection['fold']):
        raise ValueError('Remote probe selection differs from requested selection')
    ids = parts[0].get('validation_ids', [])
    if not ids or len(ids) > selection['sample_limit'] or len(ids) != len(set(ids)):
        raise ValueError('Invalid probe validation IDs')
    if set(ids) & set(parts[0].get('train_ids', [])):
        raise ValueError('Probe train/validation IDs overlap')
    with (directory/'oof_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predicted = [row[partition['id_column']] for row in csv.DictReader(stream)]
    if len(predicted) != len(ids) or set(predicted) != set(ids):
        raise ValueError('Probe predictions differ from selected validation IDs')
    return {'status': 'verified_probe', 'run_id': manifest['run_id'],
            'rows': len(ids), 'candidate_eligible': False, 'leaderboard_updated': False}
