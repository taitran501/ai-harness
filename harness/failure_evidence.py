"""Preserve partial failed-run files without treating them as valid model results."""
import hashlib
import json
from pathlib import Path
import shutil


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_failed_run(source, destination, identity, exit_code, runner_log):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Failure snapshot destination already exists')
    destination.mkdir(parents=True)
    files = {}
    if source.exists():
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Refusing linked failure evidence')
            if not path.is_file():
                continue
            relative = Path('partial_run')/path.relative_to(source)
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files[relative.as_posix()] = _sha(target)
    target_log = destination/'remote_runner.log'
    shutil.copy2(runner_log, target_log)
    files['remote_runner.log'] = _sha(target_log)
    manifest = {'schema_version': 1, 'kind': 'failed_run_evidence', 'identity': identity,
                'exit_code': exit_code, 'quality_score': None, 'files': files}
    (destination/'failure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def verify_failed_snapshot(directory, expected_identity):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'failure_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'failed_run_evidence' or not manifest.get('files'):
        raise ValueError('Missing failure evidence manifest')
    if any(manifest.get('identity', {}).get(key) != value for key, value in expected_identity.items()):
        raise ValueError('Failed snapshot identity mismatch')
    for name, digest in manifest['files'].items():
        relative = Path(name)
        path = (directory/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(directory):
            raise ValueError('Unsafe failed snapshot path')
        if not path.is_file() or _sha(path) != digest:
            raise ValueError('Failed snapshot SHA-256 mismatch')
    return manifest
