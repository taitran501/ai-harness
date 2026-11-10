"""Verified model upload and owned-volume cleanup for governed Modal dispatch."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import uuid

from harness.io_utils import atomic_write_text


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path, receipt):
    atomic_write_text(path, json.dumps(receipt, indent=2))


def stage_modal_models(transfer_plan, receipt_path, *, exp_id, environment_name, volume_api=None):
    """Called only by real dispatch, never preview. Creates a new volume, not a cache."""
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise ValueError('Model staging receipt already exists; inspect the previous attempt')
    if not environment_name or not exp_id or not transfer_plan.get('transfers'):
        raise ValueError('Explicit environment, experiment and nonempty transfer plan required')
    transfers = copy.deepcopy(transfer_plan['transfers'])
    seen = set()
    for transfer in transfers:
        if not transfer.get('files'):
            raise ValueError('Empty model transfer')
        for item in transfer['files']:
            remote = PurePosixPath(item['remote_path'])
            if not remote.is_absolute() or '..' in remote.parts or '\\' in str(remote) or str(remote) in seen:
                raise ValueError('Unsafe or duplicate remote model path')
            seen.add(str(remote))
            digest = item.get('sha256', '')
            path = Path(item['local_path'])
            if not re.fullmatch('[0-9a-f]{64}', digest) or not path.is_file():
                raise ValueError('Missing verified model input')
            if item.get('bytes') is not None and path.stat().st_size != item['bytes']:
                raise ValueError('Model file size mismatch before upload')
            if _hash(path) != digest:
                raise ValueError('Model SHA-256 mismatch before upload')
    owner_token = uuid.uuid4().hex
    volume_name = 'aih-model-job-' + owner_token
    receipt = {'schema_version': 1, 'exp_id': exp_id, 'environment_name': environment_name,
               'owner_token': owner_token, 'volume_name': volume_name, 'created': False,
               'dispatch_started': False,
               'status': 'creating', 'local_hashes_verified': True,
               'remote_contents_verified': False, 'transfers': transfers}
    _save(receipt_path, receipt)  # Intent survives an interrupted provider call.
    if volume_api is None:
        import modal
        volume_api = modal.Volume
    try:
        # Public API defaults to fail-if-existing; explicit false prevents adoption.
        volume_api.objects.create(volume_name, allow_existing=False, environment_name=environment_name)
        receipt.update(created=True, status='uploading')
        _save(receipt_path, receipt)
        volume = volume_api.from_name(volume_name, environment_name=environment_name)
        with volume.batch_upload(force=False) as upload:
            for transfer in transfers:
                for item in transfer['files']:
                    upload.put_file(item['local_path'], item['remote_path'])
        receipt['status'] = 'uploaded_pending_runtime_hash_verification'
        _save(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
        _save(receipt_path, receipt)
        raise


def _owned_receipt(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    token = receipt.get('owner_token', '')
    if not re.fullmatch('[0-9a-f]{32}', token) or receipt.get('volume_name') != 'aih-model-job-' + token:
        raise ValueError('Invalid model-volume ownership receipt')
    if receipt.get('created') is not True or not receipt.get('environment_name'):
        raise ValueError('No confirmed owned model volume to clean')
    return receipt


def mark_modal_dispatch_started(receipt_path, *, identity=None, app_name=None):
    receipt = _owned_receipt(receipt_path)
    receipt['dispatch_started'] = True
    if identity is not None:
        receipt['dispatch_identity'] = identity
    if app_name is not None:
        receipt['app_name'] = app_name
    _save(receipt_path, receipt)


def upload_owned_task_data(project, receipt_path, expected_fingerprint, *, volume_api=None):
    """Upload declared consented data beside models in the same owned volume."""
    from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
    from harness.data_bundle import stage_declared_data
    receipt = _owned_receipt(receipt_path)
    if receipt.get('dispatch_started') is not False:
        raise ValueError('Cannot upload task data after dispatch')
    current = dataset_fingerprint(project)
    if current['fingerprint'] != expected_fingerprint or not UploadConsentStore(project).is_granted('modal', expected_fingerprint):
        raise ValueError('Task-data fingerprint changed or upload consent missing')
    if any(item['remote_path'] == '/task-data' or item['remote_path'].startswith('/task-data/')
           for transfer in receipt['transfers'] for item in transfer['files']):
        raise ValueError('Model path overlaps task-data')
    with tempfile.TemporaryDirectory(prefix='aih-owned-task-data-') as temporary:
        staged = stage_declared_data(project, temporary)
        files = []
        for relative in staged['manifest']['files']:
            original = (project.root/relative).resolve()
            copied = Path(temporary)/relative
            if not original.is_relative_to(project.root.resolve()):
                raise ValueError('Task data escapes workspace')
            files.append({'local_path': str(original), 'sha256': _hash(copied)})
        receipt.update(task_data_files=files, task_data_fingerprint=expected_fingerprint,
                       task_data_uploaded=False)
        _save(receipt_path, receipt)
        if volume_api is None:
            import modal
            volume_api = modal.Volume
        volume = volume_api.from_name(receipt['volume_name'], environment_name=receipt['environment_name'])
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(temporary, '/task-data')
        receipt['task_data_uploaded'] = True
        _save(receipt_path, receipt)
    return receipt


def cleanup_undispatched_models(receipt_path, *, volume_api=None):
    """Recover upload/preparation failures only when no job launch was attempted."""
    receipt = _owned_receipt(receipt_path)
    if receipt.get('dispatch_started') is not False:
        raise ValueError('Cannot recover staging after a job launch was attempted')
    if receipt.get('status') == 'cleaned_before_dispatch':
        return receipt
    files = [item for transfer in receipt.get('transfers', []) for item in transfer.get('files', [])]
    if not files:
        raise ValueError('Missing retained local model manifest')
    for item in files + receipt.get('task_data_files', []):
        if _hash(item['local_path']) != item['sha256']:
            raise ValueError('Retained local model SHA-256 mismatch; preserve remote')
    if volume_api is None:
        import modal
        volume_api = modal.Volume
    volume_api.objects.delete(receipt['volume_name'], allow_missing=False,
                              environment_name=receipt['environment_name'])
    receipt.update(status='cleaned_before_dispatch', retained_local_models_verified=True)
    _save(receipt_path, receipt)
    return receipt


def cleanup_modal_models(receipt_path, local_exp_dir, *, volume_api=None):
    """Delete owned model storage only after full local artifacts validate."""
    from harness.artifacts import validate_run
    receipt_path = Path(receipt_path)
    receipt = _owned_receipt(receipt_path)
    if receipt.get('probe_selection') is not None:
        from harness.cloud_probe import verify_probe_artifacts
        identity = receipt.get('dispatch_identity')
        if not identity or not identity.get('job_id'):
            raise ValueError('Probe cleanup requires recorded dispatch identity')
        verify_probe_artifacts(local_exp_dir, expected={**identity, 'run_id': identity['job_id']},
                               selection=receipt['probe_selection'])
    else:
        validate_run(local_exp_dir, expected={'exp_id': receipt['exp_id']})
    if receipt.get('status') == 'cleaned':
        return receipt
    if volume_api is None:
        import modal
        volume_api = modal.Volume
    volume_api.objects.delete(receipt['volume_name'], allow_missing=False,
                              environment_name=receipt['environment_name'])
    receipt.update(status='cleaned', local_artifacts_verified=True,
                   local_artifact_manifest_sha256=_hash(Path(local_exp_dir)/'artifact_manifest.json'))
    _save(receipt_path, receipt)
    return receipt


def recover_failed_modal_models(receipt_path, destination, *, run_command=None, volume_api=None):
    """Require explicit terminal app state and verified failure evidence before delete."""
    from harness.failure_evidence import verify_failed_snapshot
    receipt = _owned_receipt(receipt_path)
    if receipt.get('dispatch_started') is not True or not receipt.get('app_name') or not receipt.get('dispatch_identity'):
        raise ValueError('Missing launched app identity; preserve remote')
    run_command = run_command or subprocess.run
    environment = receipt['environment_name']
    query = ['modal', 'app', 'list', '--json', '--env', environment]
    def terminal_app():
        result = run_command(query, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ValueError('Cannot confirm terminal Modal app')
        apps = [{str(k).lower().replace(' ', '_'): v for k, v in item.items()}
                for item in json.loads(result.stdout)]
        matches = [item for item in apps if item.get('description') == receipt['app_name']]
        if len(matches) != 1:
            raise ValueError('App missing or ambiguous; preserve remote')
        app = matches[0]
        if not app.get('app_id') or str(app.get('state', '')).lower() not in {'stopped', 'detached_stopped'} or str(app.get('tasks')) != '0':
            raise ValueError('App not confirmed stopped with zero tasks; preserve remote')
        return app
    app = terminal_app()
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Recovery destination exists; preserve earlier evidence')
    destination.mkdir(parents=True)
    _save(destination/'terminal_app.json', app)
    logs = run_command(['modal', 'app', 'logs', app['app_id'], '--tail', '1000', '--env', environment],
                       capture_output=True, text=True, timeout=30)
    if logs.returncode != 0:
        raise ValueError('Could not retain provider logs; preserve remote')
    (destination/'provider_tail.log').write_text(logs.stdout + logs.stderr, encoding='utf-8')
    downloaded = destination/'snapshots'
    # Modal CLI treats a nonexistent local destination as a file path, even
    # when the remote source is a directory. Create it before recursive get.
    downloaded.mkdir()
    result = run_command(['modal', 'volume', 'get', receipt['volume_name'],
                          '/failures/' + receipt['exp_id'], str(downloaded), '--env', environment],
                         capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise ValueError('Failure evidence download incomplete; preserve remote')
    manifests = list(downloaded.rglob('failure_manifest.json'))
    if not manifests:
        raise ValueError('No complete failure snapshot; preserve remote')
    verified = {}
    for manifest in manifests:
        verify_failed_snapshot(manifest.parent, receipt['dispatch_identity'])
        verified[str(manifest.relative_to(destination))] = _hash(manifest)
    retained = [item for transfer in receipt['transfers'] for item in transfer['files']]
    for item in retained + receipt.get('task_data_files', []):
        if _hash(item['local_path']) != item['sha256']:
            raise ValueError('Original local inputs changed; preserve remote')
    current = terminal_app()
    if current['app_id'] != app['app_id']:
        raise ValueError('App identity changed during recovery; preserve remote')
    recovery = {'status': 'verified', 'app_id': app['app_id'], 'failure_manifests': verified,
                'provider_tail_sha256': _hash(destination/'provider_tail.log')}
    _save(destination/'recovery_receipt.json', recovery)
    if volume_api is None:
        import modal
        volume_api = modal.Volume
    volume_api.objects.delete(receipt['volume_name'], allow_missing=False, environment_name=environment)
    receipt.update(status='cleaned_after_failure', failure_recovery_path=str(destination.resolve()),
                   failure_recovery_receipt_sha256=_hash(destination/'recovery_receipt.json'))
    _save(receipt_path, receipt)
    return receipt
