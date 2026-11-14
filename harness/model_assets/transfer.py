"""Validate exact local-to-remote model transfer plans without provider access."""
from pathlib import Path, PurePosixPath


def plan_model_transfers(local_plan, remote_plan):
    """No file copying, hashing or provider writes; runtime still verifies SHA-256."""
    if local_plan.backend != 'local' or remote_plan.backend != 'modal':
        raise ValueError('Model transfer requires local and Modal plans')
    transfers = []
    local_sources = local_plan.required_sources
    for remote in remote_plan.required_sources:
        matches = [s for s in local_sources if s['slot'] == remote['slot']]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous local model source')
        local = matches[0]
        for key in ('model_id', 'definition_digest', 'bundle_digest', 'revision', 'files'):
            if local.get(key) != remote.get(key):
                raise ValueError('Local and remote model facts differ: ' + key)
        if local.get('missing_prerequisites') or remote.get('missing_prerequisites'):
            raise ValueError('Model source has missing prerequisites')
        if not remote.get('volume_name') or not remote.get('files'):
            raise ValueError('Model transfer needs a volume and nonempty file manifest')
        remote_root = PurePosixPath(remote['mount_path'])
        if '..' in remote_root.parts:
            raise ValueError('Unsafe model mount path')
        try:
            relative_root = remote_root.relative_to('/vol/models')
        except ValueError as exc:
            raise ValueError('Model mount must be within /vol/models') from exc
        local_root = Path(local['mount_path']).resolve()
        files = []
        for item in local['files']:
            relative = PurePosixPath(item['relative_path'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
                raise ValueError('Unsafe model transfer file path')
            source = (local_root/Path(*relative.parts)).resolve()
            if not source.is_relative_to(local_root):
                raise ValueError('Local model transfer escapes its root')
            destination = PurePosixPath('/')/relative_root/relative
            files.append({**item, 'local_path': str(source), 'remote_path': str(destination)})
        transfers.append({'slot': remote['slot'], 'volume_name': remote['volume_name'],
                          'definition_digest': remote['definition_digest'],
                          'bundle_digest': remote['bundle_digest'], 'files': files,
                          'total_bytes': sum(f['bytes'] for f in files) if all(
                              isinstance(f.get('bytes'), int) for f in files) else None})
    return {'status': 'planned_only', 'transfers': transfers,
            'local_hashes_verified': False, 'remote_contents_verified': False,
            'ownership_verified': False, 'provider_writes': False}
