"""Short reentrant cross-process JSON transactions; never lock training work."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import threading
from harness.io_utils import atomic_write_json

_locks={}
_guard=threading.Lock()
_local=threading.local()

@contextmanager
def workspace_lock(directory):
    directory=Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
    key=str(directory)
    with _guard: lock=_locks.setdefault(key,threading.RLock())
    with lock:
        held=getattr(_local,"held",{})
        if key in held:
            yield; return
        path=directory/".state.lock"
        with path.open("a+b") as stream:
            stream.seek(0); stream.write(b"0"); stream.flush(); stream.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            held[key]=True; _local.held=held
            try: yield
            finally:
                held.pop(key)
                stream.seek(0)
                if os.name=="nt": msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

@contextmanager
def transaction(path,default=None):
    path=Path(path)
    with workspace_lock(path.parent):
        value=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else copy.deepcopy(default or {})
        yield value
        value["revision"]=int(value.get("revision",0))+1
        atomic_write_json(path,value,indent=2,ensure_ascii=False)

def merge_changed(current,before,after):
    """Apply a snapshot's changes, preserving concurrent unrelated updates and append-only events."""
    for key,value in after.items():
        if key in before and value==before[key]: continue
        if isinstance(value,dict) and isinstance(before.get(key),dict) and isinstance(current.get(key),dict):
            merge_changed(current[key],before[key],value)
        elif isinstance(value,list) and isinstance(before.get(key),list):
            added=[x for x in value if x not in before[key]]
            removed=[x for x in before[key] if x not in value]
            current[key]=[x for x in current.get(key,[]) if x not in removed]
            current[key].extend(x for x in added if x not in current[key])
        else: current[key]=copy.deepcopy(value)
    return current
