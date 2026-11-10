# Create the destination directory before recursive Modal recovery

The first live Track1 failure produced a valid remote snapshot but recovery failed with Permission denied: the Modal CLI treats a nonexistent local destination as a single output file. Recovery now creates its snapshots directory before modal volume get. The mocked download regression now asserts this precondition.

Live recovery then downloaded the exact failed job snapshot, verified its file hashes/identity and retained inputs, rechecked the stopped app and deleted only the owned volume. The local receipt records cleaned_after_failure; an official volume-list check confirmed the named volume absent. This does not claim a successful model inference.
