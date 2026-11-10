# Preserve lightweight script diagnostics in Modal results

LifeMate paired inference needs raw action/stop traces and both arm scores.
The generic Modal return bundle previously omitted custom script files and
the actual validation manifest, losing useful evidence from completed jobs.

Add explicit lightweight names for script trace/progress/metrics, adapter/data
receipts and validation partitions/splits. Continue excluding weights. Existing
hash-bound bundle import verifies these files and old completed runs stay intact.
The running script flushes JSONL per action/episode for live container inspection;
it does not depend on the runner's buffered child stdout.
