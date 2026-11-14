# Interactive prediction payload validation

LifeMate oracle probe 414af9ac1e4349c78df9705185c18401 produced aligned
variable-length tool-call lists, but the shared prediction validator attempted
np.asarray before task-specific scoring and failed on the ragged shape.

For interactive_agent only, preserve structured payloads after existing ID,
truth-target and prediction-coverage checks. Reject missing episode values;
leave episode schema and semantics to the custom task evaluator. Numeric and
classification contracts retain their existing checks.

For an exclusively interactive task with a configured custom scorer, omit the
generic classification evaluator after custom scoring. The custom hook remains
authoritative and all pre-scoring integrity checks still run. This removes the
misleading generic failure warning without suppressing custom-hook failures.

Validation: focused ragged/coverage regression tests, existing evaluation
usability tests, then a new immutable LifeMate Harness probe. No old artifacts
are rewritten. This framework edit affects the canonical shared installation.
