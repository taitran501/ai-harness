# Honor declarative data.folds in full validation

LifeMate exp016 Modal job ff4c76fcfe2e reached the remote trainer but failed
before model loading: data.folds=[0] was ignored, exporting five partitions to
a single-fold training adapter. Smoke --fold0 masked this dispatch mismatch.

After constructing and validating the complete split map, select data.folds
when CLI --fold is absent. Explicit CLI selection retains precedence. Reject
empty, duplicate and unknown configured folds. Keep the full legacy fold map
and enforce OOF coverage against only the selected manifest IDs.

Regression tests cover full script execution with configured fold0, CLI
precedence and malformed selection. Retry through a new immutable cloud job;
never alter the failed provider run or its artifacts.
