# Reject cloud probes whose limits would be lost

The implicit probe path selected local, but explicit --backend modal/kaggle could still dispatch. The remote bridge does not forward --debug, --fold or --probe-samples, and generated templates assume full-run artifact paths. A supposedly small probe could therefore run with different selection than intended. Config sample_limit also needs a supported remote probe artifact path.

The main runner now rejects explicit real cloud dispatch for these sample-probe selections before provider dispatch. Dry-run remains allowed for static planning; full validation dispatch and local probes retain their routes. This enforces the existing local-smoke contract; it does not implement remote GPU probes or authorize a full run as a substitute for one.

Five Track1 checks include actual CLI routing with cloud dispatch replaced by failure and require the exact refusal message, plus config sample_limit for both backends. Together with preview tests7pass; existing backend policy18pass and Ruff passes. No provider calls or GPU were made. A remote-probe protocol must preserve selection, evidence type and artifact import before it can be supported.
