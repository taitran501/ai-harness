# Plan: Implement các gap framework ảnh hưởng đến performance của agents

**Ngày:** 2026-10-19 · **Nguồn:** audit forensic conversation Antigravity
(`arc-prize-2026-arc-agi-3`, session `ce5361e6` + 4 conv phụ) ·
**Trạng thái:** PR #109 đã merge (abb485b) — plan này cover **các gap còn lại**.

## 1. Context — agent đã chết ở đâu

Kết quả audit (bằng chứng: step index trong conversation DB):

| # | Failure | Hậu quả cho agent | Status |
|---|---------|-------------------|--------|
| A | `ensure_dependencies` pin exact local (`numpy==1.26.4, pandas==3.0.5, ...`) vs Kaggle image offline | Kernel chết **trước khi train**, lặp exp002/exp005/exp006 (idx 490, 1649, 2139) | ✅ Pin `>=` trong PR #109; fail startup do pin framework-derived được fix triệt để bởi G3 (PR TBD: mode `requirement_check` + detected versions) |
| B | `AttributeError: module 'src' has no attribute 'evaluator:evaluate_rhae'` ở `validate_run` remote (idx 653) | Run train xong vẫn chết ở finalization | ✅ Fix hook-parsing trong PR #109 |
| C | `Kaggle output import failed: No module named 'src'` (idx 726, 1715) | **2 kernel COMPLETE bị mất output** — agent waste cả chu kỳ compute | ✅ **Đã fix** (G1 — PR #111: hermetic `project_root` import + diagnostic `RuntimeError`) |
| D | `400 Bad Request` khi `submission dispatch` — chỉ hiện error line, **không có API error body** (idx 2500) | Agent mất thời gian đoán là daily-limit; không thể phân biệt limit/slug/file sai (idx 2500-2557) | ✅ **Đã fix** (G2 — PR #110: surfacing API error body + URL) |
| E | CLI `kaggle` unauthenticated (không `kaggle.json`) — "Authentication required" giữa chừng flow | Mọi lệnh CLI trần fail; agent phải tự đi tìm kaggle.json | ⚠️ **Còn** (G4) |
| F | "Job ID contains unsupported characters" không nói pattern hợp lệ (×3 attempts) | Agent thử sai `test_job`, `job_test`, ... | ✅ **Đã fix** (G5 — PR #110: error kèm `JOB_ID_PATTERN` + ví dụ) |
| G | Agent **re-dispatch y nguyên config fail** nhiều lần (exp002 ×3, exp005 tới kernel v4) | Waste chu kỳ push+poll (mỗi lần ~phút + quota kernel) | ⚠️ Optional (G6) |

**Correction so với audit lần trước:** bundle code **đã** chứa `src/` + `configs/`
từ commit đầu (`398df9e` — `for rel in ("src", "configs")` trong
`kaggle_template.py` L527-531) nên C **không** phải do thiếu `src/` trong bundle.
C là do **import context** (sys.path/env) của `validate_run` — còn fragile với
mọi process chạy import artifact **tách riêng** (inspect CLI, cwd khác workspace).

## 2. Scope & ưu tiên

| ID | Gap | Ảnh hưởng agent | Effort | Priority |
|----|-----|-----------------|--------|----------|
| G1 | Scorer import hermetic + diagnostic fail trong `validate_run` / `migrate_downloaded_artifacts` | Cao (mất output run đã chạy) | S-M | **P0** |
| G2 | Surface API error body (400/5xx) khi dispatch submission | Cao (agent không tự chẩn đoán được) | S | **P0** |
| G3 | `ensure_dependencies`: detected versions + `requirement_check: strict\|warn\|off` | Trung bình (chống fail kernel startup tương lai) | M | P1 |
| G4 | `ensure_cli_ready` thêm auth probe (advisory) | Trung bình (fail CLI giữa chừng) | S | P1 |
| G5 | Job-ID error message kèm pattern + ví dụ | Thấp | XS | P2 |
| G6 | Re-dispatch advisory (error-signature dedup warning) | Trung bình (chống waste loop) — **optional** | M | P3 |

**Không trong scope** (không phải gap framework):
- Wheels ARC-AGI-3 yêu cầu Python ≥3.12 vs local 3.11.9 → vấn đề **environment
  workspace** (cần Python 3.12 hoặc wheels cũ), xử lý riêng trong workspace.
- PowerShell JSON escaping của agent → hành vi agent; AGENTS.md đã có cảnh báo;
  `--param-file` là workaround đã khám phá.

## 3. Thiết kế chi tiết

### G1 — Scorer import hermetic + diagnostic (P0)

**Vấn đề:** `validate_run` (artifacts.py L131-163) gọi
`importlib.import_module(module)` cho `metrics_evaluator`. `sys.path` của module
scorer (`src.evaluator`) chỉ chắc chắn khi: cwd = workspace root, **hoặc**
`HARNESS_PROJECT_ROOT` env đã set (runner set trong subprocess remote —
runner.py L1244-1248 — nhưng **không** set trong parent process
`run_kaggle.py`, và **không** có trong separate CLI process như
`ai-harness kaggle`). Hiện tại chỉ có `ensure_hook_project_roots_on_sys_path()`
(hook_ref.py L137-146) — phụ thuộc ambient env, không deterministic.

**Đổi:**
1. `harness/hook_ref.py` — extend:
   ```python
   def ensure_hook_project_roots_on_sys_path(project_root: str | os.PathLike | None = None) -> None:
       # giữ logic hiện có, thêm:
       if project_root is not None:
           insert_project_root(Path(project_root))
   ```
2. `harness/templates/kaggle_template.py` —
   - `migrate_downloaded_artifacts(download_dir, target_artifacts_dir, *, project_root=None)`
     (L1121): nhận `project_root`, gọi
     `ensure_hook_project_roots_on_sys_path(project_root)` **trước**
     `validate_run(source_run)` và `validate_run(destination)`.
   - `pull_kernel_output` (L938): truyền `project_root` (đã là tham số!) vào
     `migrate_downloaded_artifacts(...)` tại call site L1084.
3. `harness/artifacts.py` — trong `validate_run`, wrap import:
   ```python
   try:
       mod = importlib.import_module(module)
   except ModuleNotFoundError as exc:
       raise RuntimeError(
           f"metrics_evaluator hook {hook!r} is not importable "
           f"(module={module!r}, name={name!r}). "
           f"project_root={os.environ.get('HARNESS_PROJECT_ROOT', '<unset>')}; "
           f"sys.path[:5]={sys.path[:5]!r}. "
           "Verify the scorer module exists in the workspace (and in the Kaggle "
           "code bundle for remote runs)."
       ) from exc
   ```
   *(Giữ nguyên digest check — **không** relax integrity contract.)*
4. `harness/templates/kaggle_template.py` — code generation của `run_kaggle.py`:
   sau `os.chdir(workspace)` thêm
   `os.environ["HARNESS_PROJECT_ROOT"] = str(workspace)` (defense-in-depth cho
   `export_run` chạy trong parent process remote).

**Tests** (mới trong `tests/`, dùng `make_workspace()` từ `tests/fixtures.py`):
- `test_scoring_import_hermetic`: chdir sang dir khác workspace, build artifact
  dir hợp lệ (manifest + run_config + src/evaluator.py), gọi
  `migrate_downloaded_artifacts(..., project_root=ws)` → success; không truyền
  `project_root` + module thiếu → `RuntimeError` **chứa hook ref + sys.path
  excerpt** trong message.
- Remote codegen test: assert generated `run_kaggle.py` chứa
  `HARNESS_PROJECT_ROOT`.

**docs/changes:** `scorer-import-hermetic-context.md`.

### G2 — Surface API error body khi dispatch submission (P0)

**Vấn đề:** `KaggleClient.submit_submission` (kaggle_compat.py L250-272) gọi
legacy `api.competition_submit_code(...)`; HTTP 400 fly ra thành
`requests.HTTPError` — harness chỉ stringify (submission_dispatch.py L123
`except Exception`) → agent chỉ thấy `400 Client Error: Bad Request for url: ...`,
**mất response body** (nơi Kaggle ghi "daily limit reached" v.v.).

**Đổi:** `harness/kaggle_compat.py` — `submit_submission`:
```python
try:
    resp = api.competition_submit_code(...) if kernel else api.competition_submit(...)
except Exception as exc:  # noqa: BLE001 - surface Kaggle API errors with body
    response = getattr(exc, "response", None)
    body = ""
    try:
        body = (response.text or "")[:2000] if response is not None else ""
    except Exception:
        pass
    url = getattr(response, "url", "") if response is not None else ""
    detail = f" response body: {body}" if body else " (no response body)"
    raise RuntimeError(
        f"Kaggle submission dispatch failed ({exc})"
        + (f" url={url}" if url else "")
        + detail
    ) from exc
```
- **Verification step** (trước khi code): kiểm tra exception type thật do legacy
  `KaggleApi.competition_submit_code` raise (requests.HTTPError có `.response`
  hay kaggle exception riêng) — điều chỉnh extractor cho phù hợp; thiết kế
  fallback `getattr` là an toàn với cả hai.
- Không đổi contract: `submission_dispatch.main` đã stringify vào JSON payload
  `{"ok": false, "error": ...}` — agent đọc được body ngay.

**Tests:**
- monkeypatch legacy api raise exception giả có `.response` (object với
  `.text`/`.url`) → assert `RuntimeError` chứa body.
- exception không có `.response` → message fallback `(no response body)`.
- Path success không đổi (`ref`/`message` normalize).

**docs/changes:** `submission-dispatch-error-body.md`.

### G3 — `ensure_dependencies`: detected versions + mode `requirement_check` (P1 — ✅ đã fix, PR TBD)

**Vấn đề:** kernel fail startup chỉ nói requirement thiếu, **không nói image
đang có version gì** → agent không biết gap là 1 minor hay major (aud: exp006
fail `pandas>=3.0.0` trong khi image có pandas 2.x — nếu có detected version,
agent biết ngay "image 2.2.x không chạy code pandas-3").

**Đổi:**
1. `harness/kaggle_runtime.py` — `ensure_dependencies(runtime, *, enable_internet=False, requirement_check="strict")`:
   - Với mỗi requirement fail, đọc detected version qua
     `importlib.metadata.version(distribution)` (đã import sẵn — không import
     native module) → `"not installed"` nếu thiếu.
   - Error message mới:
     `"Kaggle internet is disabled and the image does not satisfy: pandas>=3.0.0"
     " (detected: pandas=2.2.2). ..."`
   - `requirement_check == "warn"`: print WARNING line (vào kernel log) rồi
     **continue**; `"off"`: skip check hẳn (vẫn log versions detected).
2. Task contract — `configs/task.yaml` mới thêm field (optional, default strict):
   ```yaml
   kaggle:
     runtime:
       requirement_check: warn   # strict | warn | off
   ```
   - Wire: `cloud_runner.py` L311-312 đọc cùng chỗ với `enable_internet` /
     `kaggle_bundle_mode` → pass vào runtime manifest → template truyền cho
     `ensure_dependencies`.
3. Docs: `docs/TASK_SPEC_TEMPLATE.md` + `docs/kaggle_execution.md` (bảng version
   Kaggle gcp image hiện tại + khi nào dùng warn).

**Tests:**
- mock metadata (fixture package thiếu/đúng version) → assert message chứa
  `detected:`; `warn` không raise; `off` không raise; `strict` raise như cũ.
- Contract parse: `requirement_check` absent → `strict`; value sai → ValueError.

**docs/changes:** `kaggle-requirement-check-modes.md`.

### G4 — CLI auth probe advisory trong `ensure_cli_ready` (P1)

**Vấn đề:** preflight (PR #109) chỉ probe `kaggle --version` — CLI present nhưng
**unauthenticated** vẫn trượt tới giữa push mới fail ("Authentication required",
aud idx có kaggle.json missing; agent tự `Get-ChildItem -Recurse` tìm credentials).

**Đổi:** `harness/templates/kaggle_template.py` — `ensure_cli_ready(*, check_python=True, check_auth=True)`:
```python
# sau version probe OK, cache cùng _cli_version_cache (tupled):
if check_auth:
    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "-v", "1"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        proc = None  # timeout không block dispatch
    if proc is not None and proc.returncode != 0:
        blob = ((proc.stdout or "") + (proc.stderr or "")).lower()
        if any(k in blob for k in ("auth", "login", "credential", "kaggle.json")):
            print(
                "[kaggle preflight] WARNING: Kaggle CLI reports no usable "
                "authentication. Kernel push/submission via CLI will fail with "
                "'Authentication required'. Fix: `kaggle auth login` or place "
                "~/.kaggle/kaggle.json. (SDK-based dispatch may still work.)",
                file=sys.stderr, flush=True,
            )
```
- **Advisory, không fatal**: kagglesdk path có thể đang auth mà CLI không (aud:
  chính case này — SDK OAuth ok, CLI không) → chỉ WARNING, dispatch tiếp.
- Timeout/exception của probe: silently skip (preflight không được trở thành
  failure mode mới).

**Tests:** monkeypatch `subprocess.run` → (a) auth-fail output → capsys có
WARNING, không raise; (b) returncode 0 → không warning; (c) timeout → skip an
n toàn.

**docs/changes:** `kaggle-cli-auth-preflight.md`.

### G5 — Job-ID error message (P2)

**Đổi:** `harness/job_state.py` L18:
```python
raise ValueError(
    f"Job ID contains unsupported characters: {job_id!r}. "
    f"Expected pattern: {JOB_ID_PATTERN.pattern} "
    "(e.g. 'job_exp001_full', 'job_exp002-smoke-0')."
)
```
*(Cùng pattern cho `submission_manager.py` L225 nếu message chưa đủ — kiểm tra
khi implement.)*

**Tests:** assert message chứa pattern + example.

**docs/changes:** gộp vào G2 (cùng chủ đề error-surface) hoặc file riêng
`job-id-error-message.md` (nên: riêng, 1 file/1 change theo convention).

### G6 — Re-dispatch advisory (P3, **optional**)

**Vấn đề:** agent re-dispatch y nguyên config vừa fail (×3 exp002, ×4 exp005) —
mỗi lần waste kernel quota + thời gian poll.

**Thiết kế (nếu làm):**
- `harness/job_state.py`: lưu `last_error_signature` (hash của
  `(exp_id, resolved_config_sha256, backend, first_error_line)`) vào
  `jobs/<job_id>.worker.json` khi job FAILED (nơi đã có status update).
- `harness/cloud_runner.run_via_kaggle`: trước khi dispatch, quét jobs cùng
  exp_id; nếu có job FAILED gần đây với **cùng signature** →
  `[dispatch preflight] WARNING: previous job <id> failed with the same error
  ("<first line>"); the resolved config was not changed. Consider changing the
  configuration before re-dispatching.`
- Advisory only (không block) — đúng precedent advisory warnings của PR #109.

**Tests:** fixture job state cũ + dispatch mock → assert warning.

## 4. Trật tự implement & PR batching

| PR | Chứa | Lý do batch |
|----|------|-------------|
| **PR-A** | G2 + G5 | Cùng chủ đề *error surface*, code chạm 2 file nhỏ, test isolate — merge sớm, value cao nhất cho agent |
| **PR-B** | G1 | Chạm integrity path (`validate_run`, `migrate_downloaded_artifacts`) — cần review kỹ + test hermetic riêng |
| **PR-C** | G3 | Chạm contract schema + runtime manifest — test rộng hơn |
| **PR-D** | G4 | Độc lập, nhỏ |
| PR-E | G6 | Chỉ nếu P0-P1 xong và còn budget |

Mỗi PR: branch từ `main` mới nhất → implement → `docs/changes/<slug>.md`
(đường link `_TEMPLATE.md`) + hàng index trong `docs/changes/README.md` →
test gates → PR (follow flow PR #109).

## 5. Verification gates

**Mỗi PR (trước khi push):**
```bash
python tools/run_tests.py --smoke   # ~2s, critical invariants
python tools/run_tests.py --fast    # iteration
python tools/run_tests.py           # full suite trước push (100%)
ruff check harness/ tools/ tests/   # static gate
```
**Constraints test:**
- Integrity contract **không relax**: sealed runs cũ (manifest có
  `custom_scorer_sha256`) vẫn validate pass/fail đúng như trước (mới chỉ thêm
  diagnostic message khi import fail).
- Contract backward-compat: `requirement_check` absent ⇒ `strict`; mọi task.yaml
  cũ parse không đổi hành vi.
- Codegen test đảm bảo `run_kaggle.py` generated không đổi ngoài dòng
  `HARNESS_PROJECT_ROOT` (sha256 bundle identity tests).

**E2E (sau khi merge PR-A..D):**
1. Re-dispatch `exp006` (hoặc exp mới) ở `arc-prize-2026-arc-agi-3` workspace:
   push → poll → **kernel COMPLETE** (verify fix A của PR #109 trên kernel thật:
   `ensure_dependencies` với pin `>=` hiện tại vs image).
2. Output import về local artifact dir thành công (`import src.evaluator` qua
   `migrate_downloaded_artifacts`).
3. Submission dispatch (chờ UTC daily reset): nếu bị 400 → error phải **chứa
   response body** (acceptance của G2).
4. `ai-harness kaggle` inspect trên kernel đó (separate process) không gặp
   "No module named 'src'".

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| G1 đổi hành vi `validate_run` → phá seal-validation cũ | Chỉ **add** sys.path + message; digest check nguyên vẹn; test with sealed manifest fixture |
| G3 `warn/off` tạo precedent bypass correctness | Default `strict`; `off` log detected versions; docs ghi rõ warn chỉ khi đã verify image tương thích |
| G4 auth probe 60s làm chậm dispatch | Probe chỉ 1 lần/process (cache), advisory, timeout skip an toàn |
| G2 extractor không khớp exception type thật của legacy Kaggle | Step verify trước khi code (mock cả 2 shape: requests.HTTPError + kaggle exception) |
| Concurrency khi re-dispatch E2E | Concurrency guard của `run_tests.py` (PR #108) + chạy E2E riêng khỏi CI |

## 7. Definition of done (toàn bộ plan)

- [ ] PR-A..PR-D merged, full suite 100%, ruff clean
- [ ] 4-5 files `docs/changes/` + index rows
- [ ] E2E arc-prize: 1 run harness-dispatched **COMPLETE → import → metrics.json** thành công
- [ ] Submission dispatch error (nếu có 400) chứa API body
- [ ] (nếu làm) G6 warning xuất hiện khi re-dispatch config vừa fail
