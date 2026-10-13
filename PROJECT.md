# Project: AI-Harness Web Dashboard UI/UX Optimization

## Architecture
- **Frontend Layer**: React 18 SPA with TypeScript, Tailwind CSS, Lucide icons, Vite (`dashboard/`).
- **Static Assets Delivery**: `dashboard/dist/` packaged into `tools/web/` via `scripts/copy-package-ui.mjs` and served by `tools/dashboard_server.py`.
- **Backend API**: Python HTTP server (`tools/dashboard_server.py`) exposing REST endpoints:
  - `/api/competition-state`: State inspection and action execution (`/api/competition-state/<action>`).
  - Telemetry: `/api/tree`, `/api/experiments`, `/api/recommendation`, `/api/status`, `/api/task-info`, `/api/submissions`.
- **Localization Infrastructure**: `dashboard/src/constants/i18n.ts` (`LanguageProvider`, `useLanguage`, `t(key, fallback, params)`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Optimistic Context Settings | Refactor "Advanced Agent Context Injection Settings" to optimistic local state with 0ms UI response | M1 | Survey R1 |
| 2 | Async Persistence & Rollback | Debounced (300ms) background POST to `/api/competition-state/agent-context` with rollback on error, eliminating `setBusy(true)` and `onRefreshAll()` | M1 | Survey R1 |
| 3 | Silent Background Refresh | Guard `loadData()` in `App.tsx` so only cold initial load triggers full unmount `loading`; background updates refresh silently | M1 | Survey R1 |
| 4 | Setup Stage KPI Contextual State | In Stage 1 Setup before Phase 3 baseline, display "Not Started" / "Awaiting Phase 3 Baseline" (VI: "Chưa bắt đầu" / "Chờ Baseline Giai đoạn 3") instead of 0.0000 | M2 | Survey R2 |
| 5 | Clean Misleading CV Labels | Remove premature "(CV)" labels from `TaskConfigModal.tsx` (fix line 307 fallback), `CompetitionPipelineHub.tsx`, `LeaderboardTable.tsx`, `NodeInspector.tsx` | M2 | Survey R2 |
| 6 | i18n Foundation Expansion | Expand `i18n.ts` with ~120 translation keys and parameter interpolation `t(key, fallback, params)` | M3 | Survey R3 |
| 7 | Full UI Localization (EN/VI) | Wire `t(...)` across `CompetitionJourneyWorkspace.tsx`, `WorkspaceHub.tsx`, `Header.tsx`, `NodeInspector.tsx`, `TaskConfigModal.tsx`, `App.tsx` | M3 | Survey R3 |
| 8 | Resilient App Telemetry Fetch | Use `Promise.allSettled` in `App.tsx` and render dismissable top banner instead of blocking full-page red screen | M4 | Survey R4 |
| 9 | WorkspaceHub Fetch Resilience | Replace raw non-dismissable error banner in `WorkspaceHub.tsx` with dismissable status card, retry button, and polling error suppression | M4 | Survey R4 |
| 10| Full E2E Build & Suite Pass | Verify `npm run build:package` code 0, Python test suite 100% pass, and verify zero regressions | M5 | Survey Verification |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Zero-Flicker Context Settings | Optimistic local state, debounced async persistence, silent background refresh in `CompetitionJourneyWorkspace.tsx` and `App.tsx` | none | DONE |
| M2 | Setup Stage KPI & CV Clean Up | Replace 0.0000/Verified with contextual baseline states; sanitize `(CV)` labels across modal and setup views | M1 | DONE |
| M3 | Comprehensive EN/VI Localization | Expand `i18n.ts` with standard ML terminology and wire `t(...)` across all workspaces with zero mixed text | M2 | DONE |
| M4 | API Fetch Resilience & Error Handling | Replace blocking red screens in `App.tsx` and `WorkspaceHub.tsx` with dismissable cards and retry handlers | M3 | DONE |
| M5 | E2E Verification & Integrity Pass | Frontend package build, full Python test suites, and final end-to-end audit | M4 | DONE |

## Interface Contracts
### Checkbox State ↔ Async Background Persistence
- Local state: `localAgentCtxCfg: Record<string, boolean>` initialized from `compState?.agent_context_config`.
- On change: Immediate local state update `setLocalAgentCtxCfg(next)`. Non-blocking debounced (300ms) POST to `api.postCompetitionAction('agent-context', { config: next })`.
- Status indicator: Subtle inline badge ("Saving..." / "Saved" / "Error") in settings card header; no global `busy` lock.
- On failure: Restore `localAgentCtxCfg` to previous synced ref; display subtle inline warning.

### Baseline Status Contract
- `isBaselineEstablished = Boolean(gates?.baseline_locked && typeof currentBest.baseline?.score === 'number' && !isNaN(currentBest.baseline.score) && experiments.length > 0)`.
- When `selectedPhase === 1 && !isBaselineEstablished`:
  - KPI Card 2 title: `t('best_score_label', 'Best Score')`
  - Score display: `t('status_not_started', 'Not Started')`
  - Delta badge: `t('status_awaiting_phase3', 'Awaiting Phase 3')` (neutral styling: `bg-zinc-800 text-zinc-400 border border-zinc-700`)
  - Subtitle: `t('status_awaiting_phase3_baseline', 'Awaiting Phase 3 Baseline')`
  - Footer: `t('anchor_label', 'Anchor: —')` with `t('status_pending_phase3', 'Pending Phase 3')`

### i18n Translation Contract
- Signature: `t(key: string, fallback?: string, params?: Record<string, string | number>): string`
- Fallback resolution: `UI_TRANSLATIONS[lang]?.[key] || UI_TRANSLATIONS.en[key] || fallback || key`
- Interpolation: Replace `{varName}` with `params[varName]`.

### Error Resilience Contract
- `App.tsx`: `Promise.allSettled` for telemetry calls. If non-fatal endpoints fail, display dismissable top banner with `Retry Connection` while keeping header and navigation accessible.
- `WorkspaceHub.tsx`: Dismissable status card with manual `Retry` button. Suppress periodic error alerts when dismissed by user.

## Code Layout
- `dashboard/src/components/CompetitionJourneyWorkspace.tsx`: Competition lifecycle, stage controls, agent context checkboxes, KPI cards.
- `dashboard/src/components/WorkspaceHub.tsx`: Workspace health check, account status, dismissable error banner.
- `dashboard/src/components/Header.tsx`: Navigation header, language switcher, global status.
- `dashboard/src/components/NodeInspector.tsx`: Experiment inspection panel, metric labels.
- `dashboard/src/components/TaskConfigModal.tsx`: Task configuration modal, validation split tab.
- `dashboard/src/components/CompetitionPipelineHub.tsx`: Pipeline stages, validation tab.
- `dashboard/src/components/LeaderboardTable.tsx`: Leaderboard columns, metric headers.
- `dashboard/src/constants/i18n.ts`: Translation dictionaries, language context, `t(...)` hook.
- `dashboard/src/App.tsx`: Top-level application shell, routing, telemetry loading, resilient error screen.
- `tools/dashboard_server.py`: Backend Python server.
- `tests/test_dashboard.py`: Dashboard server test suite.
