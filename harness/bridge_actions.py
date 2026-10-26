"""Action-handler mixin for ``AgentBridge`` (the 30 typed ``_handle_*`` methods).

Extracted verbatim from ``harness/agent_bridge.py`` (M4 refactoring,
PR-E). ``AgentBridge`` mixes this class in, so handler behavior —
dispatch, grounding-gate enforcement, audit writes — is unchanged.

Handlers lazy-import heavier harness modules inside their bodies, so
this module only needs the shared types from ``harness.bridge_common``
plus the modules used at handler scope.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from harness.bridge_common import ActionObservation, TypedActionName
from harness.grounding_gate import GroundingGate


class ActionHandlerMixin:
    """The 30 typed action handlers, moved verbatim from ``AgentBridge``."""

    def _handle_research_memory_group(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        memory = self._research_memory()
        if act == "record_macro_review":
            from harness.macro_research import record_macro_review
            result = record_macro_review(memory, params, session_id=session.session_id)
        elif act == "get_macro_hypothesis_map":
            from harness.macro_research import compute_macro_hypothesis_map
            result = compute_macro_hypothesis_map(memory, memory.sync())
        elif act == "check_retrospective_status":
            from harness.macro_research import retrospective_status
            from harness.stagnation import StagnationController
            result = retrospective_status(memory, memory.sync(), StagnationController(self.project).reconcile_results())
        elif act == "candidate_preflight":
            from harness.candidate_preflight import run_candidate_preflight
            result = run_candidate_preflight(
                self.project.root,
                candidate_path=params.get("candidate_path") or params.get("candidate") or params.get("path"),
                incumbent_path=params.get("incumbent_path") or params.get("incumbent") or params.get("baseline"),
                id_col=params.get("id_col"),
                pred_col=params.get("pred_col"),
            )
        elif act == "reconcile_submissions":
            from harness.reconciliation import reconcile_provider_submissions
            comp = params.get("competition") or (self.spec.get("competition", {}).get("id") if hasattr(self, "spec") and isinstance(self.spec, dict) else None) or "competition"
            result = reconcile_provider_submissions(self.project.root, competition=comp)
        elif act == "acknowledge_external_submissions":
            from harness.reconciliation import acknowledge_external_submissions
            result = acknowledge_external_submissions(
                self.project.root,
                refs=params.get("refs") or params.get("unexplained_refs") or [],
                reason=params.get("reason", "Operator acknowledged external run"),
                operator=params.get("operator"),
            )
        elif act == "register_recovery_probe":
            from harness.stagnation import StagnationController
            result = StagnationController(self.project).register_recovery_probe(**params)
        elif act == "search_research_memory":
            result = {"matches": memory.search(params.get("query", ""), params.get("limit", 8))}
        elif act == "record_research_insight":
            # Defensive extraction with rich aliases
            title = (
                params.get("title")
                or params.get("name")
                or params.get("insight")
                or params.get("summary")
            )
            observation = (
                params.get("observation")
                or params.get("hypothesis")
                or params.get("description")
                or params.get("content")
                or params.get("details")
                or params.get("text")
                or ""
            )
            if not title:
                title = (str(observation).split(".")[0][:80] if observation else "Untitled Insight")
            mechanism = (
                params.get("mechanism")
                or params.get("rationale")
                or params.get("reason")
                or params.get("hypothesis")
                or "Empirical mechanism"
            )
            next_action = (
                params.get("next_action")
                or params.get("action")
                or params.get("plan")
                or params.get("next_steps")
                or params.get("todo")
                or "Execute probe"
            )
            evidence = params.get("evidence")
            if isinstance(evidence, str) and " " in evidence and not any(e in evidence for e in [r["exp_id"] for r in memory.sync()]):
                observation = f"{observation} [Evidence reference: {evidence}]".strip()
                evidence = None
            result = memory.record_insight(
                title=title,
                observation=observation,
                mechanism=mechanism,
                next_action=next_action,
                evidence=evidence,
                parent_insight_id=params.get("parent_insight_id"),
                family_id=params.get("family_id", "unclassified"), source_refs=params.get("source_refs"),
                session_id=session.session_id,
            )
        elif act == "plan_research_probe":
            # Defensive extraction with aliases and sensible defaults
            control = (
                params.get("control")
                or params.get("baseline")
                or params.get("reference")
                or params.get("comparator")
                or "exp000_baseline"
            )
            intervention = (
                params.get("intervention")
                or params.get("treatment")
                or params.get("candidate")
                or params.get("experiment")
                or params.get("action")
                or params.get("description")
                or "Intervention probe"
            )
            expected_observation = (
                params.get("expected_observation")
                or params.get("expected_result")
                or params.get("expected")
                or params.get("expectation")
                or "Primary metric improves relative to control baseline"
            )
            alternative_explanation = (
                params.get("alternative_explanation")
                or params.get("alternative")
                or params.get("alternative_hypothesis")
                or params.get("fallback_explanation")
                or "Outcome may stem from sample variance or noise rather than mechanism"
            )
            decision_rule = (
                params.get("decision_rule")
                or params.get("rule")
                or params.get("criteria")
                or params.get("decision")
                or params.get("evaluation_rule")
                or "Proceed if metric improves"
            )
            budget = (
                params.get("budget")
                or params.get("max_budget")
                or params.get("probe_budget")
                or params.get("cost")
                or "1 run"
            )
            stop_rule = (
                params.get("stop_rule")
                or params.get("stop_criteria")
                or params.get("stopping_rule")
                or params.get("termination_rule")
                or "Stop upon regression or budget limit"
            )
            insight_id = (
                params.get("insight_id")
                or params.get("insight")
                or params.get("id")
                or params.get("parent_insight_id")
            )
            if not insight_id:
                # Find most recent insight from current research memory
                recent_insights = [e for e in memory.entries() if e.get("kind") == "insight"]
                if recent_insights:
                    insight_id = recent_insights[-1]["id"]
                else:
                    raise ValueError("plan_research_probe requires 'insight_id' linking to an existing insight")
            result = memory.plan_probe(
                insight_id=insight_id,
                control=control,
                intervention=intervention,
                expected_observation=expected_observation,
                alternative_explanation=alternative_explanation,
                decision_rule=decision_rule,
                budget=budget,
                stop_rule=stop_rule,
                session_id=session.session_id,
            )
        elif act == "trace_research_insight":
            insight_id = params.get("insight_id") or params.get("insight") or params.get("id")
            result = memory.trace(insight_id, params.get("offset", 0), params.get("limit", 20))
        elif act == "review_hypothesis":
            cfg = params.get("config")
            if cfg is None:
                path = (self.project_root / params["config_path"]).resolve()
                path.relative_to(self.project_root)
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            result = memory.review(cfg, params.get("revisit_reason", ""))
        elif act == "record_research_decision":
            # Defensive extraction with rich aliases
            from harness.research_memory import normalize_decision_outcome
            outcome = normalize_decision_outcome(
                params.get("outcome")
                or params.get("decision")
                or params.get("status")
                or params.get("result")
                or ""
            )
            conclusion = (
                params.get("conclusion")
                or params.get("lesson")
                or params.get("summary")
                or params.get("note")
                or params.get("finding")
                or params.get("description")
                or ""
            )
            lesson = (
                params.get("lesson")
                or params.get("conclusion")
                or params.get("summary")
                or ""
            )
            next_action = (
                params.get("next_action")
                or params.get("action")
                or params.get("plan")
                or params.get("todo")
                or params.get("next_steps")
                or ""
            )
            evidence = params.get("evidence")
            if isinstance(evidence, str) and " " in evidence and not any(e in evidence for e in [r["exp_id"] for r in memory.sync()]):
                extra_note = f"[Evidence reference: {evidence}]"
                conclusion = f"{conclusion} {extra_note}".strip()
                evidence = None
            result = memory.record_decision(
                outcome=outcome or None,
                conclusion=conclusion,
                evidence=evidence,
                next_action=next_action,
                insight_id=params.get("insight_id"),
                lesson=lesson,
                revisit_when=params.get("revisit_when") or "",
                session_id=session.session_id,
                rule_id=params.get("rule_id"),
                candidate_evidence=params.get("candidate_evidence") or (
                    {"id": params["candidate_evidence_id"], "kind": params.get("evidence_kind", "full_run"),
                     "partition": params.get("evidence_partition", "oof")} if params.get("candidate_evidence_id") else None),
                claimed_outcome=params.get("claimed_outcome"),
            )
        else:
            if not str(params.get("summary") or "").strip() or not str(params.get("next_action") or "").strip():
                raise ValueError(
                    "record_research_checkpoint requires 'summary' and 'next_action' "
                    "text fields (the checkpoint field is 'summary', not 'conclusion')"
                )
            result = memory.checkpoint(
                summary=params["summary"],
                next_action=params["next_action"],
                open_questions=params.get("open_questions", []),
                session_id=session.session_id,
                insight_ids=params.get("insight_ids"),
            )
        return ActionObservation(call.step_index, act, "SUCCESS", "Research memory action completed",
                                 data=result, elapsed_seconds=round(time.time() - t0, 3))

    def _handle_suggest_next_interventions(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        suggestions = self._suggest_next_interventions(limit=params.get("limit", 5))
        n = len(suggestions.get("suggestions", []))
        if suggestions["stagnation"].get("reassess_recommended"):
            summary = (f"Stagnation detected ({suggestions['stagnation']['consecutive_non_improvements']} "
                       f"consecutive non-improvements). {n} untried intervention(s) suggested.")
        else:
            summary = f"{n} untried intervention(s) suggested to continue brainstorming."
        return ActionObservation(call.step_index, act, "SUCCESS", summary,
                                 data=suggestions, elapsed_seconds=round(time.time() - t0, 3))

    def _handle_update_workspace_settings(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        if params.get("gates"):
            raise ValueError("Gates require their verified actions")
        for key in ("mode", "manual_phase_override", "agent_context_config"):
            if key in params:
                if key == "agent_context_config":
                    # Persist only known AgentContextConfig fields; arbitrary
                    # agent-supplied keys would later crash priming render.
                    from harness.competition_state import AgentContextConfig

                    stored = dict(
                        self.comp_state.state.get("agent_context_config")
                        or {}
                    )
                    for field_name in AgentContextConfig.__dataclass_fields__:
                        if field_name in params[key]:
                            stored[field_name] = params[key][field_name]
                    self.comp_state.state[key] = stored
                else:
                    self.comp_state.state[key] = params[key]
        cid = params.get("toggle_pin_candidate")
        if cid:
            pinned = self.comp_state.state["candidate_pool"][
                "pinned_candidates"
            ]
            if cid in pinned:
                pinned.remove(cid)
            else:
                pinned.append(cid)
        self.comp_state.save()
        return ActionObservation(
            call.step_index, act, "SUCCESS", "Workspace settings updated"
        )

    def _handle_update_task_contract(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.io_utils import atomic_write_text
        from harness.task_spec import UniversalTaskSpec

        document = params["task"]
        UniversalTaskSpec.from_dict({"task": document})
        atomic_write_text(
            self.project.task_config_path,
            yaml.safe_dump(
                {"task": document}, sort_keys=False, allow_unicode=True
            ),
        )
        self._refresh_task_context()
        return ActionObservation(
            call.step_index,
            act,
            "SUCCESS",
            "Task contract updated; evidence will be rechecked",
        )

    def _handle_write_experiment_config(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.io_utils import atomic_write_text

        path = (self.project_root / params["path"]).resolve()
        path.relative_to(self.project_root / "configs")
        if path == self.project.task_config_path:
            raise ValueError("Use update_task_contract")
        cfg = yaml.safe_load(params["content"])
        GroundingGate.verify_config(
            cfg, config_path=str(path), raise_on_error=True
        )
        exp = cfg.get("experiment", {}).get("id", path.stem)
        cfg.setdefault("experiment", {})["id"] = exp
        if params.get("insight_id"):
            cfg.setdefault("research", {})["insight_id"] = params["insight_id"]
        if (self.artifacts_dir / exp / "artifact_manifest.json").is_file():
            raise ValueError(
                "Use a new experiment ID to change a run configuration"
            )
        from harness.state_store import workspace_lock

        with workspace_lock(self.artifacts_dir):
            review = self._review_research_config(cfg, params)
            if review["revisit_reason"]:
                cfg.setdefault("research", {})["revisit_reason"] = review["revisit_reason"]
            atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            self._research_memory().propose(cfg, review["revisit_reason"], session.session_id)
        return ActionObservation(
            call.step_index,
            act,
            "SUCCESS",
            "Experiment config saved",
            artifact_path=str(path),
            data={"novelty_review": review},
        )

    def _handle_confirm_task_spec(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        self.comp_state.state["gates"]["spec_confirmed"] = True
        self.comp_state._append_audit_event(
            "TASK_SPEC_CONFIRMED",
            "A user approved the attached task specification.",
            {"task_config": str(self.project.task_config_path)},
        )
        self.comp_state.save()
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary="Task contract marked reviewed and confirmed.",
            artifact_path=str(
                self.project.task_config_path.relative_to(self.project_root)
            ).replace("\\", "/"),
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_run_eda(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.eda import run_eda

        result = run_eda(self.project_root, data_path=params.get("data_path"))
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"EDA profiled {len(result.get('reports', []))} dataset file(s) without copying raw rows.",
            artifact_path=result["paths"]["markdown"],
            data={
                "json_report": result["paths"]["json"],
                "reports": result.get("reports", []),
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_record_phase_output(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.phase_workflow import PhaseWorkflow

        phase_id = params.get("phase_id")
        response = params.get("response")
        response_file = params.get("response_file")
        if response is None and response_file:
            response_path = (self.project_root / response_file).resolve()
            response_path.relative_to(self.project_root)
            response = response_path.read_text(encoding="utf-8")
        if not isinstance(response, str):
            raise ValueError(
                "Provide `response` or a workspace-relative `response_file`."
            )
        workflow = PhaseWorkflow(
            self.project_root, self.project.task_config_path
        )
        record = workflow.record(
            str(phase_id or ""),
            response,
            sources=params.get("sources", []),
            artifacts=params.get("artifacts", []),
            metadata=params.get("metadata", {}),
        )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Recorded phase `{record['phase_id']}` response with {len(record['sources'])} source(s) and {len(record['artifacts'])} artifact(s).",
            artifact_path=record["response_path"],
            data=record,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_bootstrap_task(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from tools.bootstrap_task import bootstrap_competition_task

        dataset_dir = params.get("dataset_dir", "dataset")
        context_file = params.get("context_file", "docs/TASK.md")
        output_config = params.get("output_config", "configs/task.yaml")
        proj_root = params.get("project_root") or self.project_root
        res = bootstrap_competition_task(
            dataset_dir=dataset_dir,
            context_file=context_file,
            output_config=output_config,
            project_root=proj_root,
        )
        self._refresh_task_context()
        self.comp_state.state["gates"]["spec_confirmed"] = False
        self.comp_state.state["gates"]["data_risks_confirmed"] = False
        self.comp_state.save()
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Bootstrapped task '{res.get('task_name')}' ({res.get('modality')}, {res.get('task_type')}) with {res.get('samples_count')} samples.",
            data=res,
            artifact_path=res.get("config_file"),
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_inspect_data_readiness(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        report = self.comp_state.run_data_readiness_audit(params.get("config"))
        warns = [w.get("title", "") for w in report.get("warnings", [])]
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Executed {len(report.get('checks_performed', []))} declarative data checks; {len(warns)} warnings logged.",
            diagnostics_bottlenecks=warns,
            artifact_path=str(
                self.comp_state.data_report_file.relative_to(self.project_root)
            ).replace("\\", "/"),
            data={
                "checks_performed": report.get("checks_performed", []),
                "warnings": warns,
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_confirm_data_risks(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        if not self.comp_state.state["gates"].get("spec_confirmed"):
            raise ValueError(
                "Review and confirm the task specification before confirming data risks."
            )
        report = self.comp_state.get_data_readiness_report()
        if report.get("status") == "BLOCKED":
            raise ValueError(
                "The latest data readiness report is BLOCKED; fix its critical checks first."
            )
        self.comp_state.state["gates"]["data_risks_confirmed"] = True
        self.comp_state._append_audit_event(
            "DATA_RISKS_CONFIRMED",
            "A user reviewed and confirmed the latest data readiness report.",
            {
                "report_status": report.get("status"),
                "report_path": str(self.comp_state.data_report_file),
            },
        )
        self.comp_state.save()
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Data readiness report ({report.get('status')}) reviewed and confirmed.",
            artifact_path=str(
                self.comp_state.data_report_file.relative_to(self.project_root)
            ).replace("\\", "/"),
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_establish_baseline_anchor(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        gates = self.comp_state.state.get("gates", {})
        if not gates.get("spec_confirmed") or not gates.get(
            "data_risks_confirmed"
        ):
            raise ValueError(
                "Review the task contract and confirm data risks before establishing the baseline."
            )
        pool = self.comp_state.compute_candidate_pool(journal)
        default_base_id = (
            self.comp_state.state["gates"].get("baseline_exp_id")
            or self.comp_state.spec.get("baseline_anchor", {}).get(
                "default_exp_id"
            )
            or "baseline"
        )
        base_id = params.get("baseline_id") or pool.get(
            "baseline_id", default_base_id
        )
        from harness.baseline import validate_baseline_artifact

        baseline = validate_baseline_artifact(
            self.artifacts_dir,
            str(base_id),
            str(self.comp_state.spec.get("primary_metric") or "score"),
            task_spec=self.task_spec,
            plugin=self.task_plugin,
        )
        base_score = float(baseline["score"])
        self.comp_state.state["gates"]["baseline_locked"] = True
        self.comp_state.state["gates"]["baseline_exp_id"] = base_id
        self.comp_state.state["gates"]["baseline_evidence"] = baseline
        self.comp_state.save()
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Locked official baseline anchor '{base_id}' at {base_score:.4f}.",
            composite_score=base_score,
            metric_delta=0.0,
            fold_variance=None,
            artifact_path=str(
                Path(baseline["metrics_path"]).relative_to(self.project_root)
            ).replace("\\", "/"),
            data={
                "baseline_id": base_id,
                "baseline_score": base_score,
                "evidence": baseline,
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_inspect_experiment(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        pool = self.comp_state.compute_candidate_pool(journal)
        default_exp_id = (
            (pool.get("overall_best") or {}).get("id")
            or self.comp_state.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        exp_id = params.get("exp_id", default_exp_id)
        cand = next(
            (c for c in pool["all_candidates"] if c["id"] == exp_id), None
        ) or (pool["overall_best"] if pool["overall_best"] else {})
        cand_score = float(
            cand.get("composite_score")
            or cand.get("score")
            or self.comp_state.spec.get("primary_metric_baseline")
            or 0.0
        )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Inspected '{cand.get('id', exp_id)}': score={cand_score:.4f}, std={cand.get('fold_std', 0.015)}.",
            composite_score=cand_score,
            metric_delta=cand.get("delta_vs_baseline", 0.0),
            fold_variance=cand.get("fold_std", 0.015),
            data=cand,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_inspect_diagnostics(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        pool = self.comp_state.compute_candidate_pool(journal)
        default_exp_id = (
            (pool.get("overall_best") or {}).get("id")
            or self.comp_state.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        exp_id = params.get("exp_id", default_exp_id)
        diag_file = self.artifacts_dir / exp_id / "DIAGNOSTICS.md"
        node = journal.nodes.get(exp_id)
        bottlenecks = []
        if node and node.verified and isinstance(node.diagnostics_summary, dict):
            bottlenecks = node.diagnostics_summary.get(
                "critical_bottlenecks", []
            )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Found {len(bottlenecks)} verified diagnostic bottlenecks from '{exp_id}'; missing diagnostics are not evidence of no errors.",
            diagnostics_bottlenecks=bottlenecks,
            artifact_path=(
                str(diag_file.relative_to(self.project_root)).replace("\\", "/")
                if diag_file.exists()
                else None
            ),
            data={"exp_id": exp_id, "bottlenecks": bottlenecks},
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_create_experiment(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        pool = self.comp_state.compute_candidate_pool(journal)
        branch = params.get("branch", "feat")
        descriptor = params.get("descriptor", "iteration")
        default_parent = (
            (pool.get("overall_best") or {}).get("id")
            or self.comp_state.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        parent_id = params.get("parent_id", default_parent)
        configs_dir = self.project_root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        exp_id = params.get("exp_id")
        auto_exp_note = ""
        if not exp_id:
            next_num = self.get_next_experiment_num()
            exp_id = f"exp{next_num:03d}_{branch}_{descriptor}"
        else:
            # A requested ID must never overwrite immutable artifacts. Instead of
            # failing, auto-assign the next free ID and report it in the result.
            collision_dir = None
            for hidden in [".ai-harness", ".ml-harness"]:
                art_root = self.project_root / hidden / "artifacts"
                if not art_root.exists():
                    continue
                collision_dir = next(
                    (
                        cand_dir
                        for cand_dir in art_root.iterdir()
                        if cand_dir.is_dir() and (cand_dir / exp_id).exists()
                    ),
                    None,
                )
                if collision_dir is not None:
                    break
            if collision_dir is not None:
                requested_exp_id = exp_id
                next_num = self.get_next_experiment_num()
                exp_id = f"exp{next_num:03d}_{branch}_{descriptor}"
                auto_exp_note = (
                    f"Requested ID '{requested_exp_id}' already has immutable artifacts "
                    f"at '{collision_dir / requested_exp_id}' (completed runs are "
                    f"immutable under the Correctness Contract); auto-assigned "
                    f"'{exp_id}' instead. "
                )

        cfg_path = configs_dir / f"{exp_id}.yaml"
        uncertainty = str(params.get("uncertainty_level", "low")).lower()
        task_name = self.comp_state.spec.get("name", "Competition Task")
        primary_metric = self.comp_state.spec.get("primary_metric", "score")
        parent_node = (
            journal.get_node(parent_id)
            if hasattr(journal, "get_node")
            else journal.nodes.get(parent_id)
        )
        parent_score = (
            parent_node.get_score()
            if parent_node
            else float(
                self.comp_state.spec.get("primary_metric_baseline") or 0.0
            )
        )

        modality = (
            getattr(self.task_spec, "modality", None)
            or self.comp_state.spec.get("modality", "")
        ).lower()
        task_type = (
            getattr(self.task_spec, "task_type", None)
            or self.comp_state.spec.get("task_type", "")
        ).lower()
        is_reg = (
            task_type in ("regression", "ranking")
            or "regression" in task_type
        )

        if "input_variants" in params:
            input_variants = params["input_variants"]
        elif (
            hasattr(self, "task_spec")
            and self.task_spec
            and getattr(self.task_spec, "inputs_dict", None)
            and self.task_spec.inputs_dict
        ):
            input_variants = list(self.task_spec.inputs_dict.keys())
        else:
            input_variants = ["default"]

        # Resolve model configuration dynamically from params or task_spec
        spec_model_cfg = None
        if hasattr(self, "task_spec") and self.task_spec:
            spec_meta = getattr(self.task_spec, "metadata", {}) or {}
            spec_model_cfg = (
                spec_meta.get("model")
                or spec_meta.get("default_model")
                or spec_meta.get("baseline_model")
            )
        if not spec_model_cfg and hasattr(self, "comp_state") and self.comp_state:
            spec_model_cfg = (
                self.comp_state.spec.get("model")
                or self.comp_state.spec.get("default_model")
                or self.comp_state.spec.get("baseline_model")
            )

        spec_name = None
        if isinstance(spec_model_cfg, dict):
            spec_name = spec_model_cfg.get("name")
        elif isinstance(spec_model_cfg, str):
            spec_name = spec_model_cfg

        params_model_cfg = params.get("model")
        if isinstance(params_model_cfg, dict):
            model_payload = dict(params_model_cfg)
            if "model_params" in params and isinstance(params["model_params"], dict):
                model_payload.setdefault("params", {}).update(params["model_params"])
        elif isinstance(params_model_cfg, str):
            model_payload = {
                "name": params_model_cfg,
                "params": dict(params.get("model_params", {})),
            }
        elif "model_name" in params:
            model_payload = {
                "name": params["model_name"],
                "params": dict(params.get("model_params", {})),
            }
        elif spec_model_cfg:
            if isinstance(spec_model_cfg, dict):
                model_payload = dict(spec_model_cfg)
            else:
                model_payload = {"name": spec_model_cfg, "params": {}}
            if "model_params" in params and isinstance(params["model_params"], dict):
                model_payload.setdefault("params", {}).update(params["model_params"])
        elif modality == "tabular":
            model_payload = {
                "name": "lightgbm",
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                },
            }
            if "model_params" in params and isinstance(params["model_params"], dict):
                model_payload["params"].update(params["model_params"])
        elif modality in {"text", "nlp"}:
            if is_reg:
                model_payload = {
                    "name": "tfidf_ridge",
                    "params": {"alpha": 1.0},
                }
            else:
                model_payload = {
                    "name": "tfidf_logistic_regression",
                    "params": {
                        "C": 1.5,
                    },
                }
            if "model_params" in params and isinstance(params["model_params"], dict):
                model_payload["params"].update(params["model_params"])
        else:
            model_payload = {
                "name": "baseline",
                "params": {},
            }
            if "model_params" in params and isinstance(params["model_params"], dict):
                model_payload["params"].update(params["model_params"])

        if "name" not in model_payload and "factory" not in model_payload:
            model_payload["name"] = spec_name or ("lightgbm" if modality == "tabular" else "baseline")
        if "params" not in model_payload:
            model_payload["params"] = {}

        # Strict GroundingGate-compliant YAML structure (task-agnostic defaults)
        cfg_payload: Dict[str, Any] = {
            "experiment": {
                "id": exp_id,
                "parent_id": parent_id,
                "branch": branch,
                "problem_addressed": params.get(
                    "problem_addressed",
                    f"Test a declared intervention on '{task_name}' ({primary_metric}); inspect verified parent diagnostics before implementation.",
                ),
                "hypothesis": params.get(
                    "hypothesis",
                    f"Targeted '{branch}' intervention ({descriptor}) improves '{primary_metric}' over parent '{parent_id}' by >= {self._get_spec_min_improvement(session):.4f}.",
                ),
                "intervention": params.get(
                    "intervention",
                    f"Apply '{descriptor}' configuration under branch '{branch}' while keeping CV split alignment fixed.",
                ),
                "description": f"Created via AgentBridge session {session.session_id}",
            },
            "grounding": {
                "source_type": params.get("source_type", "empirical_journal"),
                "source_id": params.get(
                    "source_id", "research_memory:unverified_proposal"
                ),
                "retrieval_summary": params.get(
                    "retrieval_summary",
                    f"Declared proposal targeting {primary_metric} on {task_name}; mechanism and source require scientist review.",
                ),
                # Anti-hallucination: do NOT fabricate a 'verified' channel.
                # If the agent did not supply a real verified_via, mark the
                # source as NOT verified so the GroundingGate / execution gate
                # can refuse to trust an ungrounded proposal. A fabricated
                # 'local_registry' here is exactly the bypass the gate exists
                # to stop (see GroundingGate.VERIFIED_VIA comment).
                "verified_via": params.get("verified_via", "not_verified"),
            },
            "assumption_register": {
                "known_facts": params.get(
                    "known_facts",
                    [
                        (f"Verified parent '{parent_id}' achieved {primary_metric}={parent_score:.4f}."
                         if parent_node and parent_node.verified else
                         f"Task contract declares primary metric '{primary_metric}'; parent performance is not verified."),
                    ],
                ),
                "assumptions": params.get(
                    "assumptions",
                    [
                        f"Intervention '{descriptor}' addresses primary error slices without increasing fold variance.",
                    ],
                ),
                "uncertainty_level": uncertainty,
                "verification_probe": params.get(
                    "verification_probe",
                    f"Run single-fold smoke test: ai-harness-run --project-root . --task-config {self.project.task_config_path.relative_to(self.project_root)} --config configs/{exp_id}.yaml --debug --fold 0",
                ),
            },
            "data": {
                "folds": params.get("folds", [0, 1, 2, 3, 4]),
                "input_variants": input_variants,
                "selection_variant": params.get("selection_variant", input_variants[0]),
            },
            "model": model_payload,
        }

        # Pre-verify with GroundingGate. B1-a: do NOT raise here so the
        # config is still written (keeping the create_experiment workflow
        # intact); instead we surface the grounding state in the
        # observation. The real circuit breaker is harness.runner.run(),
        # which still calls GroundingGate.verify_config(raise_on_error=True)
        # at the start of _execute_run() and refuses to train on an
        # unverified non-baseline proposal. This way an agent cannot create
        # a config with a fabricated 'verified_via' AND run it.
        grounding_ok, grounding_errors, grounding_receipt = GroundingGate.verify_config(
            cfg_payload, config_path=cfg_path, raise_on_error=False
        )
        from harness.io_utils import atomic_write_text
        from harness.state_store import workspace_lock

        if params.get("insight_id"):
            cfg_payload.setdefault("research", {})["insight_id"] = params["insight_id"]
        cfg_path.resolve().relative_to(configs_dir.resolve())
        if cfg_path.exists() or (self.artifacts_dir / exp_id / "artifact_manifest.json").exists():
            raise ValueError("Experiment ID already exists; choose a new ID")
        with workspace_lock(self.artifacts_dir):
            if cfg_path.exists():
                raise ValueError("Experiment ID already exists; choose a new ID")
            review = self._review_research_config(cfg_payload, params)
            if review["revisit_reason"]:
                cfg_payload.setdefault("research", {})["revisit_reason"] = review["revisit_reason"]
            atomic_write_text(cfg_path, yaml.safe_dump(cfg_payload, sort_keys=False, allow_unicode=True))
            self._research_memory().propose(cfg_payload, review["revisit_reason"], session.session_id)
        rel_cfg = str(cfg_path.relative_to(self.project_root)).replace(
            "\\", "/"
        )

        if grounding_ok:
            grounding_status = "PASSED"
            grounding_verification = "declared_source"
            grounding_summary = (
                f"Created config '{rel_cfg}' with declared grounding (Parent: {parent_id})."
            )
        else:
            grounding_status = "UNVERIFIED"
            grounding_verification = "unverified_placeholder"
            grounding_summary = (
                f"Created config '{rel_cfg}' (Parent: {parent_id}) but the grounding source is "
                f"NOT verified. The runner will REFUSE to train this until you supply a real "
                f"'grounding.source_id' + 'grounding.verified_via' (verify the source first). "
                f"Grounding issues: {grounding_errors}"
            )
            next_required_step = (
                f"Grounding required before run_smoke_test / run_full_validation (the runner "
                f"circuit-breaks on 'not_verified'). Verify a real source with your host "
                f"tools, then update the 'grounding' block in {rel_cfg} with source_id + "
                f"verified_via in one of: search_web | read_url_content | huggingface_hub | "
                f"mcp_gemini-docs | local_registry | verified_baseline. If the parent "
                f"baseline was verified locally, cite it (verified_via: verified_baseline). "
                f"Check with: python harness/grounding_gate.py --config {rel_cfg}"
            )

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=(auto_exp_note + grounding_summary).strip(),
            artifact_path=rel_cfg,
            data={
                "exp_id": exp_id,
                "config_path": rel_cfg,
                "auto_assigned_exp_id": exp_id if auto_exp_note else None,
                "grounding_status": grounding_status,
                "grounding_verification": grounding_verification,
                "grounding_errors": grounding_errors,
                "next_required_step": next_required_step
                if grounding_status == "UNVERIFIED"
                else None,
                "novelty_review": review,
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_run_smoke_test(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        cfg_path = params.get("config_path")
        if not cfg_path:
            available = sorted(
                list((self.project_root / "configs").glob("exp*.yaml"))
            )
            cfg_path = (
                str(available[0].relative_to(self.project_root)).replace(
                    "\\", "/"
                )
                if available
                else "configs/baseline.yaml"
            )
        # Defense-in-depth (same as run_full_validation): reject config paths
        # that resolve outside the project root.
        cfg_path = str(self._validate_config_path(cfg_path))
        fold = str(params.get("fold", 0))
        cmd = [
            sys.executable,
            "-m",
            "harness.runner",
            "--project-root",
            str(self.project_root),
            "--task-config",
            str(self.project.task_config_path),
            "--config",
            cfg_path,
            "--debug",
            "--fold",
            fold,
            "--skip-leaderboard",
        ]
        passed, retcode, out_text, elapsed = self._run_subprocess_safe(
            cmd, timeout=min(180.0, float(params.get("_deadline_seconds", 180)))
        )
        # Only count successful smoke runs toward the budget: a FAILED smoke
        # test is a diagnostic signal, and counting it inflated smoke_runs
        # while also consuming the budget meant to bound real probe cycles.
        if passed:
            session.compute_usage["smoke_runs"] = (
                int(session.compute_usage.get("smoke_runs", 0)) + 1
            )
        failure_info = None
        if not passed:
            from harness.failure_classifier import classify_failure

            failure_info = classify_failure(out_text, returncode=retcode)
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS" if passed else "FAILED",
            summary=f"Smoke test on Fold {fold} ({cfg_path}): {'PASSED' if passed else 'FAILED'}.",
            artifact_path=cfg_path,
            data={"returncode": retcode, "tail": out_text[-500:]},
            elapsed_seconds=elapsed,
            failure=failure_info,
        )

    def _handle_run_probe(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        cfg_path = params.get("config_path")
        if not cfg_path:
            available = sorted(
                list((self.project_root / "configs").glob("exp*.yaml"))
            )
            cfg_path = (
                str(available[0].relative_to(self.project_root)).replace(
                    "\\", "/"
                )
                if available
                else "configs/baseline.yaml"
            )
        # Defense-in-depth (same as run_full_validation): reject config paths
        # that resolve outside the project root.
        cfg_path = str(self._validate_config_path(cfg_path))
        probe_samples = str(params.get("probe_samples", 100))
        fold = str(params.get("fold", 0))
        cmd = [
            sys.executable,
            "-m",
            "harness.runner",
            "--project-root",
            str(self.project_root),
            "--task-config",
            str(self.project.task_config_path),
            "--config",
            cfg_path,
            "--debug",
            "--fold",
            fold,
            "--probe-samples",
            probe_samples,
            "--skip-leaderboard",
        ]
        passed, retcode, out_text, elapsed = self._run_subprocess_safe(
            cmd, timeout=min(120.0, float(params.get("_deadline_seconds", 120)))
        )
        session.compute_usage["smoke_runs"] = (
            int(session.compute_usage.get("smoke_runs", 0)) + 1
        )
        failure_info = None
        if not passed:
            from harness.failure_classifier import classify_failure

            failure_info = classify_failure(out_text, returncode=retcode)
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS" if passed else "FAILED",
            summary=f"Fast Probe run ({probe_samples} samples, Fold {fold}, {cfg_path}): {'PASSED' if passed else 'FAILED'}.",
            artifact_path=cfg_path,
            data={"returncode": retcode, "tail": out_text[-500:]},
            elapsed_seconds=elapsed,
            failure=failure_info,
        )

    def _handle_run_full_validation(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        cfg_path = params.get("config_path")
        if not cfg_path:
            raise ValueError(
                "Parameter 'config_path' is required for 'run_full_validation'."
            )
        self._validate_config_path(cfg_path)

        backend = str(params.get("backend", "local")).lower()
        force = bool(params.get("force", False))

        full_cfg_file = self.project_root / cfg_path
        raw_cfg = (
            yaml.safe_load(full_cfg_file.read_text(encoding="utf-8"))
            if full_cfg_file.exists()
            else {}
        )
        exp_id = raw_cfg.get("experiment", {}).get("id", Path(cfg_path).stem)
        review = self._review_research_config(raw_cfg, params, for_execution=True)
        # Advisory (never blocks): same model family as earlier work.
        mechanism_warnings = [
            f"{m.get('exp_id')} ({m.get('outcome') or m.get('evidence_status')}: "
            f"{m.get('score') if m.get('score') is not None else 'n/a'})"
            for m in (review.get("mechanism_matches") or [])
        ]

        if backend in ("modal", "kaggle"):
            from harness.cloud_job_runner import CloudJobManager

            manager = CloudJobManager(self.project_root)
            job = manager.start_job(
                backend=backend,
                config_path=cfg_path,
                gpu=str(params.get("gpu", "A10")),
                auto_audit=bool(params.get("auto_audit", True)),
                auto_sync=True,
                baseline_csv=params.get("baseline_csv"),
            )
            session.compute_usage["experiments_used"] = (
                int(session.compute_usage.get("experiments_used", 0)) + 1
            )
            session.compute_usage["cloud_runs"] = (
                int(session.compute_usage.get("cloud_runs", 0)) + 1
            )
            if exp_id not in session.experiments_run:
                session.experiments_run.append(exp_id)
            return ActionObservation(
                step_index=call.step_index,
                action=act,
                status="SUCCESS",
                summary=f"Dispatched {backend} job `{job.job_id}` for experiment `{exp_id}`; result import waits for a terminal provider state.",
                artifact_path=job.artifact_dir,
                data={
                    "job_id": job.job_id,
                    "backend": backend,
                    "status": job.status,
                    "provider_status": job.provider_status,
                    "exp_id": exp_id,
                },
                elapsed_seconds=round(time.time() - t0, 3),
            )

        if backend != "local":
            raise ValueError(f"Unsupported execution backend: {backend}")
        cmd = [
            sys.executable,
            "-m",
            "harness.runner",
            "--project-root",
            str(self.project_root),
            "--task-config",
            str(self.project.task_config_path),
            "--config",
            cfg_path,
            "--backend",
            "local",
        ]
        if force:
            cmd.append("--force")

        passed, retcode, out_text, elapsed = self._run_subprocess_safe(
            cmd,
            timeout=min(1800.0, float(params.get("_deadline_seconds", 1800))),
        )

        # Update compute usage
        session.compute_usage["experiments_used"] = (
            int(session.compute_usage.get("experiments_used", 0)) + 1
        )
        session.compute_usage["local_runs"] = (
            int(session.compute_usage.get("local_runs", 0)) + 1
        )

        if exp_id not in session.experiments_run:
            session.experiments_run.append(exp_id)

        max_rej = int(session.budget.get("max_consecutive_rejections") or 3)
        if not passed:
            session.consecutive_rejections += 1
            session.reassess_recommended = (
                session.consecutive_rejections >= max_rej
            )
            from harness.failure_classifier import classify_failure
            return ActionObservation(
                step_index=call.step_index,
                action=act,
                status="FAILED",
                summary=f"Full validation ({backend}) failed for '{exp_id}'.",
                advisory=(
                    "REASSESS_RECOMMENDED"
                    if session.reassess_recommended
                    else None
                ),
                artifact_path=str(
                    (self.artifacts_dir / exp_id).relative_to(self.project_root)
                ).replace("\\", "/"),
                data={
                    "returncode": retcode,
                    "tail": out_text[-600:],
                    **({"mechanism_warnings": mechanism_warnings} if mechanism_warnings else {}),
                },
                elapsed_seconds=elapsed,
                failure=classify_failure(out_text, returncode=retcode),
            )

        # Reload TreeJournal and evaluate metrics
        journal.sync_from_artifacts(self.artifacts_dir)
        pool = self.comp_state.compute_candidate_pool(journal)
        base_score = float(
            pool.get("baseline_score")
            or self.comp_state.spec.get("primary_metric_baseline")
            or 0.0
        )
        oof_path = self.artifacts_dir / exp_id / "oof_predictions.csv"

        if oof_path.exists():
            comp_score, detail_scores = self._evaluate_oof_dataframe(oof_path)
        else:
            node = journal.get_node(exp_id)
            comp_score = node.get_score() if node else 0.0
            detail_scores = {"composite_score": comp_score}

        delta = round(
            (comp_score - base_score)
            * (1 if self.task_spec.direction == "maximize" else -1),
            4,
        )
        min_imp = self._get_spec_min_improvement(session)
        verified_runs = self._research_memory().sync()
        current = next((r for r in verified_runs if r["exp_id"] == exp_id), None)
        comparable = [r["utility"] for r in verified_runs if current and
                      r["exp_id"] != exp_id and r["comparison_group"] == current["comparison_group"]]
        incumbent_gain = current["utility"] - max(comparable) if current and comparable else None
        if incumbent_gain is None or (incumbent_gain > 0 and incumbent_gain >= min_imp):
            session.consecutive_rejections = 0
        else:
            session.consecutive_rejections += 1
        session.reassess_recommended = session.consecutive_rejections >= max_rej

        diag_json = self.artifacts_dir / exp_id / "diagnostics.json"
        bottlenecks = []
        if diag_json.exists():
            try:
                bottlenecks = json.loads(
                    diag_json.read_text(encoding="utf-8")
                ).get("critical_bottlenecks", [])
            except Exception:
                pass

        validation_cfg = self.comp_state.spec.get("validation") or {}
        data_cfg = (
            self.comp_state.spec.get("data_config")
            or self.comp_state.spec.get("data")
            or {}
        )
        validation_strategy = (
            validation_cfg.get("strategy")
            or validation_cfg.get("scheme")
            or data_cfg.get("split_strategy")
            or "configured validation"
        )
        fold_count = (
            validation_cfg.get("n_folds")
            or validation_cfg.get("num_folds")
            or data_cfg.get("n_folds")
        )
        validation_label = (
            f"{validation_strategy} ({fold_count} folds)"
            if fold_count
            else str(validation_strategy)
        )

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Completed {validation_label} ({backend}) for '{exp_id}': Composite={comp_score:.4f} (Δ={delta:+.4f} vs baseline).",
            composite_score=comp_score,
            metric_delta=delta,
            diagnostics_bottlenecks=bottlenecks,
            advisory=(
                "REASSESS_RECOMMENDED" if session.reassess_recommended else None
            ),
            artifact_path=str(oof_path.relative_to(self.project_root)).replace(
                "\\", "/"
            ),
            data={
                "exp_id": exp_id,
                "backend": backend,
                "scores": detail_scores,
                "incumbent_utility_gain": incumbent_gain,
                **({"mechanism_warnings": mechanism_warnings} if mechanism_warnings else {}),
            },
            elapsed_seconds=elapsed,
        )

    def _handle_inspect_cloud_job_or_wait_cloud_job(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.cloud_job_runner import CloudJobManager

        job_id = params.get("job_id")
        if not job_id:
            raise ValueError(f"Parameter `job_id` is required for `{act}`.")
        manager = CloudJobManager(self.project_root)
        wait_seconds = min(
            3600.0, max(0.0, float(params.get("timeout_seconds", 0)))
        )
        poll_seconds = min(
            30.0, max(0.25, float(params.get("poll_interval", 3)))
        )
        deadline = time.monotonic() + wait_seconds
        job = manager.get_job(str(job_id))
        if not job:
            if "/" in str(job_id) or str(job_id).startswith("kernel-"):
                from harness.templates.kaggle_template import KaggleKernelManager

                slug = str(job_id)
                effective_timeout = wait_seconds if wait_seconds > 0 else 3600.0
                if act == TypedActionName.WAIT_CLOUD_JOB.value:
                    k_res = KaggleKernelManager.wait_for_kernel(
                        exp_id="",
                        kernel_slug=slug,
                        poll_interval=poll_seconds,
                        timeout_seconds=effective_timeout,
                    )
                else:
                    k_res = KaggleKernelManager.get_kernel_status(exp_id="", kernel_slug=slug)

                status = k_res.get("status", "UNKNOWN")
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status=(
                        "SUCCESS"
                        if status not in {"FAILED", "CANCELLED", "ERROR", "TIMEOUT"}
                        else "FAILED"
                    ),
                    summary=f"Cloud kernel `{slug}` is {status}.",
                    data={"job": {"job_id": slug, "kernel_slug": slug, "provider": "kaggle", "status": status, **k_res}},
                    elapsed_seconds=round(time.time() - t0, 3),
                )
            raise ValueError(
                f"Cloud job `{job_id}` was not found in this task workspace."
            )
        while (
            act == TypedActionName.WAIT_CLOUD_JOB.value
            and job.get("status") in {"PENDING", "RUNNING"}
            and time.monotonic() < deadline
        ):
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
            job = manager.get_job(str(job_id)) or job
        artifact_data: Dict[str, Any] = {}
        if job.get("status") == "COMPLETED":
            artifact_dir = self.project_root / str(
                job.get("artifact_dir") or ""
            )
            metrics_file = artifact_dir / "metrics.json"
            if metrics_file.is_file():
                try:
                    artifact_data["metrics"] = json.loads(
                        metrics_file.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    artifact_data["metrics_error"] = (
                        "metrics.json is not valid JSON"
                    )
            artifact_data["artifact_paths"] = (
                [
                    str(path.relative_to(self.project_root)).replace("\\", "/")
                    for path in artifact_dir.iterdir()
                    if path.is_file()
                ]
                if artifact_dir.is_dir()
                else []
            )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status=(
                "SUCCESS"
                if job.get("status") not in {"FAILED", "CANCELLED"}
                else "FAILED"
            ),
            summary=f"Cloud job `{job_id}` is {job.get('status')}; artifact identity verified={job.get('artifact_identity_verified', False)}.",
            artifact_path=job.get("artifact_dir"),
            data={"job": job, **artifact_data},
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_synthesize_candidate_or_promote_candidate_or_build_ensemble(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        name = params.get("name", f"{session.session_id}_router")
        mode = params.get("mode", "specialist_router")
        source_map = params.get("source_map")
        if act == TypedActionName.PROMOTE_CANDIDATE.value:
            if not params.get("exp_id"):
                raise ValueError("Promotion requires exp_id")
            mode = "single_model"
            source_map = {"source_exp": params["exp_id"]}
        if not source_map:
            raise ValueError("Explicit source_map is required")
        # Anti-duplicate-submission guard: an agent may re-synthesize the
        # same recipe under a new display name to "preserve" a submission
        # or pad the candidate pool. Reject an identical recipe unless the
        # agent explicitly justifies a revisit (the recipe is otherwise
        # byte-identical, so a new name adds no scientific signal).
        from harness.candidates import _candidate_recipe_fingerprint, _stored_recipe_fingerprint

        new_fp = _candidate_recipe_fingerprint(mode, self._candidate_source_ids(mode, source_map), source_map)
        existing_fps = {
            _stored_recipe_fingerprint(c): c["id"]
            for c in self.comp_state.state["candidate_pool"].get("synthesized_candidates", [])
            if c.get("recipe_fingerprint")
        }
        if new_fp in existing_fps and not str(params.get("revisit_reason", "")).strip():
            raise ValueError(
                f"Candidate recipe already exists as '{existing_fps[new_fp]}'. "
                f"Re-synthesizing the same sources/mode/weights under a new name does not "
                f"create a new submission. Provide 'revisit_reason' to intentionally revisit "
                f"this recipe, or choose different sources." if act != "promote_candidate" else
                f"Candidate for this experiment is already in the pool as '{existing_fps[new_fp]}'"
            )
        entry = self.comp_state.synthesize_candidate(
            name=name, mode=mode, source_map=source_map
        )
        session.compute_usage["candidate_operations"] = (
            int(session.compute_usage.get("candidate_operations", 0)) + 1
        )

        oof_abs = self.project_root / entry["oof_path"]
        synth_score, detail_scores = self._evaluate_oof_dataframe(oof_abs)
        base_score = float(
            self.comp_state.compute_candidate_pool(journal).get(
                "baseline_score"
            )
            or self.comp_state.spec.get("primary_metric_baseline")
            or 0.0
        )
        delta = round(
            (synth_score - base_score)
            * (1 if self.task_spec.direction == "maximize" else -1),
            4,
        )

        min_imp = self._get_spec_min_improvement(session)
        max_rej = int(session.budget.get("max_consecutive_rejections") or 3)
        if delta >= min_imp:
            session.consecutive_rejections = 0
        else:
            session.consecutive_rejections += 1
        session.reassess_recommended = session.consecutive_rejections >= max_rej

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Synthesized candidate '{entry['id']}' via '{mode}' (Real OOF Composite={synth_score:.4f}, Δ={delta:+.4f}).",
            composite_score=synth_score,
            metric_delta=delta,
            fold_variance=None,
            advisory=(
                "REASSESS_RECOMMENDED" if session.reassess_recommended else None
            ),
            artifact_path=entry["oof_path"],
            data={**entry, "scores": detail_scores},
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_register_external_prediction(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.candidates import register_external_prediction

        name = params.get("name") or params.get("exp_id")
        if not name:
            raise ValueError("Action 'register_external_prediction' requires 'name' (or 'exp_id').")
        oof_file = params.get("oof_file") or params.get("oof_path")
        if not oof_file:
            raise ValueError("Action 'register_external_prediction' requires 'oof_file'.")
        test_file = params.get("test_file") or params.get("test_path")

        res = register_external_prediction(
            project=self.project,
            name=name,
            oof_path=oof_file,
            test_path=test_file,
            author=params.get("author", "grandmaster"),
            source=params.get("source", "external_import"),
            notes=params.get("notes", ""),
        )
        session.compute_usage["candidate_operations"] = (
            int(session.compute_usage.get("candidate_operations", 0)) + 1
        )
        cand_score = float(res.get("primary_score", 0.0))
        base_score = float(
            self.comp_state.compute_candidate_pool(journal).get("baseline_score")
            or self.comp_state.spec.get("primary_metric_baseline")
            or 0.0
        )
        delta = round(
            (cand_score - base_score) * (1 if self.task_spec.direction == "maximize" else -1),
            4,
        )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Registered external prediction '{name}' ({res.get('metric')}={cand_score:.4f}, Δ={delta:+.4f}, rows={res.get('rows')}).",
            composite_score=cand_score,
            metric_delta=delta,
            artifact_path=res.get("artifact_dir"),
            data=res,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_optimize_blend(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.candidates import find_optimal_blend_weights

        components = params.get("components") or params.get("component_ids") or []
        if isinstance(components, str):
            try:
                components = json.loads(components)
            except Exception:
                components = [c.strip() for c in components.split(",") if c.strip()]
        if len(components) < 2:
            raise ValueError("Action 'optimize_blend' requires at least 2 experiment IDs in 'components'.")

        blend_mode = params.get("mode") or params.get("blend_mode")
        opt_res = find_optimal_blend_weights(
            project=self.project,
            component_ids=components,
            blend_mode=blend_mode,
            maxiter=int(params.get("maxiter", 150)),
        )

        create_cand = params.get("create_candidate", True)
        cand_entry = None
        if create_cand and opt_res.get("superiority_satisfied"):
            name = params.get("name", f"blend_{'_'.join(components[:3])}")
            cand_entry = self.comp_state.synthesize_candidate(
                name=name,
                mode=opt_res["blend_mode"],
                source_map={
                    "components": components,
                    "weights": opt_res["optimal_weights"],
                },
            )
            session.compute_usage["candidate_operations"] = (
                int(session.compute_usage.get("candidate_operations", 0)) + 1
            )

        if cand_entry:
            summary_msg = f"Optimized blend candidate '{cand_entry['id']}' ({opt_res['blend_mode']}) at {opt_res['best_score']:.4f} (Δ vs best constituent={opt_res['gain_vs_best_constituent']:+.4f})."
            art_path = cand_entry.get("oof_path")
        else:
            summary_msg = f"Blend optimization found weights {opt_res['weights_dict']}, best_score={opt_res['best_score']:.4f} (Δ vs best={opt_res['gain_vs_best_constituent']:+.4f})."
            art_path = None

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=summary_msg,
            composite_score=opt_res["best_score"],
            metric_delta=opt_res["gain_vs_best_constituent"],
            artifact_path=art_path,
            data={"optimization": opt_res, "candidate": cand_entry},
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_optimize_threshold(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.endgame_submission import ThresholdOptimizer

        pool = self.comp_state.compute_candidate_pool(journal)
        default_exp = (
            (pool.get("overall_best") or {}).get("id")
            or self.comp_state.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        exp_id = params.get("exp_id", default_exp)
        from harness.threshold_data import load_binary_threshold_data

        y_true, y_prob, source_metadata = load_binary_threshold_data(
            self.project, self.task_spec, self.task_plugin, exp_id
        )

        metric = params.get("metric", "f1")
        num_th = int(params.get("num_thresholds", 181))
        best_th, best_score = ThresholdOptimizer.optimize_threshold(
            y_true=y_true,
            y_prob=y_prob,
            metric=metric,
            num_thresholds=num_th,
        )

        preds_05 = (y_prob >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
        m_lower = metric.lower()
        if m_lower in {"kappa", "cohen_kappa"}:
            base_05_score = float(cohen_kappa_score(y_true, preds_05))
        elif m_lower in {"accuracy", "acc"}:
            base_05_score = float(accuracy_score(y_true, preds_05))
        else:
            base_05_score = float(f1_score(y_true, preds_05, zero_division=0))
        delta = round(best_score - base_05_score, 4)

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Optimized decision threshold for '{exp_id}': best_th={best_th:.3f} ({metric}={best_score:.4f}, Δ vs 0.5={delta:+.4f}).",
            composite_score=best_score,
            metric_delta=delta,
            data={
                "exp_id": exp_id,
                "metric": metric,
                "best_threshold": best_th,
                "best_score": round(best_score, 5),
                "baseline_score_at_05": round(base_05_score, 5),
                "gain": delta,
                **source_metadata,
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_generate_microdose_probe(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        from harness.endgame_submission import MicroDoseProbeGenerator

        base = params.get("base") or params.get("base_exp_id") or params.get("base_csv")
        probe = params.get("probe") or params.get("probe_exp_id") or params.get("probe_csv")
        if not base or not probe:
            raise ValueError("Action 'generate_microdose_probe' requires 'base' and 'probe'.")

        base_path = Path(base)
        if not base_path.exists():
            cand = self.artifacts_dir / base / "test_predictions.csv"
            if cand.is_file():
                base_path = cand
            else:
                raise FileNotFoundError(f"Base submission file not found: {base}")

        probe_path = Path(probe)
        if not probe_path.exists():
            cand = self.artifacts_dir / probe / "test_predictions.csv"
            if cand.is_file():
                probe_path = cand
            else:
                raise FileNotFoundError(f"Probe submission file not found: {probe}")

        alpha = float(params.get("alpha", 0.03))
        if alpha > 0.08:
            alpha = 0.08

        out_file = params.get("output_csv")
        if not out_file:
            out_file = self.project_root / "submissions" / f"probe_{base_path.parent.name}_{probe_path.parent.name}_a{int(alpha*100)}.csv"

        report = MicroDoseProbeGenerator.generate_probe(
            base_csv=base_path,
            probe_csv=probe_path,
            output_csv=out_file,
            alpha=alpha,
            space=params.get("space", "logit"),
            id_column=self.task_spec.id_column,
        )

        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Generated micro-dose probe (α={alpha:.2f}, space='{report['space']}'): Spearman r={report['spearman_correlation_vs_base']:.4f}, max_diff={report['max_absolute_diff']:.4f}.",
            artifact_path=report["output_path"],
            data=report,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_prepare_submission(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        pool = self.comp_state.compute_candidate_pool(journal)
        best_cand = pool.get("overall_best") or {}
        default_exp = (
            best_cand.get("id")
            or self.comp_state.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        exp_id = params.get("exp_id", default_exp)
        cand_csv = params.get("candidate_csv")
        if not cand_csv:
            default_cand = self.artifacts_dir / exp_id / "test_predictions.csv"
            if default_cand.is_file():
                cand_csv = str(default_cand)
            else:
                raise ValueError(
                    f"Action 'prepare_submission' requires 'candidate_csv' or existing 'test_predictions.csv' in {self.artifacts_dir / exp_id}."
                )
        cand_path = Path(cand_csv)
        if not cand_path.is_absolute():
            cand_path = self.project_root / cand_path

        from harness.artifacts import validate_test_prediction

        validate_test_prediction(
            cand_path,
            task_spec=self.task_spec,
            plugin=self.task_plugin,
        )
        # Anti-bypass submission guards: dedup, MMES, significance. The
        # agent drives submission through this bridge action; the standalone
        # audit gate CLI is not on this path, so enforce the core checks here.
        guard_evidence = self._guard_submission(cand_path, exp_id, sub_mgr)
        capabilities = self.task_plugin.capabilities()
        if not capabilities.get("submission_builder") or not capabilities.get(
            "submission_validator"
        ):
            raise ValueError("Task plugin must implement submission hooks")
        sub_spec = getattr(self.task_spec, "submission_spec", None) or self.comp_state.spec.get("submission_spec", {})
        file_name = sub_spec.get("file_name", "submission.csv") if isinstance(sub_spec, dict) else "submission.csv"
        entry = sub_mgr.register_plugin_submission(
            exp_id=exp_id,
            candidate_csv=cand_path,
            plugin=self.task_plugin,
            plugin_context=self.plugin_context,
            file_name=file_name,
            notes=params.get("notes", ""),
        )
        # Packaging is not an external submission; confirmed sends consume quota.
        plugin_label = self.project.plugin_id or "default"
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Built and validated task submission '{entry.id}' with plugin '{plugin_label}'.",
            artifact_path=entry.archive_path,
            data={**asdict(entry), "submission_guards": guard_evidence},
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_record_external_score(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        sub_id = params.get("sub_id")
        if not sub_id:
            raise ValueError(
                "Parameter 'sub_id' is required for 'record_external_score'."
            )
        if params.get("provider_ref"):
            from harness.kaggle_compat import KaggleClient
            entry = sub_mgr.record_official_observation(sub_id, client=KaggleClient(),
                provider_ref=params["provider_ref"], partition=params.get("split_type", "public"))
        else:
            entry = sub_mgr.record_official_score(
                sub_id=sub_id,
                primary_score=(params["primary_score"] if "primary_score" in params else params.get("score")),
                scores=params.get("scores") or params.get("custom_scores"),
                split_type=params.get("split_type", "public"),
                notes=params.get("notes", ""),
            )
        blockers = (
            [
                "CV vs LB Divergence detected (significant score delta between LB and Local CV)!"
            ]
            if entry.divergence_warning
            else []
        )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Recorded external score for '{sub_id}' (source={entry.official_score_source}): {entry.official_scores} (Divergence={entry.divergence_warning}).",
            new_blockers=blockers,
            data=asdict(entry),
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_freeze_candidate(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        s1 = params.get("slot_1_id")
        s2 = params.get("slot_2_id")
        if not s1:
            candidates = self.comp_state.state["candidate_pool"].get(
                "synthesized_candidates", []
            )
            if candidates:
                s1 = candidates[0].get("id")
                if len(candidates) > 1 and not s2:
                    s2 = candidates[1].get("id")
            else:
                raise ValueError(
                    "Parameter 'slot_1_id' is required for 'freeze_candidate' when candidate pool is empty."
                )
        lock = self.comp_state.freeze_final_candidates(
            slot_1_id=s1,
            slot_2_id=s2,
            notes=params.get(
                "notes", f"Frozen via AgentBridge session {session.session_id}"
            ),
        )
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Frozen final portfolio ({s1}, {s2}) into FROZEN_FINAL_LOCK.json.",
            artifact_path="submissions/FROZEN_FINAL_LOCK.json",
            data=lock,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_emergency_unfreeze(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        reason = params.get("reason")
        if not reason:
            raise ValueError("Unfreeze requires an explicit reason")
        res = self.comp_state.emergency_unfreeze(reason=reason)
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Emergency unfreeze executed: {reason}",
            data=res,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_finalize_competition(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        res = self.comp_state.finalize_competition(params, journal, sub_mgr)
        session.status = "COMPLETED"
        session.stop_reason = "COMPETITION_FINALIZED"
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Generated Competition Closure & Reproducibility Report at '{res['report_path']}'.",
            artifact_path=res["report_path"],
            data=res,
            elapsed_seconds=round(time.time() - t0, 3),
        )

    def _handle_get_next_experiment_id(
        self,
        call,
        session,
        journal,
        sub_mgr,
        act,
        params,
        t0,
    ) -> ActionObservation:
        next_num = self.get_next_experiment_num()
        branch = params.get("branch", "feat")
        descriptor = params.get("descriptor", "iteration")
        suggested_id = f"exp{next_num:03d}_{branch}_{descriptor}"
        return ActionObservation(
            step_index=call.step_index,
            action=act,
            status="SUCCESS",
            summary=f"Next available experiment number is {next_num} (suggested ID: '{suggested_id}').",
            data={
                "next_num": next_num,
                "next_id_prefix": f"exp{next_num:03d}",
                "suggested_exp_id": suggested_id,
            },
            elapsed_seconds=round(time.time() - t0, 3),
        )
