#!/usr/bin/env python3
"""Task-Agnostic ML/DL Experimentation Dashboard & Antigravity Priming Prompt Server.

Zero-dependency standard library Python HTTP server providing:
1. REST APIs for the AIDE solution tree, experiments, diagnostics, and UCB recommendations.
2. Dynamic Antigravity Priming Prompt generation to eliminate agent amnesia and hallucinated certainty.
3. Clean modern web SPA frontend.
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import ipaddress
import json
import re
import socketserver
import subprocess
import sys
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Ensure harness directory and project root in sys.path
HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from tools.dashboard_config import (
    ARTIFACTS_DIR,
    DEFAULT_STRATEGY_CARDS,
    HARNESS_PROJECT,
    PROJECT_ROOT,
    TASK_PRESETS,
    generate_prompt_by_template,
    get_next_experiment_id,
    list_available_tasks,
    load_task_config,
    save_task_config,
    switch_task,  # noqa: F401  (facade: re-exported for external importers)
)

from harness.agent_bridge import AgentBridge
from harness.cloud_inspector import CloudPreflightInspector
from harness.cloud_job_runner import CloudJobManager
from harness.competition_state import CompetitionStateMachine
from harness.environment_detector import EnvironmentDetector
from harness.llm_engine import LLMEngine
from harness.skill_injector import inject_skill_file
from harness.submission_manager import SubmissionManager
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import (
    generate_modal_script,
    push_dataset_to_modal,
)
from harness.tree_journal import TreeJournal

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler serving REST API and dashboard frontend."""

    def __init__(self, *args, **kwargs):
        candidates = [
            PROJECT_ROOT / "dashboard" / "dist",
            HARNESS_DIR / "dashboard" / "dist",
            HARNESS_DIR / "tools" / "web",
            PROJECT_ROOT / "tools" / "web",
            HARNESS_DIR / "dashboard",
            PROJECT_ROOT / "dashboard",
        ]
        self.web_dir = next((p for p in candidates if p.exists() and (p / "index.html").exists()), candidates[0])
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    @staticmethod
    def _is_loopback_name(hostname: Optional[str]) -> bool:
        if not hostname:
            return False
        hostname = hostname.rstrip(".").casefold()
        if hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _reject_nonlocal_request(self) -> bool:
        """Reject DNS rebinding hosts and cross-origin browser requests before dispatch."""
        host_header = self.headers.get("Host", "")
        try:
            host_parts = urllib.parse.urlsplit("//" + host_header)
            local_host = self._is_loopback_name(host_parts.hostname)
        except ValueError:
            local_host = False
        origin = self.headers.get("Origin")
        valid_origin = True
        if origin:
            try:
                origin_parts = urllib.parse.urlsplit(origin)
                valid_origin = (
                    origin_parts.scheme in {"http", "https"}
                    and self._is_loopback_name(origin_parts.hostname)
                    and origin_parts.netloc.casefold() == host_header.casefold()
                )
            except ValueError:
                valid_origin = False
        if not local_host or not valid_origin:
            self.send_error(403, "Dashboard API is restricted to the local same-origin workspace.")
            return True
        return False

    def do_GET(self) -> None:
        if self._reject_nonlocal_request():
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/tree":
            self.handle_api_tree()
        elif path == "/api/experiments":
            self.handle_api_experiments()
        elif path.startswith("/api/node/"):
            node_id = path.replace("/api/node/", "").strip()
            self.handle_api_node(node_id)
        elif path == "/api/recommend":
            self.handle_api_recommend(parsed_url.query)
        elif path == "/api/priming_prompt":
            self.handle_api_priming_prompt(parsed_url.query)
        elif path == "/api/prompt_templates":
            self.handle_api_prompt_templates()
        elif path == "/api/status":
            self.handle_api_status()
        elif path == "/api/submissions":
            self.handle_api_submissions()
        elif path == "/api/submissions/download":
            params = urllib.parse.parse_qs(parsed_url.query)
            sub_id = params.get("id", [""])[0]
            mgr = SubmissionManager()
            sub = next((s for s in mgr.get_all_submissions() if s.id == sub_id), None)
            rel_file = (
                getattr(sub, "archive_path", None)
                or getattr(sub, "zip_path", None)
                or getattr(sub, "submission_file", None)
                if sub
                else None
            )
            if not rel_file or not (PROJECT_ROOT / rel_file).exists():
                self.send_json({"error": "Submission package file not found on disk."}, status=404)
                return
            fpath = PROJECT_ROOT / rel_file
            data_bytes = fpath.read_bytes()
            self.send_response(200)
            ctype = "application/zip" if fpath.suffix == ".zip" else "text/csv"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{fpath.name}"')
            self.send_header("Content-Length", str(len(data_bytes)))
            self.end_headers()
            self.wfile.write(data_bytes)
        elif path == "/api/task_info":
            self.handle_api_task_info()
        elif path == "/api/tasks":
            self.send_json({"tasks": list_available_tasks()})
        elif path == "/api/task_presets":
            self.send_json({"presets": TASK_PRESETS})
        elif path == "/api/next_exp_id":
            self.handle_api_next_exp_id(parsed_url.query)
        elif path == "/api/environment/status":
            self.send_json(EnvironmentDetector.get_full_status())
        elif path == "/api/configs":
            self.handle_api_configs()
        elif path == "/api/strategy_cards":
            self.handle_api_strategy_cards()
        elif path == "/api/config":
            self.handle_api_get_config(parsed_url.query)
        elif path == "/api/cloud/preflight":
            self.handle_api_cloud_preflight(parsed_url.query)
        elif path == "/api/cloud/job":
            self.handle_api_cloud_get_job(parsed_url.query)
        elif path == "/api/cloud/jobs":
            self.send_json({"jobs": CloudJobManager().list_jobs()})
        elif path == "/api/cloud/logs/stream":
            self.handle_api_cloud_logs_stream(parsed_url.query)
        elif path == "/api/sync":
            self.handle_api_sync()
        elif path == "/api/llm/providers":
            self.handle_api_llm_providers()
        elif path == "/api/llm/sample_data":
            self.handle_api_llm_sample_data()
        elif path == "/api/llm/hf_models":
            self.handle_api_llm_hf_models(parsed_url.query)
        elif path == "/api/llm/unsloth_models":
            self.handle_api_llm_unsloth_models(parsed_url.query)
        elif path == "/api/llm/provider_models":
            self.handle_api_llm_provider_models(parsed_url.query)
        elif path == "/api/competition-state":
            comp = CompetitionStateMachine(PROJECT_ROOT)
            snap = comp.evaluate_journey_snapshot(TreeJournal(auto_sync=True), SubmissionManager(), load_task_config())
            self.send_json(snap)
        elif path == "/api/agent-bridge/context":
            params = urllib.parse.parse_qs(parsed_url.query)
            sid = params.get("session_id", [None])[0]
            bridge = AgentBridge(PROJECT_ROOT)
            pkt = bridge.get_context_packet(session_id=sid)
            pkt["markdown"] = bridge.render_context_markdown(session_id=sid)
            self.send_json(pkt)
        elif path == "/api/agent-bridge/sessions":
            bridge = AgentBridge(PROJECT_ROOT)
            self.send_json({"sessions": bridge.list_sessions()})
        elif path == "/api/phases":
            from harness.phase_workflow import PhaseWorkflow
            self.send_json(PhaseWorkflow(PROJECT_ROOT, HARNESS_PROJECT.task_config_path).phase_status())
        elif path == "/api/phases/render":
            from harness.phase_workflow import PhaseWorkflow
            params = urllib.parse.parse_qs(parsed_url.query)
            phase_id = params.get("phase", [None])[0]
            try:
                self.send_json({"phase_id": phase_id, "prompt": PhaseWorkflow(PROJECT_ROOT, HARNESS_PROJECT.task_config_path).render(phase_id)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/eda/latest":
            eda_dir = ARTIFACTS_DIR / "eda"
            reports = sorted(eda_dir.glob("eda_*.json"), reverse=True) if eda_dir.exists() else []
            if reports:
                try:
                    self.send_json(json.loads(reports[0].read_text(encoding="utf-8")))
                except Exception as exc:
                    self.send_json({"error": f"Could not read EDA report: {exc}"}, status=500)
            else:
                self.send_json({"report": None})
        elif path == "/api/cloud/status":
            from harness.cloud_policy import UploadConsentStore, provider_preflight
            self.send_json({
                "providers": {name: provider_preflight(HARNESS_PROJECT, name) for name in ("kaggle", "modal")},
                "upload_consent": UploadConsentStore(HARNESS_PROJECT).status(),
            })
        else:
            # Fallback to serving static frontend
            super().do_GET()

    def do_OPTIONS(self) -> None:
        if self._reject_nonlocal_request():
            return
        self.send_response(200)
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self._reject_nonlocal_request():
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/agent-bridge/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                bridge = AgentBridge(PROJECT_ROOT)
                sub_action = path.replace("/api/agent-bridge/", "").strip()

                if sub_action == "start-session":
                    sess = bridge.start_session(
                        goal=data.get("goal", "Iterative competition optimization via Antigravity IDE"),
                        budget=data.get("budget"),
                        session_id=data.get("session_id"),
                    )
                    self.send_json({"success": True, "session": asdict(sess)})
                elif sub_action == "act":
                    sess, obs = bridge.execute_action(
                        action=data.get("action", "inspect_diagnostics"),
                        params=data.get("params", {}),
                        rationale=data.get("rationale", ""),
                        session_id=data.get("session_id"),
                        goal=data.get("goal"),
                        budget=data.get("budget"),
                        user_approved=bool(data.get("user_approved", False)),
                    )
                    self.send_json({
                        "success": obs.status == "SUCCESS",
                        "session": asdict(sess),
                        "observation": asdict(obs),
                    })
                elif sub_action == "approve":
                    sid = data.get("session_id")
                    if not sid:
                        sessions = bridge.list_sessions(limit=1)
                        sid = sessions[0]["session_id"] if sessions else None
                    if not sid:
                        self.send_json({"error": "No active session to approve."}, status=400)
                        return
                    sess, obs = bridge.approve_pending_action(sid)
                    self.send_json({
                        "success": True,
                        "session": asdict(sess),
                        "observation": asdict(obs) if obs else None,
                    })
                else:
                    self.send_json({"error": f"Unknown agent-bridge action: {sub_action}"}, status=404)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
            return
        elif path.startswith("/api/competition-state/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                comp = CompetitionStateMachine(PROJECT_ROOT)
                action = path.replace("/api/competition-state/", "").strip()

                aliases={"synthesize-candidate":"synthesize_candidate","freeze":"freeze_candidate","unfreeze":"emergency_unfreeze","finalize":"finalize_competition","data-audit":"inspect_data_readiness"}
                if action in aliases:
                    params=dict(data)
                    params["slot_1_id"]=data.get("slot_1_safe_cv",data.get("slot_1_id"))
                    params["slot_2_id"]=data.get("slot_2_aggressive_lb",data.get("slot_2_id"))
                    sess,obs=AgentBridge(PROJECT_ROOT).execute_action(aliases[action],params=params,session_id=data.get("session_id"),user_approved=bool(data.get("user_approved",False)))
                    self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs),"session":asdict(sess)})
                    return
                if action == "gate":
                    for key in data.get("gates",{}):
                        if key not in {"spec_confirmed","data_risks_confirmed"}:
                            self.send_json({"error": f"Gate {key} requires its verified action"}, status=409)
                            return
                    bridge=AgentBridge(PROJECT_ROOT)
                    for key,value in data.get("gates",{}).items():
                        if not value:
                            raise ValueError("Review evidence cannot be cleared through the legacy endpoint")
                        _,obs=bridge.execute_action("confirm_task_spec" if key=="spec_confirmed" else "confirm_data_risks",user_approved=bool(data.get("user_approved",False)))
                        if obs.status!="SUCCESS":
                            self.send_json({"success": False, "observation": asdict(obs)}, status=409)
                            return
                    settings={k:v for k,v in data.items() if k!="gates"}
                    _, obs = bridge.execute_action("update_workspace_settings", params=settings)
                    if obs.status != "SUCCESS":
                        raise ValueError(obs.summary)
                    comp=bridge.comp_state
                elif action == "data-audit":
                    comp.run_data_readiness_audit(data.get("config"))
                elif action == "agent-context":
                    if "config" in data:
                        comp.state["agent_context_config"].update(data["config"])
                        comp.save()
                elif action == "synthesize-candidate":
                    comp.synthesize_candidate(
                        name=data.get("name", "specialist_router_v1"),
                        mode=data.get("mode", "specialist_router"),
                        source_map=data.get("source_map", {}),
                    )
                elif action == "freeze":
                    comp.freeze_final_candidates(
                        slot_1_id=data.get("slot_1_safe_cv", "baseline"),
                        slot_2_id=data.get("slot_2_aggressive_lb"),
                        notes=data.get("notes", ""),
                    )
                elif action == "unfreeze":
                    comp.emergency_unfreeze(reason=data.get("reason", "Manual unfreeze via UI"))
                elif action == "finalize":
                    comp.finalize_competition(data, TreeJournal(auto_sync=True), SubmissionManager())
                elif action == "agent-run":
                    self.send_json({"error": "The dashboard does not run an internal agent. Use the attached IDE agent with `ai-harness phase render` and the Agent Bridge."}, status=409)
                    return
                elif action == "agent-approve":
                    self.send_json({"error": "Use the IDE agent's Agent Bridge session approval endpoint."}, status=409)
                    return

                snap = comp.evaluate_journey_snapshot(TreeJournal(auto_sync=True), SubmissionManager(), load_task_config())
                self.send_json({"success": True, "state": snap})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
            return
        elif path == "/api/submissions/record-score":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                sub_id = data.get("sub_id")
                official_scores = data.get("official_scores", {})
                notes = data.get("notes", "")

                mgr = SubmissionManager()
                if sub_id not in mgr.submissions:
                    self.send_json({"error": f"Submission '{sub_id}' not found."}, status=404)
                    return

                split_type = str(official_scores.get("split_type", "public"))
                primary_metric = str(load_task_config().get("primary_metric") or "score")
                primary_score = official_scores.get(
                    "primary_metric",
                    official_scores.get("score", official_scores.get(primary_metric)),
                )
                custom_scores = {
                    key: value
                    for key, value in official_scores.items()
                    if key not in ("split_type", "primary_metric", "score", primary_metric)
                }

                sess,obs=AgentBridge(PROJECT_ROOT).execute_action("record_external_score",params={"sub_id":sub_id,"primary_score":primary_score,"split_type":split_type,"notes":notes,"custom_scores":custom_scores},user_approved=bool(data.get("user_approved",False)),session_id=data.get("session_id"))
                self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs),"session":asdict(sess)})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/task_info":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                active_task = load_task_config()
                for identity_key in ("id", "name", "task_name", "plugin_id", "plugin"):
                    if identity_key in data and data.get(identity_key) != active_task.get(identity_key):
                        self.send_json({"error": "Each task needs its own workspace. Attach another workspace to change task identity or plugin."}, status=409)
                        return
                save_task_config(data)
                # Auto-inject updated agent skill for this task
                try:
                    inject_skill_file(task_config=data)
                except Exception as skill_err:
                    print(f"⚠️ Warning: Auto-injecting skill failed: {skill_err}")
                self.send_json({"success": True, "task": load_task_config(), "skill_injected": True})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/tasks/switch":
            self.send_json({"error": "Each task needs its own workspace. Run `ai-harness attach` from the task workspace you want to use."}, status=409)
        elif path == "/api/tasks/create":
            self.send_json({"error": "Create or bootstrap the task in its own workspace, then run `ai-harness attach`."}, status=409)
        elif path == "/api/environment/modal/activate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_modal_activate(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/modal/token":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_modal_token(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/modal/launch_auth":
            try:
                res = EnvironmentDetector.launch_modal_auth()
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/kaggle/launch_auth":
            try:
                res = EnvironmentDetector.launch_kaggle_auth()
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/kaggle/profile":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_kaggle_profile(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/run":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_run(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/consent":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                provider = data.get("provider")
                if provider not in {"kaggle", "modal"} or data.get("confirm_upload") is not True:
                    self.send_json({"error": "Choose Kaggle or Modal and explicitly confirm this task dataset upload."}, status=400)
                    return
                from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
                record = UploadConsentStore(HARNESS_PROJECT).grant(provider, dataset_fingerprint(HARNESS_PROJECT))
                self.send_json({"success": True, "consent": record})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/phases/record":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                _,obs=AgentBridge(PROJECT_ROOT).execute_action("record_phase_output",params=data)
                self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs)},status=200 if obs.status=="SUCCESS" else 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/eda/run":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                _,obs=AgentBridge(PROJECT_ROOT).execute_action("run_eda",params=data)
                self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs)},status=200 if obs.status=="SUCCESS" else 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/cloud/stop":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_stop(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/audit":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_audit(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/skill/inject":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                task_cfg = data if data else load_task_config()
                p = inject_skill_file(task_config=task_cfg)
                self.send_json({"success": True, "skill_path": str(p.relative_to(PROJECT_ROOT))})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/submissions/register":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                exp_id = data.get("exp_id")
                if not exp_id:
                    self.send_json({"error": "Field 'exp_id' is required."}, status=400)
                    return

                sess,obs=AgentBridge(PROJECT_ROOT).execute_action("prepare_submission",params=data,session_id=data.get("session_id"),user_approved=bool(data.get("user_approved",False)))
                self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs),"session":asdict(sess),"error":obs.summary if obs.status!="SUCCESS" else None},status=200 if obs.status=="SUCCESS" else 400)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/generate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_generate(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/push-dataset":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_push_dataset(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/kaggle-status":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_kaggle_status(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/pull-output":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_pull_output(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/config":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_save_config(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/config/create":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_create_config(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/test_prompt":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_test_prompt(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/generate_peft_script":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_generate_peft(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/generate_unsloth_script":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_generate_unsloth(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/llm/key":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_save_key(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/llm/verify":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_verify_key(data)
            except Exception as e:
                self.send_json({"error": str(e), "success": False, "valid": False}, status=400)
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_api_tree(self) -> None:
        journal = TreeJournal()
        graph_data = journal.get_graph_data()
        self.send_json(graph_data)

    def handle_api_experiments(self) -> None:
        journal = TreeJournal()
        experiments = []
        best = journal.get_best_node()
        best_id = best.id if best else None

        for n in journal.nodes.values():
            experiments.append({
                "id": n.id,
                "branch": n.branch,
                "parent_id": n.parent_id or "ROOT",
                "approach_type": getattr(n, "approach_type", "ml"),
                "model_family": getattr(n, "model_family", "linear"),
                "score": round(n.get_score(), 4),"verified":n.verified,"eligible":n.eligible,"direction":n.direction,"primary_metric":n.primary_metric_name,
                "param_count": n.param_count,
                "param_size": getattr(n, "param_size", None),
                "llm_spec": getattr(n, "llm_spec", None),
                "status": n.status,
                "is_best": (n.id == best_id),
                "is_buggy": n.is_buggy,
                "hypothesis": n.atomic_hypothesis,
                "intervention": n.intervention,
                "timestamp": n.timestamp,
            })
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        self.send_json({"total": len(experiments), "experiments": experiments})

    def handle_api_node(self, node_id: str) -> None:
        journal = TreeJournal()
        node = journal.get_node(node_id)
        if not node:
            self.send_json({"error": f"Node '{node_id}' not found."}, status=404)
            return

        # Attempt to load grounding receipt if present
        grounding_data = None
        exp_dir = ARTIFACTS_DIR / node_id
        receipt_file = exp_dir / "GROUNDING_RECEIPT.json"
        if receipt_file.exists():
            try:
                with open(receipt_file, "r", encoding="utf-8") as f:
                    grounding_data = json.load(f)
            except Exception:
                pass

        # Attempt to load target experiment config
        config_data = None
        cand_files = [
            PROJECT_ROOT / "configs" / f"{node.id}.yaml",
            ARTIFACTS_DIR / node.id / "run_config.yaml",
        ]
        configs_dir = PROJECT_ROOT / "configs"
        if configs_dir.exists():
            prefix = node.id.split("_")[0]
            for cfg in configs_dir.glob(f"{prefix}*.yaml"):
                if cfg not in cand_files:
                    cand_files.append(cfg)

        for cand in cand_files:
            if cand.exists():
                try:
                    raw_txt = cand.read_text(encoding="utf-8")
                    config_data = {
                        "path": str(cand.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "content": raw_txt,
                    }
                    break
                except Exception:
                    pass

        # Determine llm_spec and param_size
        llm_spec = getattr(node, "llm_spec", None)
        param_size = getattr(node, "param_size", None)

        if config_data and config_data.get("content"):
            try:
                parsed_cfg = yaml.safe_load(config_data["content"]) or {}
                if not llm_spec and parsed_cfg.get("llm"):
                    llm_spec = parsed_cfg.get("llm")
                if not param_size and llm_spec and isinstance(llm_spec, dict):
                    param_size = llm_spec.get("param_size")
                if not param_size and parsed_cfg.get("model", {}).get("param_size"):
                    param_size = parsed_cfg["model"]["param_size"]
            except Exception:
                pass

        # If node represents an LLM, ensure a sensible spec structure
        node_approach = getattr(node, "approach_type", "ml")
        lower_id = node.id.lower()
        if (node_approach == "llm" or any(k in lower_id for k in ["llm", "qwen", "deepseek", "llama", "unsloth", "cot", "qlora"])) and not llm_spec:
            model_name = node.metrics.get("model") or node.id
            if "deepseek" in lower_id:
                llm_spec = {
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    "param_size": "14B",
                    "quantization": "4-bit AWQ / NF4",
                    "paradigm": "zero_shot_cot",
                    "context_length": 8192,
                    "provider": "huggingface",
                    "use_cot": True,
                }
                param_size = param_size or "14B"
            elif "qwen" in lower_id or "unsloth" in lower_id:
                llm_spec = {
                    "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                    "param_size": "7B",
                    "quantization": "4-bit NormalFloat (NF4)",
                    "paradigm": "unsloth_qlora",
                    "context_length": 8192,
                    "provider": "unsloth",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    "use_cot": True,
                }
                param_size = param_size or "7B"
            else:
                llm_spec = {
                    "model_name": model_name,
                    "param_size": param_size or "7B",
                    "quantization": "4-bit NF4",
                    "paradigm": "unsloth_qlora",
                    "context_length": 4096,
                    "provider": "unsloth",
                }

        details = {
            "id": node.id,
            "parent_id": node.parent_id,
            "branch": node.branch,
            "approach_type": getattr(node, "approach_type", "ml"),
            "model_family": getattr(node, "model_family", "linear"),
            "score": round(node.get_score(), 4),
            "param_count": node.param_count,
            "param_size": param_size,
            "llm_spec": llm_spec,
            "status": node.status,
            "is_buggy": node.is_buggy,
            "atomic_hypothesis": node.atomic_hypothesis,
            "intervention": node.intervention,
            "notes": node.notes,
            "timestamp": node.timestamp,
            "metrics": node.metrics,
            "diagnostics_summary": node.diagnostics_summary,
            "grounding_receipt": grounding_data,
            "config": config_data,
        }
        self.send_json(details)

    def handle_api_recommend(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        max_params = None
        if "max_params" in params:
            try:
                max_params = int(params["max_params"][0])
            except ValueError:
                pass

        journal = TreeJournal()
        rec = journal.recommend_next_branch(max_params=max_params)
        self.send_json(rec)

    def handle_api_prompt_templates(self) -> None:
        self.send_json({
            "templates": [
                {
                    "id": "antigravity_attach",
                    "ide": "antigravity",
                    "intent": "attach",
                    "name": "Antigravity — Setup & Attach",
                    "badge": "Setup",
                    "description": "Initial session prompt: Ingest task specification, verify split integrity, and establish foundational baseline.",
                },
                {
                    "id": "antigravity_iterate",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "Antigravity — Continue Iteration",
                    "badge": "New Chat",
                    "description": "Context injection for active iterations: Reigning champion, UCB tree recommendation, diagnosed bottlenecks, and 4-step Alignment Gate.",
                },
                {
                    "id": "codex_attach",
                    "ide": "codex",
                    "intent": "attach",
                    "name": "OpenAI Codex — Setup & Attach",
                    "badge": "CLI Setup",
                    "description": "Terminal instructions for Codex CLI: Inspect repository rules, verify canonical data splits, and execute baseline smoke test.",
                },
                {
                    "id": "codex_iterate",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "OpenAI Codex — Continue Iteration",
                    "badge": "CLI Iteration",
                    "description": "Autonomous ML engineering loop: Single atomic changes, task-configured validation, and Git diff reporting.",
                },
                # Backward-compatibility aliases
                {
                    "id": "antigravity",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "Antigravity Agent (Default)",
                    "badge": "Active",
                    "description": "Alias for Antigravity Continue Iteration prompt.",
                },
                {
                    "id": "codex",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "OpenAI Codex CLI (Default)",
                    "badge": "Active",
                    "description": "Alias for Codex Continue Iteration prompt.",
                },
                {
                    "id": "aide",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "AIDE ML Solution Tree (Legacy)",
                    "badge": "Legacy",
                    "description": "Legacy alias routed to Antigravity iteration prompt.",
                },
                {
                    "id": "minimal",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "Minimal Direct Patch (Legacy)",
                    "badge": "Legacy",
                    "description": "Legacy alias routed to Codex iteration prompt.",
                },
            ]
        })

    def handle_api_priming_prompt(self, query: str = "") -> None:
        params = urllib.parse.parse_qs(query)
        ide = params.get("ide", [None])[0]
        intent = params.get("intent", [None])[0]

        if ide and intent:
            template_id = f"{ide.strip().lower()}_{intent.strip().lower()}"
        else:
            template_id = params.get("template", ["antigravity"])[0]

        journal = TreeJournal()
        res = generate_prompt_by_template(journal, template_id=template_id)
        self.send_json(res)

    def handle_api_status(self) -> None:
        journal = TreeJournal()
        best = journal.get_best_node()
        self.send_json({
            "status": "online",
            "total_nodes": len(journal.nodes),
            "best_node": best.id if best else None,
            "best_score": round(best.get_score(), 4) if best else 0.0,
            "distinct_families": journal.count_distinct_model_families(),
        })

    def handle_api_submissions(self) -> None:
        mgr = SubmissionManager()
        data = [asdict(s) for s in mgr.submissions.values()]
        data.sort(key=lambda x: x.get("registered_at", 0), reverse=True)
        self.send_json({"total": len(data), "submissions": data})

    def handle_api_task_info(self) -> None:
        task = load_task_config()
        artifact_path = ARTIFACTS_DIR.relative_to(PROJECT_ROOT).as_posix()
        plugin_id = HARNESS_PROJECT.plugin_id or "default"
        plugin_info: Dict[str, Any] = {"id": plugin_id, "version": None, "capabilities": {}}
        try:
            from harness.plugins import TaskPluginContext, load_task_plugin
            from harness.task_spec import load_active_task_spec

            task_spec = load_active_task_spec(HARNESS_PROJECT.task_config_path)
            context = TaskPluginContext(
                project_root=PROJECT_ROOT,
                task_config_path=HARNESS_PROJECT.task_config_path,
                task_spec=task_spec,
                plugin_id=plugin_id,
            )
            plugin = load_task_plugin(HARNESS_PROJECT.plugin_id, context)
            plugin_info.update(version=plugin.version, capabilities=plugin.capabilities())
        except Exception as exc:
            plugin_info["error"] = str(exc)
        self.send_json({
            "platform": task.get("platform", "codabench"),
            "task_name": task.get("task_name") or task.get("name", "General ML/DL Benchmark"),
            "name": task.get("name", "General ML/DL Benchmark"),
            "task_type": task.get("task_type") or task.get("type", "Predictive Modeling"),
            "type": task.get("type", "Predictive Modeling"),
            "modality": task.get("modality", "generic"),
            "primary_metric": task.get("primary_metric", "Score"),
            "direction": task.get("direction", "maximize"),
            "primary_metric_baseline": task.get("primary_metric_baseline"),
            "primary_metric_target": task.get("primary_metric_target"),
            "inputs": task.get("inputs", []),
            "subtasks": task.get("subtasks", task.get("metrics", [])),
            "metrics": task.get("metrics", []),
            "languages": task.get("languages", []),
            "description": task.get("description", ""),
            "data_config": task.get("data_config", {}),
            "competition_rules": task.get("competition_rules", {}),
            "submission_spec": task.get("submission_spec", {}),
            "artifacts_dir": artifact_path,
            "task_plugin": plugin_info,
            "budget_capabilities":{"local":{"wall_deadline":True,"gpu_allocation":False},"modal":{"hard_budget":False},"kaggle":{"hard_budget":False}},
            "runner_utilities":{"calibration":False,"threshold_optimization":False},
        })

    def handle_api_configs(self) -> None:
        configs_dir = PROJECT_ROOT / "configs"
        cfg_files = []
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                if f.name != "task.yaml":
                    cfg_files.append(f"configs/{f.name}")
        self.send_json({"configs": cfg_files})

    def handle_api_strategy_cards(self) -> None:
        cards_file = PROJECT_ROOT / "configs" / "strategy_cards.yaml"
        if cards_file.exists():
            try:
                raw_cards = yaml.safe_load(cards_file.read_text(encoding="utf-8")) or {}
                if isinstance(raw_cards, list):
                    return self.send_json({"strategies": raw_cards})
                elif isinstance(raw_cards, dict) and "strategies" in raw_cards:
                    return self.send_json({"strategies": raw_cards["strategies"]})
            except Exception:
                pass
        self.send_json({"strategies": [c for c in DEFAULT_STRATEGY_CARDS if c.get("model_name")!="threshold_optimizer"],"utility_capabilities":{"calibration":"custom_integration","thresholds":"custom_integration"}})

    def handle_api_get_config(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        cfg_path = params.get("path", [""])[0].strip()
        exp_id = params.get("exp_id", [""])[0].strip()

        target_file = None
        if cfg_path:
            norm_path = (PROJECT_ROOT / cfg_path).resolve()
            if (PROJECT_ROOT in norm_path.parents or norm_path == PROJECT_ROOT) and norm_path.exists():
                target_file = norm_path
        elif exp_id:
            for cand in [
                PROJECT_ROOT / "configs" / f"{exp_id}.yaml",
                ARTIFACTS_DIR / exp_id / "run_config.yaml",
            ]:
                if cand.exists():
                    target_file = cand
                    break
            if not target_file:
                prefix = exp_id.split("_")[0]
                configs_dir = PROJECT_ROOT / "configs"
                if configs_dir.exists():
                    for cand in configs_dir.glob(f"{prefix}*.yaml"):
                        target_file = cand
                        break

        if not target_file or not target_file.exists():
            self.send_json({"error": f"Config not found for path='{cfg_path}' exp_id='{exp_id}'"}, status=404)
            return

        try:
            content = target_file.read_text(encoding="utf-8")
            parsed = yaml.safe_load(content) or {}
            self.send_json({
                "path": str(target_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "content": content,
                "parsed": parsed,
            })
        except Exception as e:
            self.send_json({"error": f"Failed to read/parse config: {e}"}, status=400)

    def handle_api_save_config(self, data: Dict[str, Any]) -> None:
        raw_path = data.get("path", "").strip()
        content = data.get("content", "")

        if not raw_path:
            self.send_json({"error": "Field 'path' is required."}, status=400)
            return

        if not (raw_path.endswith(".yaml") or raw_path.endswith(".yml")):
            self.send_json({"error": "Config path must end with .yaml or .yml."}, status=400)
            return

        dest = (PROJECT_ROOT / raw_path).resolve()
        if PROJECT_ROOT not in dest.parents:
            self.send_json({"error": "Path traversal prohibited."}, status=403)
            return

        try:
            parsed = yaml.safe_load(content)
            if not isinstance(parsed, dict):
                raise ValueError("YAML content must be an object/mapping.")
        except Exception as yaml_err:
            self.send_json({"error": f"Invalid YAML syntax: {yaml_err}"}, status=400)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        _, obs = AgentBridge(PROJECT_ROOT).execute_action("write_experiment_config", params={"path": raw_path, "content": content})
        if obs.status != "SUCCESS":
            self.send_json({"error": obs.summary}, status=400)
            return

        self.send_json({
            "success": True,
            "path": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "parsed": parsed,
        })

    def handle_api_create_config(self, data: Dict[str, Any]) -> None:
        exp_id = data.get("exp_id", "").strip()
        parent_id = data.get("parent_id", "exp000_template").strip()
        branch = data.get("branch", "feat").strip()
        hypothesis = data.get("hypothesis", "").strip()
        intervention = data.get("intervention", "").strip()
        problem = data.get("problem_addressed", "").strip()
        approach_type = data.get("approach_type", "dl" if branch in ("model", "loss") else "ml")
        model_family = data.get("model_family", "transformer" if approach_type == "dl" else "linear")

        if not exp_id:
            desc = data.get("descriptor") or data.get("branch") or "target"
            next_info = get_next_experiment_id(branch, desc)
            exp_id = next_info["next_id"]

        if not exp_id.startswith("exp"):
            exp_id = f"exp_{exp_id}"

        filename = f"{exp_id}.yaml" if not exp_id.endswith(".yaml") else exp_id
        dest = PROJECT_ROOT / "configs" / filename

        task_cfg = load_task_config()
        new_cfg = {
            "experiment": {
                "id": exp_id.replace(".yaml", ""),
                "branch": branch,
                "parent_id": parent_id,
                "date": datetime.date.today().isoformat(),
                "problem_addressed": problem or f"Optimization targeting branch '{branch}'.",
                "hypothesis": hypothesis or f"Hypothesis for {exp_id} on {branch} branch.",
                "intervention": intervention or f"Controlled intervention for {exp_id}.",
            },
            "task": {
                "name": task_cfg.get("name", "Competition Task"),
                "type": task_cfg.get("type", "Standard Classification"),
                "primary_metric": task_cfg.get("primary_metric", "Score"),
                "direction": task_cfg.get("direction", "maximize"),
                "approach_type": approach_type,
                "model_family": model_family,
            },
            "grounding": {
                "source_type": data.get("source_type", "huggingface_hub" if approach_type == "dl" else ("api_provider" if approach_type == "llm" else "verified_baseline")),
                "source_id": data.get("source_id", "gemini-2.0-flash" if approach_type == "llm" else ("microsoft/mdeberta-v3-base" if approach_type == "dl" else "sklearn_tfidf_canonical")),
                "retrieval_summary": data.get("retrieval_summary", f"Model verified for {task_cfg.get('name', 'task')} requirements."),
                "verified_via": data.get("verified_via", "local_registry"),
            },
            "assumption_register": {
                "known_facts": [
                    f"Baseline established under parent {parent_id}.",
                ],
                "assumptions": [
                    hypothesis or f"Branch {branch} yields measurable metric improvement.",
                ],
                "uncertainty_level": data.get("uncertainty_level", "low"),
            },
            "model": {
                "name": data.get("model_name", "gemini-2.0-flash" if approach_type == "llm" else ("mdeberta_v3" if approach_type == "dl" else "logistic_regression")),
                "provider": data.get("provider", "gemini") if approach_type == "llm" else None,
                "params": data.get("model_params", {
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "use_cot": False,
                } if approach_type == "llm" else {
                    "seed": 42,
                    "learning_rate": 2e-5 if approach_type == "dl" else 1.0,
                    "batch_size": 16 if approach_type == "dl" else 64,
                    "epochs": 3 if approach_type == "dl" else 1,
                }),
            },
            "data": {
                "train_path": task_cfg.get("data_config", {}).get("train_path", "dataset/"),
                "test_path": task_cfg.get("data_config", {}).get("test_path", "dataset/test/"),
                "folds": [0, 1, 2, 3, 4],
            },
            "training": {
                "seed": 42,
                "folds": [0, 1, 2, 3, 4],
                "debug": False,
            },
        }

        if approach_type == "llm":
            m_params = data.get("model_params", {})
            param_size = data.get("param_size") or m_params.get("param_size", "7B")
            new_cfg["llm"] = {
                "model_name": data.get("model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
                "param_size": param_size,
                "quantization": m_params.get("quantization", "4-bit NormalFloat (NF4)"),
                "paradigm": m_params.get("paradigm", "unsloth_qlora"),
                "context_length": m_params.get("context_length", 8192),
                "provider": m_params.get("provider", "unsloth"),
                "lora_r": m_params.get("lora_r", 16),
                "lora_alpha": m_params.get("lora_alpha", 32),
                "lora_dropout": m_params.get("lora_dropout", 0.05),
                "target_modules": m_params.get("target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
                "use_cot": m_params.get("use_cot", True),
                "system_prompt": "You are an expert classification model specializing in nuance, semantic context, and robust reasoning.",
                "prompt_template": "Classify the following input:\n\n{text}",
            }

        yaml_content = yaml.dump(new_cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _, obs = AgentBridge(PROJECT_ROOT).execute_action("write_experiment_config", params={"path": str(dest.relative_to(PROJECT_ROOT)), "content": yaml_content})
        if obs.status != "SUCCESS":
            self.send_json({"error": obs.summary}, status=400)
            return

        self.send_json({
            "success": True,
            "path": f"configs/{dest.name}",
            "content": yaml_content,
            "parsed": new_cfg,
        })

    def handle_api_sync(self) -> None:
        """Discover completed experiment artifacts and persist the solution journal."""
        try:
            journal = TreeJournal(
                journal_path=ARTIFACTS_DIR / "journal.json",
                artifacts_dir=ARTIFACTS_DIR,
            )
            synced_count = journal.sync_from_artifacts(ARTIFACTS_DIR)
            self.send_json({
                "success": True,
                "synced_count": synced_count,
                "total_nodes": len(journal.nodes),
            })
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_api_cloud_preflight(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        backend = params.get("provider", params.get("backend", ["kaggle"]))[0]
        config_path = params.get("config", [None])[0]
        result: Dict[str, Any] = {}
        try:
            from harness.cloud_policy import provider_preflight
            result["provider"] = provider_preflight(HARNESS_PROJECT, backend)
            if config_path:
                full_path = Path(config_path).expanduser()
                if not full_path.is_absolute():
                    full_path = PROJECT_ROOT / full_path
                passed, errors, resolved_cfg = CloudPreflightInspector.run_full_preflight_check(
                    full_path, project_root=PROJECT_ROOT
                )
                result["experiment"] = {"passed": passed, "errors": errors, "resolved_config": resolved_cfg}
                result["errors"] = errors
                result["resolved_config"] = resolved_cfg
                result["passed"] = bool(passed and result["provider"].get("ready"))
            else:
                result["passed"] = bool(result["provider"].get("ready"))
            self.send_json(result)
        except Exception as exc:
            self.send_json({"passed": False, "error": str(exc)}, status=400)

    def handle_api_cloud_generate(self, data: Dict[str, Any]) -> None:
        backend = data.get("backend") or data.get("platform") or "modal"
        config_path = data.get("config_path")
        if not config_path:
            self.send_json({"error": "config_path is required."}, status=400)
            return
        gpu = data.get("gpu", "A10" if backend == "modal" else "T4")
        from harness.cloud_policy import (
            dataset_fingerprint,
            private_upload_resource_names,
        )
        fingerprint = dataset_fingerprint(HARNESS_PROJECT)
        username = KaggleKernelManager.get_authenticated_username()
        resources = private_upload_resource_names(
            HARNESS_PROJECT, fingerprint_data=fingerprint, kaggle_username=username
        )
        task_id = HARNESS_PROJECT.manifest.get("task_id") or resources["task_slug"]
        volume_name = resources["modal_volume_name"]
        task_doc = yaml.safe_load(HARNESS_PROJECT.task_config_path.read_text(encoding="utf-8")) or {}
        task_data = task_doc.get("task", task_doc)
        data_config = task_data.get("data_config") or task_data.get("data") or {}
        dataset_slug = data_config.get("kaggle_dataset_slug") or resources["kaggle_dataset_slug"]
        if backend == "kaggle" and not dataset_slug:
            self.send_json({"error": "Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry."}, status=409)
            return

        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file
        exp_doc = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        exp_id = exp_doc.get("experiment", {}).get("id", config_file.stem)
        job_id = f"preview_{exp_id}"
        try:
            from harness.remote_runtime import resolve_remote_runtime
            runtime = resolve_remote_runtime(HARNESS_PROJECT)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        if backend == "modal":
            stage_dir = PROJECT_ROOT / "tmp" / "modal" / job_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            script_path = stage_dir / f"app_{exp_id}.py"
            code_content = generate_modal_script(
                str(config_file.relative_to(PROJECT_ROOT)), gpu=gpu, exp_id=exp_id,
                volume_name=volume_name, project_root=PROJECT_ROOT,
                task_config_path=HARNESS_PROJECT.task_config_path, task_id=task_id,
                job_id=job_id, runtime_manifest=runtime,
                artifacts_rel=ARTIFACTS_DIR.relative_to(PROJECT_ROOT).as_posix(),
            )
            script_path.write_text(code_content, encoding="utf-8")

            self.send_json({
                "success": True,
                "backend": "modal",
                "script_path": str(script_path.relative_to(PROJECT_ROOT)),
                "code": code_content,
                "commands": {
                    "dataset_cmd": f"ai-harness cloud consent --provider modal --confirm-upload\n# then: python harness/cloud_runner.py --backend modal --project-root . --config {config_path}",
                    "push_cmd": f"python -m harness.cloud_runner --backend modal --project-root . --config {config_path} --gpu {gpu}",
                    "status_cmd": "modal app list",
                    "output_cmd": f"modal volume ls {volume_name}",
                }
            })
        else: # kaggle
            stage_dir = PROJECT_ROOT / "tmp" / "kaggle" / job_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            KaggleKernelManager.setup_kernel_package(
                stage_dir, str(config_file.relative_to(PROJECT_ROOT)), exp_id, dataset_slug,
                project_root=PROJECT_ROOT, task_config_path=HARNESS_PROJECT.task_config_path,
                task_id=task_id, job_id=job_id, runtime_manifest=runtime,
                enable_internet=bool((task_data.get("runtime") or {}).get("enable_internet", False)),
                bundle_mode=(task_data.get("runtime") or {}).get("kaggle_bundle_mode", "dataset"),
            )
            meta_file = stage_dir / "kernel-metadata.json"
            code_file = stage_dir / "run_kaggle.py"
            
            meta_json = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            code_content = code_file.read_text(encoding="utf-8") if code_file.exists() else ""

            self.send_json({
                "success": True,
                "backend": "kaggle",
                "script_path": str(code_file.relative_to(PROJECT_ROOT)),
                "code": code_content,
                "metadata": meta_json,
                "commands": {
                    "dataset_cmd": "ai-harness cloud consent --provider kaggle --confirm-upload\n# private by default; declare data_config.license before uploading a new dataset",
                    "push_cmd": f"python -m harness.cloud_runner --backend kaggle --project-root . --config {config_path}",
                    "kernel_cmd": f'python -m harness.templates.kaggle_template push-package --path "{stage_dir}"',
                    "status_cmd": f"kaggle kernels status {meta_json.get('id', 'antigravity/' + exp_id)}",
                    "output_cmd": f"kaggle kernels output {meta_json.get('id', 'antigravity/' + exp_id)} -p {ARTIFACTS_DIR / exp_id}",
                }
            })

    def handle_api_cloud_push_dataset(self, data: Dict[str, Any]) -> None:
        platform = (data.get("platform") or data.get("backend") or "kaggle").lower()
        if platform not in {"kaggle", "modal"}:
            self.send_json({"error": f"Unsupported provider: {platform}"}, status=400)
            return
        from harness.cloud_policy import (
            UploadConsentStore,
            dataset_fingerprint,
            private_upload_resource_names,
        )
        fingerprint = dataset_fingerprint(HARNESS_PROJECT)
        if not UploadConsentStore(HARNESS_PROJECT).is_granted(platform, fingerprint["fingerprint"]):
            self.send_json({
                "error": "Upload consent is required for this task, provider, and dataset fingerprint.",
                "fingerprint": fingerprint["fingerprint"],
                "consent_endpoint": "/api/cloud/consent",
            }, status=409)
            return
        resources = private_upload_resource_names(
            HARNESS_PROJECT,
            fingerprint_data=fingerprint,
            kaggle_username=KaggleKernelManager.get_authenticated_username() if platform == "kaggle" else None,
        )

        if platform == "modal":
            res = push_dataset_to_modal(volume_name=resources["modal_volume_name"], project_root=PROJECT_ROOT)
        else:
            dataset_slug = resources["kaggle_dataset_slug"]
            if not dataset_slug:
                self.send_json({"error": "Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry."}, status=409)
                return
            res = KaggleKernelManager.push_dataset_to_kaggle(dataset_slug=dataset_slug, project_root=PROJECT_ROOT)

        res.setdefault("dataset_fingerprint", fingerprint["fingerprint"])
        res.setdefault("visibility", "private")

        self.send_json(res, status=200 if res.get("success") else 400)

    def handle_api_cloud_kaggle_status(self, data: Dict[str, Any]) -> None:
        config_path = data.get("config_path", "")
        exp_id = data.get("exp_id") or Path(config_path).stem
        kernel_slug = data.get("kernel_slug")
        res = KaggleKernelManager.get_kernel_status(exp_id=exp_id, kernel_slug=kernel_slug)
        self.send_json(res)

    def handle_api_cloud_pull_output(self, data: Dict[str, Any]) -> None:
        platform = (data.get("platform") or data.get("backend") or "").lower()
        job_id = data.get("job_id")
        if platform not in {"kaggle", "modal"}:
            self.send_json({"error": "Choose a tracked Kaggle or Modal job."}, status=400)
            return
        if not job_id:
            self.send_json({"error": "Pull output through the tracked cloud job so task/config/artifact identity can be verified."}, status=409)
            return

        manager = CloudJobManager(PROJECT_ROOT)
        job = manager.get_job(str(job_id))
        if not job or job.get("backend") != platform:
            self.send_json({"error": "The job does not belong to this provider and workspace."}, status=404)
            return
        if job.get("status") != "COMPLETED" or not job.get("artifact_identity_verified"):
            self.send_json({"error": "Cloud job is not completed with a verified artifact identity.", "job": job}, status=409)
            return

        exp_dir = (PROJECT_ROOT / job["artifact_dir"]).resolve()
        try:
            exp_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            self.send_json({"error": "The tracked artifact path is outside this task workspace."}, status=403)
            return
        metrics_path = exp_dir / "metrics.json"
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Verified job metrics are unavailable: {exc}"}, status=409)
            return
        found_files = [path.name for path in exp_dir.iterdir() if path.is_file()]
        self.send_json({
            "success": True,
            "job_id": job_id,
            "exp_id": job["exp_id"],
            "artifacts_dir": str(exp_dir),
            "migrated_files": found_files,
            "metrics": metrics_data,
            "identity_verified": True,
        })

    def handle_api_kaggle_profile(self, data: Dict[str, Any]) -> None:
        user = data.get("username")
        key = data.get("key")
        p_name = data.get("profile_name")
        if not user or not key:
            self.send_json({"error": "username and key are required."}, status=400)
            return
        res = EnvironmentDetector.switch_kaggle_profile(user, key, p_name)
        self.send_json(res)

    def handle_api_cloud_run(self, data: Dict[str, Any]) -> None:
        config_path = data.get("config_path")
        if not config_path:
            self.send_json({"error": "config_path is required."}, status=400)
            return

        sess,obs=AgentBridge(PROJECT_ROOT).execute_action("run_full_validation",params=data,session_id=data.get("session_id"),budget=data.get("budget"))
        self.send_json({"success":obs.status=="SUCCESS","observation":asdict(obs),"session":asdict(sess),"job":obs.data})

    def handle_api_cloud_stop(self, data: Dict[str, Any]) -> None:
        job_id = data.get("job_id")
        if not job_id:
            self.send_json({"error": "job_id is required."}, status=400)
            return
        res = CloudJobManager().stop_job(job_id)
        status_code = 200 if res.get("success") else 400
        self.send_json(res, status=status_code)

    def handle_api_cloud_get_job(self, query_str: str) -> None:
        query = urllib.parse.parse_qs(query_str)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_json({"error": "Missing job_id parameter"}, status=400)
            return
        job = CloudJobManager().get_job(job_id)
        if not job:
            self.send_json({"error": f"Job '{job_id}' not found."}, status=404)
            return
        self.send_json({"job": job})

    def handle_api_cloud_logs_stream(self, query_str: str) -> None:
        query = urllib.parse.parse_qs(query_str)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_json({"error": "Missing job_id parameter"}, status=400)
            return

        start_byte = 0
        raw_start = query.get("start_byte", [None])[0]
        if raw_start is not None:
            try:
                start_byte = max(0, int(raw_start))
            except ValueError:
                self.send_json({"error": "start_byte must be an integer"}, status=400)
                return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        manager = CloudJobManager()
        for chunk in manager.stream_logs(job_id, start_byte=start_byte):
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def handle_api_cloud_audit(self, data: Dict[str, Any]) -> None:
        exp_id = data.get("exp_id")
        candidate_csv = data.get("candidate_csv")
        if not candidate_csv and exp_id:
            candidate_csv = str(ARTIFACTS_DIR / exp_id / "oof_predictions.csv")

        if not candidate_csv or not Path(candidate_csv).exists():
            self.send_json({"error": f"Candidate predictions not found: '{candidate_csv}'"}, status=400)
            return

        baseline_csv = data.get("baseline_csv")
        if not baseline_csv:
            best = TreeJournal().get_best_node()
            if best and (ARTIFACTS_DIR / best.id / "oof_predictions.csv").exists():
                baseline_csv = str(ARTIFACTS_DIR / best.id / "oof_predictions.csv")

        cmd = [sys.executable, str(HARNESS_DIR / "tools" / "audit_gate.py"), "--candidate", candidate_csv]
        if baseline_csv and Path(baseline_csv).exists():
            cmd.extend(["--baseline", baseline_csv])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        receipt = None
        receipt_path = PROJECT_ROOT / "AUDIT_RECEIPT.json"
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.send_json({
            "success": proc.returncode == 0,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "receipt": receipt,
        })

    def handle_api_llm_providers(self) -> None:
        try:
            providers = LLMEngine.list_providers()
            self.send_json({"providers": providers})
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_llm_sample_data(self) -> None:
        try:
            task_cfg = load_task_config()
            train_path_str = task_cfg.get("data_config", {}).get("train_path", "dataset/")
            train_dir = PROJECT_ROOT / train_path_str

            columns = []
            rows = []

            sample_file = None
            if train_dir.is_file():
                sample_file = train_dir
            elif train_dir.is_dir():
                for ext in ["*.tsv", "*.csv", "*.parquet", "*.jsonl"]:
                    matches = list(train_dir.glob(ext))
                    if matches:
                        sample_file = sorted(matches)[0]
                        break

            if sample_file and sample_file.exists():
                try:
                    import pandas as pd
                    sep = "\t" if sample_file.suffix == ".tsv" else ","
                    df = pd.read_csv(sample_file, sep=sep, nrows=5)
                    columns = [str(c) for c in df.columns]
                    rows = df.fillna("").to_dict(orient="records")
                except Exception:
                    pass

            if not columns:
                columns = [inp.get("name", "text") for inp in task_cfg.get("inputs", [])] or ["text"]
                rows = [{"text": "Sample text comment demonstrating LLM prompt extraction and classification."}]

            self.send_json({
                "columns": columns,
                "rows": rows,
                "total_samples": len(rows),
                "source_file": str(sample_file.relative_to(PROJECT_ROOT)).replace("\\", "/") if sample_file else None,
            })
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_llm_test_prompt(self, data: Dict[str, Any]) -> None:
        try:
            provider = data.get("provider", "gemini")
            model_name = data.get("model", "gemini-2.0-flash")
            system_prompt = data.get("system_prompt", "")
            user_prompt_template = data.get("user_prompt_template", "{text}")
            row_data = data.get("input_row", {})
            api_key = data.get("api_key")
            base_url = data.get("base_url")
            temperature = float(data.get("temperature", 0.0))
            max_tokens = int(data.get("max_tokens", 512))
            use_cot = bool(data.get("use_cot", False))
            few_shots = data.get("few_shot_examples", [])
            valid_classes = data.get("valid_classes", [])

            eff_sys, formatted_user = LLMEngine.format_prompt(
                template=user_prompt_template,
                row_data=row_data,
                system_prompt=system_prompt,
                few_shots=few_shots,
                use_cot=use_cot,
                valid_classes=valid_classes,
            )

            resp = LLMEngine.generate_completion(
                provider=provider,
                model_name=model_name,
                system_prompt=eff_sys,
                user_prompt=formatted_user,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )

            raw_text = resp.get("text", "")
            extracted = LLMEngine.extract_prediction(raw_text, valid_classes=valid_classes)

            self.send_json({
                "success": True,
                "provider": provider,
                "model": model_name,
                "effective_system_prompt": eff_sys,
                "effective_user_prompt": formatted_user,
                "raw_text": raw_text,
                "extracted": extracted,
                "latency_ms": resp.get("latency_ms", 0),
                "usage": resp.get("usage", {}),
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)

    def handle_api_llm_generate_peft(self, data: Dict[str, Any]) -> None:
        try:
            model_name = data.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
            train_path = data.get("train_path", "dataset/train.csv")
            output_dir = data.get("output_dir", str(ARTIFACTS_DIR / "exp_qlora"))
            text_column = data.get("text_column", "text")
            target_column = data.get("target_column", "target")
            lora_r = int(data.get("lora_r", 16))
            lora_alpha = int(data.get("lora_alpha", 32))
            lora_dropout = float(data.get("lora_dropout", 0.05))
            target_modules = data.get("target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            num_train_epochs = int(data.get("num_train_epochs", 3))
            batch_size = int(data.get("batch_size", 4))
            gradient_accumulation_steps = int(data.get("gradient_accumulation_steps", 4))
            learning_rate = float(data.get("learning_rate", 2e-4))

            script_code = LLMEngine.generate_qlora_script(
                model_name=model_name,
                train_path=train_path,
                output_dir=output_dir,
                text_column=text_column,
                target_column=target_column,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )

            save_to_disk = data.get("save_to_disk", False)
            saved_path = None
            if save_to_disk:
                scripts_dir = PROJECT_ROOT / "scripts" / "llm"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.split("/")[-1].lower())
                saved_file = scripts_dir / f"train_qlora_{clean_name}.py"
                saved_file.write_text(script_code, encoding="utf-8")
                saved_path = str(saved_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

            self.send_json({
                "success": True,
                "script_code": script_code,
                "saved_path": saved_path,
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)

    def handle_api_llm_save_key(self, data: Dict[str, Any]) -> None:
        try:
            key_name = data.get("key_name")
            key_value = data.get("key_value", "")
            if not key_name:
                self.send_json({"error": "key_name is required."}, status=400)
                return
            res = EnvironmentDetector.save_llm_api_key(key_name, key_value)
            self.send_json(res)
        except Exception as e:
            self.send_json({"error": str(e)}, status=400)

    def handle_api_llm_verify_key(self, data: Dict[str, Any]) -> None:
        try:
            provider = data.get("provider") or data.get("key_name") or ""
            key_value = data.get("key_value") or data.get("api_key")
            if not provider:
                self.send_json({"error": "provider or key_name is required.", "success": False, "valid": False}, status=400)
                return
            res = EnvironmentDetector.verify_llm_api_key(provider, key_value)
            self.send_json(res)
        except Exception as e:
            self.send_json({"error": str(e), "success": False, "valid": False}, status=400)

    def handle_api_llm_hf_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            search = params.get("search", [""])[0]
            pipeline = params.get("pipeline", [None])[0]
            author = params.get("author", [None])[0]
            family = params.get("family", [None])[0]
            param_range = params.get("param_range", [None])[0]
            quantization = params.get("quantization", [None])[0]
            sort = params.get("sort", ["downloads"])[0]
            limit = int(params.get("limit", [30])[0])
            force_refresh = params.get("force_refresh", ["false"])[0].lower() in ("true", "1", "yes")
            models = LLMEngine.fetch_hf_models(
                query=search,
                pipeline_tag=pipeline,
                author=author,
                family=family,
                param_range=param_range,
                quantization=quantization,
                limit=limit,
                sort=sort,
                force_refresh=force_refresh,
            )
            self.send_json({"models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_unsloth_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            search = params.get("search", [""])[0]
            family = params.get("family", [None])[0]
            param_range = params.get("param_range", [None])[0]
            quantization = params.get("quantization", [None])[0]
            sort = params.get("sort", ["downloads"])[0]
            limit = int(params.get("limit", [50])[0])
            force_refresh = params.get("force_refresh", ["false"])[0].lower() in ("true", "1", "yes")
            models = LLMEngine.fetch_unsloth_models(
                query=search,
                family=family,
                param_range=param_range,
                quantization=quantization,
                limit=limit,
                sort=sort,
                force_refresh=force_refresh,
            )
            self.send_json({"models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_provider_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            provider = params.get("provider", [""])[0]
            api_key = params.get("api_key", [None])[0]
            if not provider:
                self.send_json({"error": "provider is required"}, status=400)
                return
            models = LLMEngine.fetch_provider_live_models(provider=provider, api_key=api_key)
            self.send_json({"provider": provider, "models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_generate_unsloth(self, data: Dict[str, Any]) -> None:
        try:
            model_name = data.get("model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            train_path = data.get("train_path", "dataset/train.csv")
            output_dir = data.get("output_dir", str(ARTIFACTS_DIR / "exp_unsloth_lora"))
            text_column = data.get("text_column", "text")
            target_column = data.get("target_column", "target")
            max_seq_length = int(data.get("max_seq_length", 2048))
            lora_r = int(data.get("lora_r", 16))
            lora_alpha = int(data.get("lora_alpha", 32))
            lora_dropout = float(data.get("lora_dropout", 0.0))
            num_train_epochs = int(data.get("num_train_epochs", 3))
            batch_size = int(data.get("batch_size", 4))
            gradient_accumulation_steps = int(data.get("gradient_accumulation_steps", 4))
            learning_rate = float(data.get("learning_rate", 2e-4))

            script_code = LLMEngine.generate_unsloth_script(
                model_name=model_name,
                train_path=train_path,
                output_dir=output_dir,
                text_column=text_column,
                target_column=target_column,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )

            save_to_disk = data.get("save_to_disk", False)
            saved_path = None
            if save_to_disk:
                scripts_dir = PROJECT_ROOT / "scripts" / "llm"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.split("/")[-1].lower())
                saved_file = scripts_dir / f"train_unsloth_{clean_name}.py"
                saved_file.write_text(script_code, encoding="utf-8")
                saved_path = str(saved_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

            self.send_json({
                "success": True,
                "script_code": script_code,
                "saved_path": saved_path,
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)



def run_server(port: int = 8080) -> None:
    web_dir = Path(__file__).resolve().parent / "web"
    web_dir.mkdir(parents=True, exist_ok=True)

    handler = DashboardHandler
    socketserver.TCPServer.allow_reuse_address = True

    current_port = port
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with socketserver.ThreadingTCPServer(("127.0.0.1", current_port), handler) as httpd:
                print("=" * 65)
                print("🚀 ML/DL Experiment Dashboard & Priming Server Active")
                print(f"📡 Web URL:  http://localhost:{current_port}")
                print(f"📋 API Root: http://localhost:{current_port}/api/tree")
                print(f"🎯 Priming:  http://localhost:{current_port}/api/priming_prompt")
                print("=" * 65)
                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e) or attempt < max_retries - 1:
                print(f"Port {current_port} is busy, trying {current_port + 1}...")
                current_port += 1
            else:
                raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ML/DL Experiment Dashboard & Priming Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not attempt to open a browser window")
    args = parser.parse_args()
    run_server(port=args.port)


if __name__ == "__main__":
    main()
