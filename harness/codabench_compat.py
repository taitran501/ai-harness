"""Codabench platform compatibility layer (API client and artifact downloader).

Enables agents to query submission status, fetch evaluation logs, and download
prediction/scoring artifacts from Codabench without hand-written scripts.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile


class CodabenchClient:
    """Client for Codabench REST API."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        base_url: str = "https://www.codabench.org",
        project_root: Optional[str | Path] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or self._resolve_session_id(project_root)

    def _resolve_session_id(self, project_root: Optional[str | Path] = None) -> Optional[str]:
        # 1. Env var
        env_val = os.environ.get("CODABENCH_SESSION_ID")
        if env_val:
            return env_val.strip()

        # 2. File in project root or cwd
        candidates = [Path(".codabench_session"), Path(".codabench_token")]
        if project_root:
            p_root = Path(project_root)
            candidates.extend([p_root / ".codabench_session", p_root / ".codabench_token"])

        for cand in candidates:
            if cand.is_file():
                try:
                    return cand.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return None

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai-harness/1.0",
        }
        if self.session_id:
            headers["Cookie"] = f"sessionid={self.session_id}"
        return headers

    def get_submission(self, sub_id: int) -> Dict[str, Any]:
        """Fetch submission metadata, status, and scores."""
        import requests

        url = f"{self.base_url}/api/submissions/{sub_id}/"
        resp = requests.get(url, headers=self._get_headers())
        if resp.status_code != 200:
            raise RuntimeError(
                f"Codabench API request failed (HTTP {resp.status_code}) for submission {sub_id}: {resp.text[:200]}"
            )
        data = resp.json()
        scores = data.get("scores", []) or []
        primary_score = None
        for s in scores:
            if s.get("is_primary"):
                try:
                    primary_score = float(s.get("score"))
                except (TypeError, ValueError):
                    pass
                break
        if primary_score is None and scores:
            try:
                primary_score = float(scores[0].get("score"))
            except (TypeError, ValueError):
                pass

        return {
            "id": sub_id,
            "status": data.get("status"),
            "primary_score": primary_score,
            "scores": scores,
            "description": data.get("description", ""),
            "created_when": data.get("created_when"),
            "raw": data,
        }

    def get_details(self, sub_id: int) -> Dict[str, Any]:
        """Fetch execution details, log links, and artifact URLs."""
        import requests

        url = f"{self.base_url}/api/submissions/{sub_id}/get_details/"
        resp = requests.get(url, headers=self._get_headers())
        if resp.status_code != 200:
            raise RuntimeError(
                f"Codabench get_details failed (HTTP {resp.status_code}) for submission {sub_id}: {resp.text[:200]}"
            )
        return resp.json()

    def download_submission(
        self, sub_id: int, output_dir: str | Path
    ) -> Dict[str, Any]:
        """Download logs, prediction results, and scoring results into output_dir."""
        import requests

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        headers = self._get_headers()

        # 1. Meta
        meta_info = self.get_submission(sub_id)
        (out_path / "meta.json").write_text(
            json.dumps(meta_info.get("raw", meta_info), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2. Details
        details = self.get_details(sub_id)
        (out_path / "details.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        extracted_files: List[str] = ["meta.json", "details.json"]

        # 3. Logs
        for log_entry in details.get("logs", []):
            name = log_entry.get("name", "log")
            log_url = log_entry.get("data_file")
            if log_url:
                r_log = requests.get(log_url, headers=headers)
                if r_log.status_code == 200:
                    fn = f"{name}.txt" if not name.endswith(".txt") else name
                    (out_path / fn).write_bytes(r_log.content)
                    extracted_files.append(fn)

        # 4. Prediction Result
        pred_url = details.get("prediction_result")
        if pred_url:
            r_pred = requests.get(pred_url, headers=headers)
            if r_pred.status_code == 200 and len(r_pred.content) > 0:
                try:
                    with zipfile.ZipFile(io.BytesIO(r_pred.content)) as z:
                        for item in z.namelist():
                            content = z.read(item)
                            target_file = out_path / item
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            target_file.write_bytes(content)
                            extracted_files.append(item)
                except Exception:
                    (out_path / "prediction_result.zip").write_bytes(r_pred.content)
                    extracted_files.append("prediction_result.zip")

        # 5. Scoring Result
        score_url = details.get("scoring_result")
        if score_url:
            r_score = requests.get(score_url, headers=headers)
            if r_score.status_code == 200 and len(r_score.content) > 0:
                try:
                    with zipfile.ZipFile(io.BytesIO(r_score.content)) as z:
                        for item in z.namelist():
                            content = z.read(item)
                            target_file = out_path / item
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            target_file.write_bytes(content)
                            extracted_files.append(item)
                except Exception:
                    (out_path / "scoring_result.zip").write_bytes(r_score.content)
                    extracted_files.append("scoring_result.zip")

        return {
            "sub_id": sub_id,
            "status": meta_info.get("status"),
            "primary_score": meta_info.get("primary_score"),
            "output_dir": str(out_path),
            "files_downloaded": extracted_files,
        }
