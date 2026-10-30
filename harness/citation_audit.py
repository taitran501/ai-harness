"""Task-scoped Markdown citation discovery, evidence checks, and receipts."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import base64
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from harness.citation_sources import CitationSourceError, SourceRegistry, normalize_topic
from harness.io_utils import atomic_write_json
from harness.state_store import workspace_lock

POLICY_VERSION = "citation-audit.v1"
DISCUSSION_RECEIPT_REQUIRED = "__AI_HARNESS_DISCUSSION_RECEIPT_REQUIRED__"


class CitationAuditInputError(ValueError):
    """The requested audit input or claim document is invalid."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_source_input_failure(error: Any) -> bool:
    message = str(error)
    prefixes = ("REGISTRY_READ_ERROR", "MALFORMED_REGISTRY", "INVALID_REGISTRY_SCHEMA",
                "INVALID_ENCODING", "MALFORMED_JSON", "SOURCE_READ_ERROR",
                "UNSUPPORTED_SCHEMA_ADAPTER_IN_REGISTRY", "UNSUPPORTED_SCHEMA_ADAPTER_VERSION",
                "NORMALIZED_SOURCE_CONTENT_MISMATCH")
    return message.startswith(prefixes)


def _digest_json(value: Any) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _audit_scope(root: Path) -> Dict[str, Any]:
    from harness.artifacts import contract_identity
    from harness.project import HarnessProject
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(root)
    if not project.task_config_path or not project.task_config_path.is_file():
        raise CitationAuditInputError("ACTIVE_TASK_CONFIG_UNRESOLVED")
    task_spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(project.root, project.task_config_path, task_spec, project.plugin_id or "default")
    plugin = load_task_plugin(project.plugin_id, context)
    identity = contract_identity(task_spec, plugin)
    return {"workspace_id": project.task_id or identity["task_sha256"],
            "task_id": project.task_id, "scorer_sha256": identity["scorer_sha256"]}


def _project_scope(project_root: Path) -> tuple[list[Path], Optional[str], Optional[str]]:
    """Resolve additional documents and active task identity without artifact traversal."""
    try:
        from harness.project import HarnessProject

        project = HarnessProject.discover(project_root)
        configured = list(getattr(project, "citation_document_paths", ()))
        task_id = getattr(project, "task_id", None)
        progress = getattr(project, "active_research_progress_path", None)
        if progress and Path(progress).is_file():
            configured.append(Path(progress))
        warning = None if task_id else "ACTIVE_TASK_SCOPE_UNRESOLVED"
        return [Path(item) for item in configured], task_id, warning
    except Exception:
        # A missing task configuration should not silently pick a different task's
        # progress file. Root Markdown and explicit files remain auditable.
        return [], None, "ACTIVE_TASK_SCOPE_UNRESOLVED"


def discover_documents(project_root: str | Path, files: Optional[Iterable[str | Path]] = None) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise CitationAuditInputError(f"PROJECT_ROOT_NOT_FOUND: {root}")
    requested: list[Path] = []
    scope_warning = None
    if files is not None:
        for item in files:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_relative_to(root):
                raise CitationAuditInputError(f"DOCUMENT_PATH_ESCAPE: {path}")
            if not path.is_file():
                raise CitationAuditInputError(f"DOCUMENT_NOT_FOUND: {path}")
            requested.append(path)
    else:
        requested.extend(path for path in root.glob("*.md") if path.is_file())
        aih = root / ".ai-harness"
        if aih.is_dir():
            requested.extend(path for path in aih.glob("*.md") if path.is_file())
        configured, _, scope_warning = _project_scope(root)
        for item in configured:
            path = item if item.is_absolute() else root / item
            path = path.resolve()
            if not path.is_relative_to(root):
                raise CitationAuditInputError(f"DOCUMENT_PATH_ESCAPE: {path}")
            if not path.is_file():
                raise CitationAuditInputError(f"CONFIGURED_DOCUMENT_NOT_FOUND: {path}")
            requested.append(path)
    unique = sorted(set(requested))
    documents = []
    for path in unique:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CitationAuditInputError(f"DOCUMENT_READ_ERROR: {path}: {exc}") from exc
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CitationAuditInputError(f"INVALID_ENCODING: {path}: {exc}") from exc
        documents.append({"path": path, "relative_path": path.relative_to(root).as_posix(),
                          "text": text, "sha256": _sha(raw)})
    return {"project_root": root, "documents_requested": len(unique), "documents_read": len(documents),
            "documents": documents, "coverage": "complete" if documents and not scope_warning else ("partial" if documents else "nothing_scanned"),
            "scope_warnings": [scope_warning] if scope_warning else []}


def _mask_markdown(text: str) -> str:
    """Blank fenced and inline code while preserving line positions and offsets."""
    chars = list(text)
    fenced = False
    fence_char = ""
    fence_len = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker and (not fenced or marker.group(1)[0] == fence_char and len(marker.group(1)) >= fence_len):
            if not fenced:
                fenced = True
                fence_char, fence_len = marker.group(1)[0], len(marker.group(1))
            else:
                fenced = False
            for i in range(offset, offset + len(line)):
                if chars[i] not in "\r\n":
                    chars[i] = " "
            offset += len(line)
            continue
        if fenced:
            for i in range(offset, offset + len(line)):
                if chars[i] not in "\r\n":
                    chars[i] = " "
        offset += len(line)
    visible = "".join(chars)
    # Inline code uses a matching run of backticks.
    for match in list(re.finditer(r"(`+)(.*?)\1", visible, re.DOTALL)):
        if "\n" not in match.group(0):
            for i in range(match.start(), match.end()):
                if chars[i] not in "\r\n":
                    chars[i] = " "
    return "".join(chars)


def parse_citations(markdown: str) -> list[Dict[str, Any]]:
    """Find supported citation forms outside code and incidental URL fragments."""
    visible = _mask_markdown(markdown)
    found: list[Dict[str, Any]] = []

    def add(topic_id: str, start: int, end: int, form: str, anchor=None):
        line = markdown.count("\n", 0, start) + 1
        line_start = markdown.rfind("\n", 0, start) + 1
        line_end = markdown.find("\n", start)
        if line_end < 0:
            line_end = len(markdown)
        found.append({"topic_id": normalize_topic(topic_id), "line": line, "start": start,
                      "end": end, "form": form, "comment_anchor": anchor,
                      "context": markdown[line_start:line_end].strip()})

    # Kaggle discussion links are parsed before generic Markdown link syntax is excluded.
    url_pattern = re.compile(r"https?://(?:www\.)?kaggle\.com/(?:c/[^/]+/)?(?:competitions/[^/]+/)?discussion/(\d+)(?:/[^\s)#]*)?(?:#((?:comment|post|reply)[-_]?[\w-]+))?", re.I)
    url_ranges = []
    for match in url_pattern.finditer(visible):
        add(match.group(1), match.start(), match.end(), "kaggle_url", match.group(2))
        url_ranges.append(match.span())
    for start, end in url_ranges:
        visible = visible[:start] + " " * (end - start) + visible[end:]
    # Other URLs are not citation contexts. Mask fragments and path contents so
    # a nearby citation keyword cannot turn an incidental URL fragment into a
    # topic reference.
    url_chars = list(visible)
    for match in re.finditer(r"https?://[^\s<>)]+", visible, re.I):
        for index in range(match.start(), match.end()):
            if url_chars[index] not in "\r\n":
                url_chars[index] = " "
    visible = "".join(url_chars)

    patterns = [
        (re.compile(r"\btopic\s*#?\s*(\d{5,12})\b", re.I), "topic_label"),
        (re.compile(r"\btopic_(\d{5,12})\.json\b", re.I), "topic_filename"),
    ]
    occupied = []
    for pattern, form in patterns:
        for match in pattern.finditer(visible):
            add(match.group(1), match.start(), match.end(), form)
            occupied.append(match.span())

    # Bare #ID needs an explicit source/citation word nearby. Six-digit color
    # literals are excluded even if the same paragraph contains prose.
    for match in re.finditer(r"#\s*(\d{5,12})\b", visible):
        token = match.group(1)
        if re.search(r"[a-f]", token, re.I):
            continue
        if any(start <= match.start() < end for start, end in occupied):
            continue
        line_start = visible.rfind("\n", 0, match.start()) + 1
        line_end = visible.find("\n", match.start())
        if line_end < 0:
            line_end = len(visible)
        line = visible[line_start:line_end]
        explicit_context = re.search(r"\b(cite|citation|source|reference|discussion|kaggle|topic)\b", line, re.I)
        if not explicit_context:
            continue
        prefix = visible[max(line_start, match.start() - 24):match.start()]
        if re.search(r"\b(color|colour|hex|background|foreground)\s*:?\s*$", prefix, re.I):
            continue
        add(match.group(1), match.start(), match.end(), "explicit_bare_id")
    unique = {}
    for item in found:
        key = (item["topic_id"], item["line"], item["start"], item["form"])
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["start"], item["topic_id"]))


def _normalize_excerpt(text: str) -> str:
    # Only canonical Unicode and whitespace normalization; keep punctuation,
    # numerals, negation, and short technical terms intact.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def verify_claim_evidence(project_root: str | Path, claim: Dict[str, Any], evidence: Dict[str, Any], *, registry=None) -> Dict[str, Any]:
    registry = registry or SourceRegistry(project_root)
    try:
        source = registry.resolve_source(evidence.get("topic_id"), source_hash=evidence.get("source_sha256"),
                                         snapshot_version=evidence.get("snapshot_version"))
    except (CitationSourceError, TypeError) as exc:
        return {"integrity": "invalid" if "HASH_MISMATCH" in str(exc) or "ESCAPE" in str(exc) else "unverified",
                "claim_support": "not_reviewed", "error": str(exc)}
    segment = next((item for item in source["segments"] if item.get("locator") == evidence.get("locator")), None)
    if segment is None:
        return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "SEGMENT_LOCATOR_NOT_FOUND"}
    excerpt = evidence.get("excerpt")
    if not isinstance(excerpt, str) or not excerpt.strip():
        return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "EMPTY_EXCERPT"}
    full_text = segment.get("text", "")
    start = evidence.get("start_offset")
    end = evidence.get("end_offset")
    if start is not None or end is not None:
        if type(start) is not int or type(end) is not int or start < 0 or end < start or end > len(full_text):
            return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "INVALID_EXCERPT_OFFSETS"}
        actual = full_text[start:end]
        if _normalize_excerpt(actual) != _normalize_excerpt(excerpt):
            return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "EXCERPT_OFFSET_MISMATCH"}
    elif _normalize_excerpt(excerpt) not in _normalize_excerpt(full_text):
        return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "EXCERPT_NOT_IN_SELECTED_SEGMENT"}
    else:
        normalized_full = _normalize_excerpt(full_text)
        normalized_excerpt = _normalize_excerpt(excerpt)
        if normalized_full.count(normalized_excerpt) != 1:
            return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "AMBIGUOUS_EXCERPT_OCCURRENCE"}
    expected_hash = evidence.get("excerpt_sha256")
    if expected_hash and expected_hash != _sha(excerpt.encode("utf-8")):
        return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "EXCERPT_HASH_MISMATCH"}
    if evidence.get("source_sha256") and evidence["source_sha256"] != source["raw_sha256"]:
        return {"integrity": "invalid", "claim_support": "not_reviewed", "error": "SOURCE_HASH_MISMATCH"}
    return {"integrity": "verified", "claim_support": "not_reviewed", "source_id": source["source_id"],
            "source_sha256": source["raw_sha256"], "locator": segment["locator"],
            "segment_sha256": segment["segment_sha256"], "excerpt_sha256": _sha(excerpt.encode("utf-8")),
            "source_complete": bool(source.get("completeness", {}).get("complete", True))}


def _claim_document_path(claim: Dict[str, Any]) -> Optional[str]:
    document = claim.get("document") or {}
    return document.get("path") if isinstance(document, dict) else None


def _claim_matches_citation(root: Path, claim: Dict[str, Any], citation: Dict[str, Any], document: Dict[str, Any]) -> bool:
    """Bind a citation only to the claim's own topic and explicit document span."""
    topic_values = [claim.get("topic_id")]
    topic_values.extend(item.get("topic_id") for item in claim.get("evidence", []) if isinstance(item, dict))
    topics = set()
    for value in topic_values:
        if value is not None:
            try:
                topics.add(normalize_topic(value))
            except CitationSourceError:
                pass
    if citation["topic_id"] not in topics:
        return False

    claim_path = _claim_document_path(claim)
    if claim_path:
        path = Path(claim_path).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.is_relative_to(root) or path != document["path"]:
            return False
    elif not claim.get("ledger_field"):
        return False

    claim_text = claim.get("claim_text", "")
    binding = claim.get("document") or {}
    if binding.get("start_line") is not None and binding.get("end_line") is not None:
        lines = document["text"].splitlines()
        start_line, end_line = binding["start_line"], binding["end_line"]
        if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
            return False
        if end_line > len(lines) or claim_text not in "\n".join(lines[start_line - 1:end_line]):
            return False
        return start_line <= citation["line"] <= end_line
    return claim_text in citation["context"]


def _load_claims(path: Optional[str | Path], root: Path) -> list[Dict[str, Any]]:
    if path is None:
        return []
    claims_path = Path(path).expanduser()
    if not claims_path.is_absolute():
        claims_path = root / claims_path
    claims_path = claims_path.resolve()
    if not claims_path.is_relative_to(root) or not claims_path.is_file():
        raise CitationAuditInputError(f"CLAIMS_FILE_NOT_FOUND_OR_OUTSIDE_ROOT: {claims_path}")
    try:
        payload = json.loads(claims_path.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise CitationAuditInputError(f"INVALID_ENCODING: {claims_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CitationAuditInputError(f"MALFORMED_CLAIMS_JSON: {claims_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("claims"), list):
        raise CitationAuditInputError("INVALID_CLAIMS_SCHEMA: expected schema_version=1 and claims list")
    claim_ids = set()
    for claim in payload["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str) or not isinstance(claim.get("claim_text"), str):
            raise CitationAuditInputError("INVALID_CLAIMS_SCHEMA: each claim needs claim_id and claim_text")
        if claim["claim_id"] in claim_ids:
            raise CitationAuditInputError("INVALID_CLAIMS_SCHEMA: claim_id values must be unique")
        claim_ids.add(claim["claim_id"])
        try:
            from harness.policy_schema import validate_document
            validate_document("citation_claim", claim, version=1)
        except ValueError as exc:
            raise CitationAuditInputError(f"INVALID_CLAIMS_SCHEMA: {exc}") from exc
        document = claim.get("document")
        if isinstance(document, dict):
            document_path = Path(document["path"]).expanduser()
            document_path = document_path.resolve() if document_path.is_absolute() else (root / document_path).resolve()
            if not document_path.is_relative_to(root):
                raise CitationAuditInputError("CLAIM_DOCUMENT_PATH_ESCAPE")
            document["path"] = document_path.relative_to(root).as_posix()
    return payload["claims"]


def _load_verdict(project_root: Path, claim: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    verdict_ref = claim.get("verdict_ref")
    if not verdict_ref:
        return None
    from harness.citation_review import CitationReviewError, validate_verifier_execution
    try:
        return validate_verifier_execution(project_root, verdict_ref, claim)
    except (CitationReviewError, OSError, ValueError, TypeError):
        return None


def audit_claims(project_root: str | Path, *, files: Optional[Iterable[str | Path]] = None,
                 claims_file: Optional[str | Path] = None, strict: bool = False,
                 persist_receipt: bool = True) -> Dict[str, Any]:
    discovered = discover_documents(project_root, files)
    root = discovered["project_root"]
    claims = _load_claims(claims_file, root)
    registry = SourceRegistry(root)
    results = []
    findings = []
    citations_discovered = 0
    citations_bound = 0
    sources_read: set[str] = set()
    matched_claim_ids: set[str] = set()
    all_citation_pass = True
    try:
        audit_scope = _audit_scope(root)
        scope_valid = bool(audit_scope.get("task_id") and audit_scope.get("scorer_sha256"))
        scope_warnings = list(discovered.get("scope_warnings", []))
    except Exception as exc:
        audit_scope = {"workspace_id": None, "task_id": None, "scorer_sha256": None}
        scope_valid = False
        scope_warnings = list(discovered.get("scope_warnings", []))
        scope_warnings.append(f"ACTIVE_TASK_SCORER_SCOPE_UNRESOLVED: {exc}")
    for document in discovered["documents"]:
        citations = parse_citations(document["text"])
        citations_discovered += len(citations)
        for citation in citations:
            matching = [claim for claim in claims
                        if _claim_matches_citation(root, claim, citation, document)]
            claim_results = []
            for claim in matching:
                citations_bound += 1
                matched_claim_ids.add(claim["claim_id"])
                bound_evidence = []
                for evidence in claim.get("evidence", []):
                    try:
                        if normalize_topic(evidence.get("topic_id")) == citation["topic_id"]:
                            bound_evidence.append(evidence)
                    except (CitationSourceError, AttributeError):
                        continue
                evidence_results = [verify_claim_evidence(root, claim, evidence, registry=registry)
                                    for evidence in bound_evidence]
                for evidence_result in evidence_results:
                    if _is_source_input_failure(evidence_result.get("error", "")):
                        raise CitationAuditInputError(evidence_result["error"])
                for result in evidence_results:
                    if result.get("source_id"):
                        sources_read.add(result["source_id"])
                verdict = _load_verdict(root, claim)
                support = verdict.get("verdict") if verdict else "not_reviewed"
                integrity = "verified" if evidence_results and all(item["integrity"] == "verified" for item in evidence_results) else (
                    "invalid" if any(item["integrity"] == "invalid" for item in evidence_results) else "unverified")
                if any(item.get("source_complete") is False for item in evidence_results):
                    integrity = "unverified"
                claim_result = {"claim_id": claim["claim_id"], "claim_sha256": _sha(claim["claim_text"].encode("utf-8")),
                                "integrity": integrity, "claim_support": support,
                                "evidence": evidence_results, "verdict_ref": claim.get("verdict_ref"),
                                "claim": dict(claim),
                                "evidence_bindings": bound_evidence}
                claim_results.append(claim_result)
                if integrity != "verified" or support != "supported" or not (claim.get("ledger_field") or
                        isinstance(claim.get("document"), dict) and claim["document"].get("path") and
                        claim["document"].get("start_line") and claim["document"].get("end_line")):
                    all_citation_pass = False
                    findings.append({"path": document["relative_path"], "line": citation["line"],
                                     "topic_id": citation["topic_id"], "claim_id": claim["claim_id"],
                                     "issue": "CLAIM_NOT_VALIDLY_SUPPORTED", "severity": "blocking",
                                     "summary": f"Evidence integrity is {integrity}; recorded claim support is {support}, or the claim binding is incomplete."})
            if not matching:
                try:
                    source = registry.resolve_source(citation["topic_id"])
                    sources_read.add(source["source_id"])
                    integrity = "verified"
                except CitationSourceError as exc:
                    if _is_source_input_failure(exc):
                        raise CitationAuditInputError(str(exc)) from exc
                    integrity = "unverified"
                    findings.append({"path": document["relative_path"], "line": citation["line"],
                                     "topic_id": citation["topic_id"], "issue": "MISSING_OR_AMBIGUOUS_EVIDENCE",
                                     "severity": "blocking", "summary": str(exc)})
                claim_support = "not_reviewed"
                all_citation_pass = False
            else:
                integrity = "verified" if all(item["integrity"] == "verified" for item in claim_results) else "invalid"
                claim_support = ("supported" if all(item["claim_support"] == "supported" for item in claim_results)
                                 else "insufficient")
            # Advisory only: terms cannot establish support. Limit lint to visibly
            # distinctive words, evaluated against the selected topic segments.
            if not matching and integrity == "verified":
                source = registry.resolve_source(citation["topic_id"])
                source_text = " ".join(seg["text"] for seg in source["segments"]).casefold()
                terms = {word.casefold() for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", citation["context"])}
                missing = sorted(term for term in terms if term not in source_text and term not in {"topic", "source", "discussion", "claim", "citation"})
                if len(missing) >= 2:
                    findings.append({"path": document["relative_path"], "line": citation["line"],
                                     "topic_id": citation["topic_id"], "issue": "UNGROUNDED_TOPIC_CITATION",
                                     "severity": "warning", "missing_terms": missing,
                                     "summary": "Advisory keyword lint found terms absent from the registered source; no support was inferred."})
            results.append({"path": document["relative_path"], "line": citation["line"],
                            "topic_id": citation["topic_id"], "form": citation["form"],
                            "evidence_integrity": integrity, "claim_support": claim_support,
                            "claims": claim_results})
    selected_paths = {item["path"] for item in discovered["documents"]}
    selected_relpaths = {item["relative_path"] for item in discovered["documents"]}
    for claim in claims:
        claim_path = _claim_document_path(claim)
        if claim_path:
            path = Path(claim_path).expanduser()
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            in_scope = path in selected_paths and path.is_relative_to(root)
        else:
            in_scope = bool(claim.get("ledger_field") and any(
                Path(path).name == "RESEARCH_PROGRESS.md" for path in selected_relpaths))
        if in_scope and claim["claim_id"] not in matched_claim_ids:
            all_citation_pass = False
            findings.append({"path": claim_path or "<active research progress>",
                             "line": (claim.get("document") or {}).get("start_line", 0),
                             "topic_id": claim.get("topic_id"), "claim_id": claim["claim_id"],
                             "issue": "CLAIM_NOT_BOUND_TO_DISCOVERED_CITATION", "severity": "blocking",
                             "summary": "The explicit claim did not bind to a discovered citation in its selected document span."})
    coverage = discovered["coverage"]
    if coverage == "nothing_scanned":
        all_citation_pass = False
    receipt = None
    if persist_receipt and coverage != "nothing_scanned":
        receipt_body = {"schema_version": 1, "subject_sha256": _digest_json([{"path": item["relative_path"], "sha256": item["sha256"]} for item in discovered["documents"]]),
                        "scope": audit_scope, "policy_version": POLICY_VERSION,
                        "coverage": coverage, "citations_discovered": citations_discovered,
                        "citations_bound": citations_bound, "source_ids": sorted(sources_read),
                        "documents": [{"path": item["relative_path"], "sha256": item["sha256"]}
                                      for item in discovered["documents"]],
                        "results": results, "claim_results": [item for result in results for item in result["claims"]]}
        receipt_body["receipt_digest"] = _digest_json(receipt_body)
        from harness.policy_schema import validate_document
        validate_document("citation_audit_receipt", receipt_body, version=1)
        audits_dir = root / ".ai-harness" / "citation_audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = audits_dir / f"{receipt_body['receipt_digest']}.json"
        with workspace_lock(audits_dir):
            if not receipt_path.exists():
                atomic_write_json(receipt_path, receipt_body, indent=2, ensure_ascii=False)
        receipt = {"path": receipt_path.relative_to(root).as_posix(), "digest": receipt_body["receipt_digest"]}
    blocking = any(item["severity"] == "blocking" for item in findings)
    warnings = any(item["severity"] == "warning" for item in findings)
    if blocking:
        status = "findings"
    elif warnings:
        status = "warnings"
    else:
        status = "clean"
    report = {"schema_version": 1, "status": status, "coverage": coverage,
              "scope_warnings": scope_warnings, "scope": audit_scope,
              "documents_requested": discovered["documents_requested"], "documents_read": discovered["documents_read"],
              "citations_discovered": citations_discovered, "citations_bound": citations_bound,
              "sources_read": len(sources_read), "findings": findings, "results": results,
              "receipt": receipt, "strict": bool(strict)}
    report["strict_pass"] = bool(scope_valid and coverage == "complete" and citations_discovered > 0 and all_citation_pass and not blocking)
    return report


def validate_audit_receipt(project_root: str | Path, receipt_ref: str | Path | Dict[str, Any], *, expected_scope=None) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    ref = receipt_ref.get("path") if isinstance(receipt_ref, dict) else receipt_ref
    expected_digest = receipt_ref.get("digest") if isinstance(receipt_ref, dict) else None
    path = Path(ref)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    audit_root = (root / ".ai-harness" / "citation_audits").resolve()
    if not path.is_relative_to(audit_root) or not path.is_file():
        raise CitationAuditInputError("AUDIT_RECEIPT_NOT_FOUND_OR_OUTSIDE_ROOT")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CitationAuditInputError(f"INVALID_AUDIT_RECEIPT: {exc}") from exc
    if not isinstance(receipt, dict):
        raise CitationAuditInputError("INVALID_AUDIT_RECEIPT: expected a JSON object")
    digest = receipt.pop("receipt_digest", None)
    if digest != _digest_json(receipt):
        raise CitationAuditInputError("AUDIT_RECEIPT_DIGEST_MISMATCH")
    if expected_digest is not None and expected_digest != digest:
        raise CitationAuditInputError("AUDIT_RECEIPT_REFERENCE_DIGEST_MISMATCH")
    receipt["receipt_digest"] = digest
    if receipt.get("schema_version") != 1 or receipt.get("policy_version") != POLICY_VERSION:
        raise CitationAuditInputError("AUDIT_RECEIPT_VERSION_OR_POLICY_MISMATCH")
    scope = receipt.get("scope") or {}
    if not scope.get("task_id") or not scope.get("scorer_sha256"):
        raise CitationAuditInputError("AUDIT_RECEIPT_ACTIVE_TASK_SCOPE_UNRESOLVED")
    try:
        current_scope = _audit_scope(root)
    except Exception as exc:
        raise CitationAuditInputError(f"ACTIVE_TASK_SCOPE_UNRESOLVED: {exc}") from exc
    if scope != current_scope:
        raise CitationAuditInputError("AUDIT_RECEIPT_SCOPE_MISMATCH")
    from harness.policy_schema import validate_document
    try:
        validate_document("citation_audit_receipt", receipt, version=1)
    except ValueError as exc:
        raise CitationAuditInputError(f"AUDIT_RECEIPT_SCHEMA_INVALID: {exc}") from exc
    if expected_scope is not None and any(scope.get(key) != value for key, value in expected_scope.items()):
        raise CitationAuditInputError("AUDIT_RECEIPT_SCOPE_MISMATCH")
    if receipt.get("coverage") != "complete" or not receipt.get("claim_results"):
        raise CitationAuditInputError("AUDIT_RECEIPT_NOT_VERIFIED")
    from harness.project import HarnessProject
    active_progress = HarnessProject.discover(root).active_research_progress_path
    active_progress = active_progress.resolve() if active_progress else None
    for item in receipt.get("documents", []):
        document_path = (root / item.get("path", "")).resolve()
        if not document_path.is_relative_to(root) or not document_path.is_file():
            raise CitationAuditInputError("AUDIT_RECEIPT_DOCUMENT_NOT_FOUND_OR_OUTSIDE_ROOT")
        if document_path == active_progress:
            continue
        try:
            current_hash = _sha(document_path.read_bytes())
        except OSError as exc:
            raise CitationAuditInputError(f"AUDIT_RECEIPT_DOCUMENT_READ_ERROR: {exc}") from exc
        if current_hash != item.get("sha256"):
            raise CitationAuditInputError("AUDIT_RECEIPT_DOCUMENT_HASH_MISMATCH")
    registry = SourceRegistry(root)
    from harness.citation_review import CitationReviewError, validate_verifier_execution
    for item in receipt["claim_results"]:
        claim = item.get("claim") or {}
        if item.get("integrity") != "verified" or item.get("claim_support") != "supported" or not claim.get("claim_text"):
            raise CitationAuditInputError("AUDIT_RECEIPT_CONTAINS_UNSUPPORTED_CLAIMS")
        evidence = item.get("evidence_bindings") or []
        if not evidence or any(verify_claim_evidence(root, claim, ref, registry=registry).get("integrity") != "verified"
                               for ref in evidence):
            raise CitationAuditInputError("AUDIT_RECEIPT_EVIDENCE_STALE_OR_INVALID")
        try:
            verdict = validate_verifier_execution(root, item.get("verdict_ref"), claim)
        except (CitationReviewError, TypeError) as exc:
            raise CitationAuditInputError(f"AUDIT_RECEIPT_VERDICT_INVALID: {exc}") from exc
        if verdict.get("verdict") != "supported":
            raise CitationAuditInputError("AUDIT_RECEIPT_VERDICT_NOT_SUPPORTED")
        ledger_field = claim.get("ledger_field")
        document = claim.get("document") or {}
        if ledger_field:
            match = re.fullmatch(r"entry:([^/]+)/field:([A-Za-z0-9_]+)", str(ledger_field))
            if not match:
                raise CitationAuditInputError("AUDIT_RECEIPT_LEDGER_BINDING_INVALID")
            try:
                from harness.project import HarnessProject
                progress_path = HarnessProject.discover(root).active_research_progress_path
                if progress_path is None:
                    raise CitationAuditInputError("AUDIT_RECEIPT_ACTIVE_TASK_UNRESOLVED")
                ledger_path = progress_path.parent / "research_memory.json"
                ledger = json.loads(ledger_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CitationAuditInputError(f"AUDIT_RECEIPT_LEDGER_UNAVAILABLE: {exc}") from exc
            entry_id, field_name = match.groups()
            entry = next((item for item in ledger.get("entries", []) if item.get("id") == entry_id), None)
            if entry is None or field_name not in entry or str(entry[field_name]) != claim.get("claim_text"):
                raise CitationAuditInputError("AUDIT_RECEIPT_LEDGER_CLAIM_CHANGED")
        elif isinstance(document, dict) and document.get("path"):
            document_path = Path(document["path"])
            if not document_path.is_absolute():
                document_path = root / document_path
            document_path = document_path.resolve()
            if not document_path.is_relative_to(root) or not document_path.is_file():
                raise CitationAuditInputError("AUDIT_RECEIPT_CLAIM_DOCUMENT_INVALID")
            start_line, end_line = document.get("start_line"), document.get("end_line")
            if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
                raise CitationAuditInputError("AUDIT_RECEIPT_CLAIM_SPAN_REQUIRED")
            try:
                current = document_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                raise CitationAuditInputError(f"AUDIT_RECEIPT_CLAIM_DOCUMENT_READ_ERROR: {exc}") from exc
            if end_line > len(current) or claim["claim_text"] not in "\n".join(current[start_line - 1:end_line]):
                raise CitationAuditInputError("AUDIT_RECEIPT_CLAIM_DOCUMENT_CHANGED")
            document_text = "\n".join(current)
            document_record = {"path": document_path, "relative_path": document_path.relative_to(root).as_posix(),
                               "text": document_text}
            if not any(_claim_matches_citation(root, claim, citation, document_record)
                       for citation in parse_citations(document_text)):
                raise CitationAuditInputError("AUDIT_RECEIPT_CITATION_BINDING_CHANGED")
        else:
            raise CitationAuditInputError("AUDIT_RECEIPT_CLAIM_BINDING_REQUIRED")
    return receipt


def enforce_citation_admission(project_root: str | Path, subject: Dict[str, Any], *, action: str = "execute") -> Dict[str, Any]:
    """Apply the shared policy from recorded external-evidence dependencies."""
    research = subject.get("research") if isinstance(subject.get("research"), dict) else {}
    dependencies = list(subject.get("citation_dependencies") or research.get("citation_dependencies") or [])
    grounding = subject.get("grounding") if isinstance(subject.get("grounding"), dict) else {}
    dependencies.extend(grounding.get("citation_dependencies") or [])
    dependencies.extend(grounding.get("audit_receipts") or [])
    expected_claims = list(subject.get("citation_claims") or research.get("citation_claims") or [])
    expected_claims.extend(grounding.get("citation_claims") or [])
    if all(isinstance(item, str) for item in dependencies):
        dependencies = list(dict.fromkeys(dependencies))
    unique_claims = {}
    for item in expected_claims:
        if isinstance(item, dict):
            unique_claims[(item.get("claim_id"), item.get("claim_text"))] = item
    expected_claims = list(unique_claims.values()) + [item for item in expected_claims if not isinstance(item, dict)]
    if grounding.get("source_type") == "kaggle_discussion":
        if not dependencies:
            return {"status": "blocked", "admission": "blocked", "reason": "DISCUSSION_CITATION_RECEIPT_REQUIRED", "action": action, "receipt_digests": []}
    if DISCUSSION_RECEIPT_REQUIRED in dependencies:
        return {"status": "blocked", "admission": "blocked", "reason": "DISCUSSION_CITATION_RECEIPT_REQUIRED",
                "action": action, "receipt_digests": []}
    if not dependencies:
        if expected_claims:
            return {"status": "blocked", "admission": "blocked", "reason": "CITATION_RECEIPT_REQUIRED_FOR_CLAIMS",
                    "action": action, "receipt_digests": []}
        return {"status": "not_applicable", "admission": "not_applicable", "reason": "NO_EXTERNAL_CITATION_DEPENDENCIES", "action": action, "receipt_digests": []}
    checked = []
    try:
        for dependency in dependencies:
            checked.append(validate_audit_receipt(project_root, dependency))
    except CitationAuditInputError as exc:
        return {"status": "blocked", "admission": "blocked", "reason": str(exc), "action": action, "receipt_digests": []}
    for expected in expected_claims:
        claim_text = expected.get("claim_text") if isinstance(expected, dict) else None
        claim_id = expected.get("claim_id") if isinstance(expected, dict) else None
        claim_hash = _sha(str(claim_text).encode("utf-8")) if isinstance(claim_text, str) else None
        matched = any(
            result.get("claim_sha256") == claim_hash and
            (claim_id is None or result.get("claim_id") == claim_id)
            for receipt in checked for result in receipt.get("claim_results", [])
        )
        if not matched:
            return {"status": "blocked", "admission": "blocked", "reason": "CITATION_CLAIM_NOT_BOUND_TO_RECEIPT",
                    "action": action, "receipt_digests": []}
    return {"status": "allowed", "admission": "allowed", "reason": None, "action": action,
            "receipt_digests": [item["receipt_digest"] for item in checked]}


def build_portable_citation_bundle(project_root: str | Path, dependencies: Iterable[str | Path | Dict[str, Any]]) -> Dict[str, Any]:
    """Package validated receipt inputs without originating-machine paths."""
    root = Path(project_root).expanduser().resolve()
    sources: Dict[str, Dict[str, Any]] = {}
    receipts = []
    verdicts: Dict[str, Dict[str, Any]] = {}
    executions: Dict[str, Dict[str, Any]] = {}
    registry = SourceRegistry(root)
    from harness.citation_review import validate_verifier_execution

    for dependency in dependencies:
        receipt = validate_audit_receipt(root, dependency)
        receipts.append(receipt)
        for source_id in receipt.get("source_ids", []):
            record = next((item for item in registry._load()["sources"] if item.get("source_id") == source_id), None)
            if record is None:
                raise CitationAuditInputError("PORTABLE_BUNDLE_SOURCE_NOT_FOUND")
            resolved = registry.resolve_source(record["topic_id"], source_hash=record["raw_sha256"])
            sources[record["raw_sha256"]] = {"record": record,
                "snapshot_base64": base64.b64encode(resolved["snapshot_bytes"]).decode("ascii")}
        for result in receipt.get("claim_results", []):
            claim = result["claim"]
            validated = validate_verifier_execution(root, result["verdict_ref"], claim)
            verdict_path = root / result["verdict_ref"]
            verdict = json.loads(verdict_path.read_text(encoding="utf-8-sig"))
            execution_path = root / ".ai-harness" / "citation_audits" / "executions" / f"{validated['execution_id']}.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8-sig"))
            verdicts[result["verdict_ref"]] = verdict
            executions[validated["execution_id"]] = execution
    bundle = {"schema_version": 1, "receipts": receipts,
              "sources": [sources[key] for key in sorted(sources)],
              "verdicts": verdicts, "executions": executions}
    bundle["bundle_sha256"] = _digest_json(bundle)
    return bundle


def validate_portable_citation_bundle(bundle: Dict[str, Any], *, expected_task_id: Optional[str] = None,
                                      expected_scorer_sha256: Optional[str] = None,
                                      expected_claims: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Revalidate bundled source bytes, excerpts, verdict provenance, and scope."""
    from harness.citation_review import _claim_hash, _evidence_hash, _digest

    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise CitationAuditInputError("PORTABLE_BUNDLE_SCHEMA_INVALID")
    unsigned = dict(bundle)
    bundle_hash = unsigned.pop("bundle_sha256", None)
    if bundle_hash != _digest_json(unsigned):
        raise CitationAuditInputError("PORTABLE_BUNDLE_DIGEST_MISMATCH")
    source_map = {}
    from harness.citation_sources import _canonical_bytes, _normalized_segments
    for item in bundle.get("sources", []):
        record = item.get("record") or {}
        try:
            raw = base64.b64decode(item.get("snapshot_base64", ""), validate=True)
        except (ValueError, TypeError) as exc:
            raise CitationAuditInputError("PORTABLE_SOURCE_ENCODING_INVALID") from exc
        if _sha(raw) != record.get("raw_sha256"):
            raise CitationAuditInputError("PORTABLE_SOURCE_HASH_MISMATCH")
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CitationAuditInputError(f"PORTABLE_SOURCE_JSON_INVALID: {exc}") from exc
        adapter = record.get("schema_adapter")
        if adapter == "normalized_topic_v1":
            segments = _normalized_segments(payload)
        elif adapter == "data_wrapper_v1" and isinstance(payload, dict) and "data" in payload:
            segments = _normalized_segments(payload["data"])
        elif adapter == "string_list_v1" and isinstance(payload, list) and all(isinstance(v, str) for v in payload):
            segments = _normalized_segments(payload)
        else:
            raise CitationAuditInputError("PORTABLE_SOURCE_ADAPTER_INVALID")
        if segments != record.get("segments") or _sha(_canonical_bytes(segments)) != record.get("normalized_content_sha256"):
            raise CitationAuditInputError("PORTABLE_SOURCE_NORMALIZATION_MISMATCH")
        if not record.get("completeness", {}).get("complete", True):
            raise CitationAuditInputError("PORTABLE_SOURCE_INCOMPLETE")
        source_map[record["raw_sha256"]] = record

    verdicts = bundle.get("verdicts", {})
    executions = bundle.get("executions", {})
    receipt_digests = []
    reviewed_claims = set()
    for receipt_value in bundle.get("receipts", []):
        receipt = dict(receipt_value)
        receipt_digest = receipt.pop("receipt_digest", None)
        if receipt_digest != _digest_json(receipt):
            raise CitationAuditInputError("PORTABLE_RECEIPT_DIGEST_MISMATCH")
        scope = receipt.get("scope") or {}
        if ((expected_task_id is not None and scope.get("task_id") != expected_task_id) or
            (expected_scorer_sha256 is not None and scope.get("scorer_sha256") != expected_scorer_sha256)):
            raise CitationAuditInputError("PORTABLE_RECEIPT_SCOPE_MISMATCH")
        if receipt.get("policy_version") != POLICY_VERSION or receipt.get("coverage") != "complete":
            raise CitationAuditInputError("PORTABLE_RECEIPT_POLICY_OR_COVERAGE_INVALID")
        if not receipt.get("claim_results"):
            raise CitationAuditInputError("PORTABLE_RECEIPT_HAS_NO_REVIEWED_CLAIMS")
        for result in receipt.get("claim_results", []):
            claim = result.get("claim") or {}
            if result.get("integrity") != "verified" or result.get("claim_support") != "supported":
                raise CitationAuditInputError("PORTABLE_RECEIPT_UNSUPPORTED_CLAIM")
            evidence_bindings = result.get("evidence_bindings", [])
            if not evidence_bindings:
                raise CitationAuditInputError("PORTABLE_RECEIPT_MISSING_EVIDENCE_BINDING")
            for evidence in evidence_bindings:
                source = source_map.get(evidence.get("source_sha256"))
                try:
                    same_topic = source is not None and normalize_topic(source.get("topic_id")) == normalize_topic(evidence.get("topic_id"))
                except CitationSourceError:
                    same_topic = False
                if not same_topic:
                    raise CitationAuditInputError("PORTABLE_EVIDENCE_SOURCE_MISSING")
                segment = next((s for s in source["segments"] if s.get("locator") == evidence.get("locator")), None)
                excerpt = evidence.get("excerpt", "")
                if segment is None:
                    raise CitationAuditInputError("PORTABLE_EVIDENCE_EXCERPT_INVALID")
                start, end = evidence.get("start_offset"), evidence.get("end_offset")
                if start is not None or end is not None:
                    actual = segment["text"][start:end] if type(start) is int and type(end) is int and 0 <= start <= end <= len(segment["text"]) else None
                    if actual is None or _normalize_excerpt(actual) != _normalize_excerpt(excerpt):
                        raise CitationAuditInputError("PORTABLE_EVIDENCE_EXCERPT_INVALID")
                elif (_normalize_excerpt(excerpt) not in _normalize_excerpt(segment["text"]) or
                      _normalize_excerpt(segment["text"]).count(_normalize_excerpt(excerpt)) != 1):
                    raise CitationAuditInputError("PORTABLE_EVIDENCE_EXCERPT_INVALID")
                if evidence.get("excerpt_sha256") and evidence["excerpt_sha256"] != _sha(excerpt.encode("utf-8")):
                    raise CitationAuditInputError("PORTABLE_EVIDENCE_EXCERPT_HASH_MISMATCH")
            verdict = verdicts.get(result.get("verdict_ref"))
            if not verdict:
                raise CitationAuditInputError("PORTABLE_VERDICT_MISSING")
            execution = executions.get(verdict.get("execution_id"))
            if not execution or execution.get("status") != "completed" or execution.get("proposer_id") == execution.get("verifier_id"):
                raise CitationAuditInputError("PORTABLE_VERIFIER_EXECUTION_INVALID")
            if (verdict.get("claim_sha256") != _claim_hash(claim) or
                verdict.get("evidence_sha256") != _evidence_hash(claim) or verdict.get("verdict") != "supported"):
                raise CitationAuditInputError("PORTABLE_VERDICT_INPUT_MISMATCH")
            if (execution.get("claim_sha256") != verdict.get("claim_sha256") or
                execution.get("evidence_sha256") != verdict.get("evidence_sha256")):
                raise CitationAuditInputError("PORTABLE_EXECUTION_INPUT_MISMATCH")
            unsigned_verdict = dict(verdict)
            output_digest = unsigned_verdict.pop("output_digest", None)
            if output_digest != _digest(unsigned_verdict) or execution.get("output_digest") != output_digest:
                raise CitationAuditInputError("PORTABLE_VERDICT_DIGEST_MISMATCH")
            reviewed_claims.add((result.get("claim_id"), result.get("claim_sha256")))
        receipt_digests.append(receipt_digest)
    if not receipt_digests:
        raise CitationAuditInputError("PORTABLE_BUNDLE_HAS_NO_RECEIPTS")
    for claim in expected_claims or []:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_text"), str):
            raise CitationAuditInputError("PORTABLE_EXPECTED_CLAIM_INVALID")
        expected_hash = _sha(claim["claim_text"].encode("utf-8"))
        if not any(item_hash == expected_hash and (claim.get("claim_id") is None or item_id == claim.get("claim_id"))
                   for item_id, item_hash in reviewed_claims):
            raise CitationAuditInputError("PORTABLE_CLAIM_NOT_BOUND_TO_RECEIPT")
    return {"status": "allowed", "admission": "allowed", "reason": None,
            "receipt_digests": receipt_digests}

