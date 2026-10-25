"""Structured failure classification and actionable recovery guidance for AI agents."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class FailureCode:
    CONFIG_INVALID = "CONFIG_INVALID"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    IMPORT_SYMBOL_ERROR = "IMPORT_SYMBOL_ERROR"
    RESOURCE_OOM = "RESOURCE_OOM"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    DATA_CONTRACT_INVALID = "DATA_CONTRACT_INVALID"
    ARTIFACT_UNVERIFIABLE = "ARTIFACT_UNVERIFIABLE"
    UNKNOWN = "UNKNOWN"


def classify_failure(
    text: str,
    returncode: Optional[int] = None,
    exception: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Analyze error messages, output logs, or exceptions into a structured failure report.

    Returns a structured dictionary with:
    - code: stable error category code
    - category: high-level fault domain
    - confidence: "high", "medium", or "low"
    - retryable: bool
    - recovery: actionable guidance (mode, suggested_action, suggested_command)
    """
    combined = (text or "") + ("\n" + str(exception) if exception else "")

    # 1a. Missing symbol in installed module (ImportError: cannot import name 'X' from 'Y')
    sym_match = re.search(
        r"ImportError:\s*cannot import name ['\"]?([^'\"\s]+)['\"]?(?:\s+from\s+['\"]?([^'\"\s]+)['\"]?)?",
        combined,
        re.IGNORECASE,
    )
    if sym_match and not isinstance(exception, ModuleNotFoundError):
        symbol = sym_match.group(1)
        mod = sym_match.group(2) or "the target module"
        return {
            "code": FailureCode.IMPORT_SYMBOL_ERROR,
            "category": "environment",
            "confidence": "high",
            "retryable": False,
            "detail": f"Symbol '{symbol}' cannot be imported from '{mod}'",
            "recovery": {
                "mode": "inspect_import",
                "suggested_action": f"Check whether symbol '{symbol}' exists in '{mod}' or verify installed library version.",
                "suggested_command": None,
                "retryable": False,
            },
        }

    # 1b. Missing package/module (ModuleNotFoundError or No module named 'X')
    mod_match = re.search(
        r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([^'\"]+)['\"]",
        combined,
        re.IGNORECASE,
    )
    if mod_match or isinstance(exception, ModuleNotFoundError):
        pkg = None
        if mod_match:
            pkg = mod_match.group(1).split(".")[0]
        elif hasattr(exception, "name"):
            pkg = getattr(exception, "name", "").split(".")[0]
        pkg_display = pkg if pkg else "<package>"
        return {
            "code": FailureCode.DEPENDENCY_MISSING,
            "category": "environment",
            "confidence": "high",
            "retryable": False,
            "detail": f"Missing dependency: {pkg}" if pkg else "Missing Python module",
            "recovery": {
                "mode": "install_dependency",
                "suggested_action": f"Install missing dependency '{pkg_display}' in the environment.",
                "suggested_command": f"pip install {pkg_display}",
                "retryable": False,
            },
        }

    # 2. Out of Memory (CUDA or Host RAM)
    if (
        re.search(
            r"(?:CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError|MemoryError)",
            combined,
            re.IGNORECASE,
        )
        or isinstance(exception, MemoryError)
    ):
        return {
            "code": FailureCode.RESOURCE_OOM,
            "category": "resource",
            "confidence": "high",
            "retryable": False,
            "detail": "Out of memory encountered during training or inference",
            "recovery": {
                "mode": "reduce_resource_usage",
                "suggested_action": "Reduce batch size, sequence length, or enable gradient checkpointing.",
                "suggested_command": None,
                "retryable": False,
            },
        }

    # 3. Authentication & Cloud credentials
    if re.search(
        r"(?:401 Unauthorized|Kaggle authentication unavailable|KAGGLE_USERNAME|Modal token credentials not configured|kaggle\.json not found)",
        combined,
        re.IGNORECASE,
    ):
        return {
            "code": FailureCode.AUTH_REQUIRED,
            "category": "provider",
            "confidence": "high",
            "retryable": False,
            "detail": "Cloud execution provider rejected request due to missing or invalid credentials",
            "recovery": {
                "mode": "authenticate",
                "suggested_action": "Configure provider credentials or switch to local backend.",
                "suggested_command": None,
                "retryable": False,
            },
        }

    # 4. Quota exceeded & Rate limits
    if re.search(
        r"(?:429 Too Many Requests|Quota exceeded|Rate limit exceeded|exceeded your daily submission limit)",
        combined,
        re.IGNORECASE,
    ):
        return {
            "code": FailureCode.QUOTA_EXCEEDED,
            "category": "provider",
            "confidence": "high",
            "retryable": True,
            "detail": "Provider rate limit or quota exceeded",
            "recovery": {
                "mode": "backoff_or_switch",
                "suggested_action": "Wait for quota reset or switch to local execution backend.",
                "suggested_command": None,
                "retryable": True,
            },
        }

    # 5. Configuration invalid
    if re.search(
        r"(?:yaml\.parser\.ParserError|yaml\.scanner\.ScannerError|ValidationError|Invalid configuration|TaskSpecError|missing required field)",
        combined,
        re.IGNORECASE,
    ):
        return {
            "code": FailureCode.CONFIG_INVALID,
            "category": "configuration",
            "confidence": "high",
            "retryable": False,
            "detail": "Configuration syntax error or schema violation",
            "recovery": {
                "mode": "fix_config",
                "suggested_action": "Check YAML formatting and schema fields in experiment or task configuration.",
                "suggested_command": "ai-harness validate-task",
                "retryable": False,
            },
        }

    # 6. Artifact verification / drift
    if re.search(
        r"(?:hash drift|tampered evidence|checksum mismatch|ArtifactManifestError|Unverifiable run)",
        combined,
        re.IGNORECASE,
    ):
        return {
            "code": FailureCode.ARTIFACT_UNVERIFIABLE,
            "category": "artifact",
            "confidence": "high",
            "retryable": False,
            "detail": "Artifact hash drift or unverifiable evidence directory",
            "recovery": {
                "mode": "inspect_first",
                "suggested_action": "Inspect drift status and re-seal hashes if changes are verified.",
                "suggested_command": "ai-harness repair --check",
                "retryable": False,
            },
        }

    # 7. Data contract / column mismatch
    if re.search(
        r"(?:Column ['\"][^'\"]+['\"] not found|Task ground truth missing id_column|KeyError:\s*['\"][^'\"]+['\"]|Inconsistent class order)",
        combined,
        re.IGNORECASE,
    ):
        return {
            "code": FailureCode.DATA_CONTRACT_INVALID,
            "category": "data",
            "confidence": "medium",
            "retryable": False,
            "detail": "Dataset columns or split contracts do not match task specification",
            "recovery": {
                "mode": "inspect_data",
                "suggested_action": "Verify dataset columns against target_column and id_column in configs/task.yaml.",
                "suggested_command": "ai-harness validate-task",
                "retryable": False,
            },
        }

    # 8. Fallback / Unknown
    return {
        "code": FailureCode.UNKNOWN,
        "category": "code",
        "confidence": "low",
        "retryable": False,
        "detail": f"Execution exited with returncode {returncode}" if returncode is not None else "Execution failed",
        "recovery": {
            "mode": "inspect_logs",
            "suggested_action": "Inspect output logs to diagnose the experiment error.",
            "suggested_command": None,
            "retryable": False,
        },
    }
