#!/usr/bin/env python3
"""Generate LLM interpretations for LIME and SHAP explanations."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in model interpretability for autonomous vessel control. "
    "Explain why the controller assigned the LIME and SHAP weights to the features "
    "given the state at this step. Focus on causality and how the feature values "
    "relate to the rudder and throttle decisions. Be concise but specific."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lime-summary",
        type=Path,
        required=True,
        help="Path to the lime_summary.json file.",
    )
    parser.add_argument(
        "--shap-summary",
        type=Path,
        required=True,
        help="Path to the shap_summary.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "llm_reports",
        help="Directory where LLM explanations will be written.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        help="LLM chat completion endpoint (OpenAI-compatible).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        help="Model name to request from the LLM API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("LLM_API_KEY"),
        help="API key for the LLM provider. Defaults to LLM_API_KEY env var.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="LLM_API_KEY",
        help="Environment variable to read the API key from when --api-key is not set.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to guide the LLM.",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Optional file containing extra context to include in every prompt.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Limit the number of feature attributions per action (highest absolute weights).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="First step index to include (inclusive).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Last step index to include (inclusive).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to send to the LLM.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=90,
        help="Timeout in seconds for LLM API calls.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between API calls to respect rate limits.",
    )
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _trim_attributions(
    attributions: List[Dict[str, Any]],
    *,
    weight_key: str,
    max_features: Optional[int],
) -> List[Dict[str, Any]]:
    if max_features is None:
        return list(attributions)
    sorted_attr = sorted(
        attributions,
        key=lambda item: abs(float(item.get(weight_key, 0.0))),
        reverse=True,
    )
    return sorted_attr[:max_features]


def _index_by_step(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(item["step"]): item for item in items}


def _merge_steps(
    lime_data: List[Dict[str, Any]],
    shap_data: List[Dict[str, Any]],
    *,
    start_step: Optional[int],
    end_step: Optional[int],
    max_steps: Optional[int],
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    lime_by_step = _index_by_step(lime_data)
    shap_by_step = _index_by_step(shap_data)
    shared_steps = sorted(set(lime_by_step) & set(shap_by_step))
    if start_step is not None:
        shared_steps = [step for step in shared_steps if step >= start_step]
    if end_step is not None:
        shared_steps = [step for step in shared_steps if step <= end_step]
    if max_steps is not None:
        shared_steps = shared_steps[:max_steps]
    return [(step, lime_by_step[step], shap_by_step[step]) for step in shared_steps]


def _read_context_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Context file '{path}' does not exist.")
    return path.read_text(encoding="utf-8").strip()


def _build_prompt(
    step: int,
    lime_step: Dict[str, Any],
    shap_step: Dict[str, Any],
    *,
    max_features: Optional[int],
    extra_context: Optional[str],
) -> str:
    lime_rudder = _trim_attributions(
        lime_step["rudder"]["feature_attributions"],
        weight_key="weight",
        max_features=max_features,
    )
    lime_throttle = _trim_attributions(
        lime_step["throttle"]["feature_attributions"],
        weight_key="weight",
        max_features=max_features,
    )
    shap_rudder = _trim_attributions(
        shap_step["rudder"]["feature_attributions"],
        weight_key="shap_value",
        max_features=max_features,
    )
    shap_throttle = _trim_attributions(
        shap_step["throttle"]["feature_attributions"],
        weight_key="shap_value",
        max_features=max_features,
    )
    context_payload = {
        "step": step,
        "rudder": {
            "lime": {
                "prediction": lime_step["rudder"].get("prediction"),
                "rudder_cmd": lime_step["rudder"].get("rudder_cmd"),
                "helm_label": lime_step["rudder"].get("helm_label"),
                "actual_rudder": lime_step["rudder"].get("actual_rudder"),
                "feature_attributions": lime_rudder,
            },
            "shap": {
                "prediction": shap_step["rudder"].get("prediction"),
                "rudder_cmd": shap_step["rudder"].get("rudder_cmd"),
                "helm_label": shap_step["rudder"].get("helm_label"),
                "actual_rudder": shap_step["rudder"].get("actual_rudder"),
                "expected_value": shap_step["rudder"].get("expected_value"),
                "feature_attributions": shap_rudder,
            },
        },
        "throttle": {
            "lime": {
                "prediction": lime_step["throttle"].get("prediction"),
                "probabilities": lime_step["throttle"].get("probabilities"),
                "feature_attributions": lime_throttle,
            },
            "shap": {
                "prediction": shap_step["throttle"].get("prediction"),
                "raw_output": shap_step["throttle"].get("raw_output"),
                "expected_value": shap_step["throttle"].get("expected_value"),
                "feature_attributions": shap_throttle,
            },
        },
    }
    context_json = json.dumps(context_payload, indent=2)
    extra = f"\n\nAdditional context:\n{extra_context}" if extra_context else ""
    return (
        "Explain the following LIME and SHAP attributions for a single step of the NEAT controller."
        " Reference the feature values and weights to justify the action choice."
        f"{extra}\n\nStep data:\n{context_json}"
    )


def _call_llm(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response format: {data}") from exc


def explain_steps(
    *,
    lime_summary: Path,
    shap_summary: Path,
    output_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_features: Optional[int],
    start_step: Optional[int],
    end_step: Optional[int],
    max_steps: Optional[int],
    timeout: int,
    sleep_seconds: float,
    context_file: Optional[Path],
) -> None:
    lime_data = _load_json(lime_summary)
    shap_data = _load_json(shap_summary)
    extra_context = _read_context_file(context_file)
    merged = _merge_steps(
        lime_data,
        shap_data,
        start_step=start_step,
        end_step=end_step,
        max_steps=max_steps,
    )
    if not merged:
        raise RuntimeError("No overlapping steps found between LIME and SHAP summaries.")
    results = []
    for step, lime_step, shap_step in merged:
        prompt = _build_prompt(
            step,
            lime_step,
            shap_step,
            max_features=max_features,
            extra_context=extra_context,
        )
        explanation = _call_llm(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            timeout=timeout,
        )
        result = {
            "step": step,
            "llm_explanation": explanation,
            "prompt": prompt,
        }
        results.append(result)
        step_path = output_dir / f"llm_step_{step:03d}.json"
        _write_json(step_path, result)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    summary_path = output_dir / "llm_summary.json"
    _write_json(summary_path, results)


def main() -> None:
    args = _build_parser().parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Provide --api-key or set {args.api_key_env}."
        )
    explain_steps(
        lime_summary=args.lime_summary,
        shap_summary=args.shap_summary,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        max_features=args.max_features,
        start_step=args.start_step,
        end_step=args.end_step,
        max_steps=args.max_steps,
        timeout=args.request_timeout,
        sleep_seconds=args.sleep_seconds,
        context_file=args.context_file,
    )


if __name__ == "__main__":
    main()
