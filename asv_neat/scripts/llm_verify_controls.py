#!/usr/bin/env python3
"""Verify NEAT control outputs by re-querying an LLM with summary inputs."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous surface vessel controller. "
    "Given normalized input features from a NEAT controller step, infer the likely "
    "rudder command and throttle action. Negative rudder_cmd means starboard, positive "
    "means port, and values are in [-1, 1]. Throttle is a discrete action where "
    "0=maintain speed, 1=accelerate, 2=decelerate. "
    "Respond ONLY with a JSON object: {\"rudder_cmd\": <float>, \"throttle\": <int>, "
    "\"reason\": <short string>} and no extra text."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to a LIME or SHAP summary JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "llm_control_verification.json",
        help="Output JSON file for LLM verification results.",
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
        "--rudder-tolerance",
        type=float,
        default=0.1,
        help="Max absolute difference to consider rudder commands matching.",
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


def _extract_features(step: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for section in ("rudder", "throttle"):
        for item in step.get(section, {}).get("feature_attributions", []):
            name = item.get("feature")
            if not name or name in features:
                continue
            try:
                features[name] = float(item.get("value", 0.0))
            except (TypeError, ValueError):
                continue
    return features


def _select_steps(
    data: List[Dict[str, Any]],
    *,
    start_step: Optional[int],
    end_step: Optional[int],
    max_steps: Optional[int],
) -> List[Dict[str, Any]]:
    steps = list(data)
    if start_step is not None:
        steps = [item for item in steps if int(item.get("step", -1)) >= start_step]
    if end_step is not None:
        steps = [item for item in steps if int(item.get("step", -1)) <= end_step]
    if max_steps is not None:
        steps = steps[:max_steps]
    return steps


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


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        snippet = match.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = raw[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _build_prompt(step_id: int, features: Dict[str, float]) -> str:
    features_json = json.dumps(features, indent=2, sort_keys=True)
    return (
        "Given the following normalized input features from a single NEAT controller step, "
        "infer the rudder command and throttle action.\n\n"
        f"Step: {step_id}\n\n"
        f"Input features (feature: value):\n{features_json}"
    )


def verify_controls(
    *,
    summary_path: Path,
    output_path: Path,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    start_step: Optional[int],
    end_step: Optional[int],
    max_steps: Optional[int],
    rudder_tolerance: float,
    timeout: int,
    sleep_seconds: float,
) -> None:
    data = _load_json(summary_path)
    steps = _select_steps(
        data,
        start_step=start_step,
        end_step=end_step,
        max_steps=max_steps,
    )
    results = []
    matched = 0
    for item in steps:
        step_id = int(item.get("step", -1))
        features = _extract_features(item)
        prompt = _build_prompt(step_id, features)
        llm_raw = None
        llm_payload = None
        error = None
        try:
            llm_raw = _call_llm(
                api_url=api_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                timeout=timeout,
            )
            llm_payload = _extract_json_object(llm_raw)
        except RuntimeError as exc:
            error = str(exc)

        model_rudder = float(item.get("rudder", {}).get("rudder_cmd", 0.0))
        model_throttle = int(item.get("throttle", {}).get("prediction", 0))
        helm_label = item.get("rudder", {}).get("helm_label")

        llm_rudder = None
        llm_throttle = None
        llm_reason = None
        if llm_payload:
            try:
                llm_rudder = float(llm_payload.get("rudder_cmd"))
            except (TypeError, ValueError):
                llm_rudder = None
            try:
                llm_throttle = int(llm_payload.get("throttle"))
            except (TypeError, ValueError):
                llm_throttle = None
            llm_reason = llm_payload.get("reason")

        rudder_diff = None
        rudder_match = False
        if llm_rudder is not None:
            rudder_diff = abs(model_rudder - llm_rudder)
            rudder_match = rudder_diff <= rudder_tolerance

        throttle_match = llm_throttle is not None and llm_throttle == model_throttle
        overall_match = rudder_match and throttle_match
        if overall_match:
            matched += 1

        results.append(
            {
                "step": step_id,
                "features": features,
                "model": {
                    "rudder_cmd": model_rudder,
                    "helm_label": helm_label,
                    "throttle": model_throttle,
                },
                "llm": {
                    "rudder_cmd": llm_rudder,
                    "throttle": llm_throttle,
                    "reason": llm_reason,
                    "raw_response": llm_raw,
                },
                "comparison": {
                    "rudder_diff": rudder_diff,
                    "rudder_within_tolerance": rudder_match,
                    "throttle_match": throttle_match,
                    "overall_match": overall_match,
                },
                "error": error,
            }
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    summary = {
        "source_summary": summary_path.as_posix(),
        "model": model,
        "api_url": api_url,
        "total_steps": len(results),
        "matched_steps": matched,
        "match_rate": (matched / len(results)) if results else 0.0,
        "rudder_tolerance": rudder_tolerance,
    }
    _write_json(output_path, {"summary": summary, "steps": results})


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Provide --api-key or set {args.api_key_env}."
        )
    verify_controls(
        summary_path=args.summary,
        output_path=args.output,
        api_url=args.api_url,
        api_key=api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        start_step=args.start_step,
        end_step=args.end_step,
        max_steps=args.max_steps,
        rudder_tolerance=args.rudder_tolerance,
        timeout=args.request_timeout,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
