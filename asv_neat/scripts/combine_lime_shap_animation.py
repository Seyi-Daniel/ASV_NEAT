#!/usr/bin/env python3
"""Combine LIME and SHAP explanation outputs into a single animation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import imageio.v2 as imageio
from PIL import Image

FRAME_FILENAME_PATTERN = re.compile(r"frame_(\d+)\.png")
RUDDER_PLOT_PATTERN = re.compile(r"explanation_rudder_(\d+)\.png")
THROTTLE_PLOT_PATTERN = re.compile(r"explanation_throttle_(\d+)\.png")


def _find_matching_dir(base: Path, patterns: Sequence[re.Pattern[str]]) -> Path:
    """Return the first directory under ``base`` containing files for all patterns."""

    search_dirs: List[Path] = [base]
    search_dirs.extend([child for child in base.iterdir() if child.is_dir()])

    for candidate in search_dirs:
        if all(any(pattern.fullmatch(item.name) for item in candidate.iterdir() if item.is_file()) for pattern in patterns):
            return candidate
    pattern_descriptions = ", ".join(p.pattern for p in patterns)
    raise FileNotFoundError(f"Could not find directory under {base} with files matching: {pattern_descriptions}")


def _discover_frame_dir(base: Path) -> Path:
    return _find_matching_dir(base, [FRAME_FILENAME_PATTERN])


def _discover_plot_dir(base: Path) -> Path:
    return _find_matching_dir(base, [RUDDER_PLOT_PATTERN, THROTTLE_PLOT_PATTERN])


def _extract_steps(frame_dir: Path) -> List[int]:
    steps = []
    for frame_path in frame_dir.iterdir():
        if not frame_path.is_file():
            continue
        match = FRAME_FILENAME_PATTERN.fullmatch(frame_path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _load_and_resize_plots(plot_paths: Iterable[Path]) -> List[Image.Image]:
    images = [Image.open(path).convert("RGB") for path in plot_paths]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized = []
    for img in images:
        if img.size == (max_width, max_height):
            resized.append(img)
        else:
            resized.append(img.resize((max_width, max_height), Image.LANCZOS))
    return resized


def _compose_frame(
    scene_path: Path,
    lime_rudder_path: Path,
    lime_throttle_path: Path,
    shap_rudder_path: Path,
    shap_throttle_path: Path,
) -> Image.Image:
    scene = Image.open(scene_path).convert("RGB")
    plots = _load_and_resize_plots(
        [lime_rudder_path, shap_rudder_path, lime_throttle_path, shap_throttle_path]
    )

    plot_width, plot_height = plots[0].width, plots[0].height
    right_panel_width = plot_width * 2
    right_panel_height = plot_height * 2

    canvas_width = scene.width + right_panel_width
    canvas_height = max(scene.height, right_panel_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    left_y = (canvas_height - scene.height) // 2 if canvas_height > scene.height else 0
    canvas.paste(scene, (0, left_y))

    right_x = scene.width
    top_y = (canvas_height - right_panel_height) // 2 if canvas_height > right_panel_height else 0

    positions = [
        (right_x, top_y),
        (right_x + plot_width, top_y),
        (right_x, top_y + plot_height),
        (right_x + plot_width, top_y + plot_height),
    ]

    for plot, position in zip(plots, positions, strict=True):
        canvas.paste(plot, position)

    return canvas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lime-dir", type=Path, required=True, help="Path to LIME scenario output directory")
    parser.add_argument("--shap-dir", type=Path, required=True, help="Path to SHAP scenario output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined outputs")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the final GIF")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    lime_dir: Path = args.lime_dir
    shap_dir: Path = args.shap_dir
    output_dir: Path = args.output_dir
    fps: int = args.fps

    lime_frame_dir = _discover_frame_dir(lime_dir)
    lime_plot_dir = _discover_plot_dir(lime_dir)
    shap_frame_dir = _discover_frame_dir(shap_dir)
    shap_plot_dir = _discover_plot_dir(shap_dir)

    steps = _extract_steps(lime_frame_dir)
    if not steps:
        steps = _extract_steps(shap_frame_dir)
    if not steps:
        raise RuntimeError("No frame images found in either LIME or SHAP directories.")

    combined_dir = output_dir / "combined_frames"
    combined_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: List[Path] = []

    for step in steps:
        frame_name = f"frame_{step:03d}.png"
        rudder_name = f"explanation_rudder_{step:03d}.png"
        throttle_name = f"explanation_throttle_{step:03d}.png"

        scene_path = lime_frame_dir / frame_name
        if not scene_path.exists():
            scene_path = shap_frame_dir / frame_name
        lime_rudder_path = lime_plot_dir / rudder_name
        lime_throttle_path = lime_plot_dir / throttle_name
        shap_rudder_path = shap_plot_dir / rudder_name
        shap_throttle_path = shap_plot_dir / throttle_name

        required_paths = [
            scene_path,
            lime_rudder_path,
            lime_throttle_path,
            shap_rudder_path,
            shap_throttle_path,
        ]

        if not all(path.exists() for path in required_paths):
            continue

        composite = _compose_frame(
            scene_path,
            lime_rudder_path,
            lime_throttle_path,
            shap_rudder_path,
            shap_throttle_path,
        )
        output_path = combined_dir / f"combined_{step:03d}.png"
        composite.save(output_path)
        combined_frames.append(output_path)

    if not combined_frames:
        raise RuntimeError("No composite frames were generated; ensure required images exist.")

    images = [imageio.imread(frame_path) for frame_path in combined_frames]
    gif_path = output_dir / "lime_shap_explanation_animation.gif"
    imageio.mimsave(gif_path, images, fps=fps)

    print(f"Saved {len(combined_frames)} frames to {combined_dir}")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
