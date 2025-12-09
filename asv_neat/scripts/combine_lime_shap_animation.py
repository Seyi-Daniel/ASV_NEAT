#!/usr/bin/env python3
"""Combine LIME and SHAP explanation outputs into a single animation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
from PIL import Image

FRAME_FILENAME_PATTERN = re.compile(r"frame[_-]?(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
RUDDER_PLOT_PATTERN = re.compile(r"explanation_rudder_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
THROTTLE_PLOT_PATTERN = re.compile(r"explanation_throttle_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
def _index_files(base: Path, pattern: re.Pattern[str]) -> Dict[int, Path]:
    matches: Dict[int, Path] = {}
    for candidate in base.rglob("*"):
        if not candidate.is_file():
            continue
        match = pattern.fullmatch(candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        matches.setdefault(step, candidate)
    return matches


def _resize_plot_to_height(path: Path, target_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_height = max(1, target_height)
    target_width = max(1, int(image.width * target_height / max(image.height, 1)))
    return image.resize((target_width, target_height), Image.LANCZOS)


def _pad_plot_width(image: Image.Image, target_width: int) -> Image.Image:
    if image.width >= target_width:
        return image
    canvas = Image.new("RGB", (target_width, image.height), color=(255, 255, 255))
    offset_x = (target_width - image.width) // 2
    canvas.paste(image, (offset_x, 0))
    return canvas


def _compose_frame(
    scene_path: Path,
    lime_rudder_path: Path,
    lime_throttle_path: Path,
    shap_rudder_path: Path,
    shap_throttle_path: Path,
) -> Image.Image:
    scene = Image.open(scene_path).convert("RGB")
    top_height = scene.height // 2
    bottom_height = scene.height - top_height

    lime_rudder = _resize_plot_to_height(lime_rudder_path, top_height)
    shap_rudder = _resize_plot_to_height(shap_rudder_path, top_height)
    lime_throttle = _resize_plot_to_height(lime_throttle_path, bottom_height)
    shap_throttle = _resize_plot_to_height(shap_throttle_path, bottom_height)

    left_width = max(lime_rudder.width, lime_throttle.width)
    right_width = max(shap_rudder.width, shap_throttle.width)

    lime_rudder = _pad_plot_width(lime_rudder, left_width)
    lime_throttle = _pad_plot_width(lime_throttle, left_width)
    shap_rudder = _pad_plot_width(shap_rudder, right_width)
    shap_throttle = _pad_plot_width(shap_throttle, right_width)

    grid_width = left_width + right_width
    grid_height = top_height + bottom_height

    margin = 10
    canvas_width = scene.width + margin + grid_width
    canvas_height = max(scene.height, grid_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    scene_y = (canvas_height - scene.height) // 2 if canvas_height > scene.height else 0
    canvas.paste(scene, (0, scene_y))

    grid_x = scene.width + margin
    grid_y = (canvas_height - grid_height) // 2 if canvas_height > grid_height else 0

    canvas.paste(lime_rudder, (grid_x, grid_y))
    canvas.paste(shap_rudder, (grid_x + left_width, grid_y))
    canvas.paste(lime_throttle, (grid_x, grid_y + top_height))
    canvas.paste(shap_throttle, (grid_x + left_width, grid_y + top_height))

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

    lime_frames = _index_files(lime_dir, FRAME_FILENAME_PATTERN)
    shap_frames = _index_files(shap_dir, FRAME_FILENAME_PATTERN)
    lime_rudder_plots = _index_files(lime_dir, RUDDER_PLOT_PATTERN)
    lime_throttle_plots = _index_files(lime_dir, THROTTLE_PLOT_PATTERN)
    shap_rudder_plots = _index_files(shap_dir, RUDDER_PLOT_PATTERN)
    shap_throttle_plots = _index_files(shap_dir, THROTTLE_PLOT_PATTERN)

    print(
        "Discovered assets: "
        f"lime frames={len(lime_frames)}, shap frames={len(shap_frames)}, "
        f"lime plots (rudder/throttle)={len(lime_rudder_plots)}/{len(lime_throttle_plots)}, "
        f"shap plots (rudder/throttle)={len(shap_rudder_plots)}/{len(shap_throttle_plots)}"
    )

    steps = sorted(
        set(lime_frames)
        | set(shap_frames)
        | set(lime_rudder_plots)
        | set(lime_throttle_plots)
        | set(shap_rudder_plots)
        | set(shap_throttle_plots)
    )
    if not steps:
        raise RuntimeError(
            "No frame or plot images found; verify the provided directories contain outputs."
        )

    combined_dir = output_dir / "combined_frames"
    combined_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: List[Path] = []

    for step in steps:
        scene_path = lime_frames.get(step) or shap_frames.get(step)
        lime_rudder_path = lime_rudder_plots.get(step)
        lime_throttle_path = lime_throttle_plots.get(step)
        shap_rudder_path = shap_rudder_plots.get(step)
        shap_throttle_path = shap_throttle_plots.get(step)

        if not scene_path:
            continue
        if None in (
            lime_rudder_path,
            lime_throttle_path,
            shap_rudder_path,
            shap_throttle_path,
        ):
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
        raise RuntimeError("No combined frames were generated; ensure required frames and plots exist.")

    images = [imageio.imread(frame_path) for frame_path in combined_frames]
    gif_path = output_dir / "lime_shap_explanation_animation.gif"
    imageio.mimsave(gif_path, images, fps=fps)

    print(f"Saved {len(combined_frames)} frames to {combined_dir}")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
