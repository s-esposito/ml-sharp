"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
import torch.utils.data

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered videos.",
    required=True,
)
@click.option(
    "--fps",
    type=float,
    default=24.0,
    help="Frames per second for the output video.",
)
@click.option(
    "--frames-per-gaussian",
    type=int,
    default=1,
    help="Number of frames to hold each Gaussian file (for sequence rendering only).",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_cli(
    input_path: Path, output_path: Path, fps: float, frames_per_gaussian: int, verbose: bool
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    params = camera.TrajectoryParams()

    if input_path.suffix == ".ply":
        # Single file mode: render normally
        LOGGER.info("Rendering single file %s", input_path)
        gaussians, metadata = load_ply(input_path)
        render_gaussians(
            gaussians=gaussians,
            metadata=metadata,
            params=params,
            output_path=(output_path / input_path.stem).with_suffix(".mp4"),
            fps=fps,
        )
    elif input_path.is_dir():
        # Folder mode: render as sequence
        scene_paths = sorted(list(input_path.glob("*.ply")))
        if len(scene_paths) == 0:
            LOGGER.error("No PLY files found in directory %s", input_path)
            exit(1)
        
        LOGGER.info("Rendering %d Gaussian files as sequence from %s", len(scene_paths), input_path)
        render_gaussian_sequence(
            scene_paths=scene_paths,
            params=params,
            output_path=(output_path / input_path.name).with_suffix(".mp4"),
            fps=fps,
            frames_per_gaussian=frames_per_gaussian,
        )
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    fps: float = 24.0,
) -> None:
    """Render a single gaussian checkpoint file."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    device = torch.device("cuda")

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2., 0],
            [0, f_px, (height - 1) / 2., 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    video_writer = io.VideoWriter(output_path, fps=fps)

    for _, eye_position in enumerate(trajectory):
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
    video_writer.close()


def render_gaussian_sequence(
    scene_paths: list[Path],
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    fps: float = 24.0,
    frames_per_gaussian: int = 1,
) -> None:
    """Render a sequence of gaussian files, one per frame.
    
    Each frame uses a different Gaussian file while following the same camera trajectory.
    The camera trajectory completes one full loop every 48 frames and repeats as needed.
    With frames_per_gaussian > 1, each Gaussian is held for multiple frames while the camera
    continues to move, creating a slow-motion effect for scene changes.
    
    Args:
        scene_paths: List of paths to PLY files, sorted by name
        output_path: Path to save the output video
        params: Trajectory parameters
        fps: Frames per second for the output video
        frames_per_gaussian: Number of video frames to hold each Gaussian file
    """
    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    device = torch.device("cuda")
    
    # Load first Gaussian to get metadata and setup camera
    first_gaussians, first_metadata = load_ply(scene_paths[0])
    (width, height) = first_metadata.resolution_px
    f_px = first_metadata.focal_length_px
    
    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2., 0],
            [0, f_px, (height - 1) / 2., 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    
    # Create camera model and trajectory based on first Gaussian
    camera_model = camera.create_camera_model(
        first_gaussians, intrinsics, resolution_px=first_metadata.resolution_px
    )
    
    # Calculate total number of output frames
    num_gaussians = len(scene_paths)
    total_output_frames = num_gaussians * frames_per_gaussian
    
    # Camera trajectory completes a full loop every 48 frames
    trajectory_loop_frames = 48
    
    trajectory_params = camera.TrajectoryParams(
        type=params.type,
        lookat_mode=params.lookat_mode,
        max_disparity=params.max_disparity,
        max_zoom=params.max_zoom,
        distance_m=params.distance_m,
        num_steps=trajectory_loop_frames,  # Camera completes one loop in 48 frames
        num_repeats=params.num_repeats,
    )
    
    trajectory = camera.create_eye_trajectory(
        first_gaussians, trajectory_params, resolution_px=first_metadata.resolution_px, f_px=f_px
    )
    
    # Extend trajectory to cover all output frames by looping
    full_trajectory = []
    for i in range(total_output_frames):
        trajectory_idx = i % len(trajectory)
        full_trajectory.append(trajectory[trajectory_idx])
    
    # Setup renderer and video writer
    renderer = gsplat.GSplatRenderer(color_space=first_metadata.color_space)
    video_writer = io.VideoWriter(output_path, fps=fps)
    
    # Render each frame with corresponding Gaussian file
    # Each Gaussian is held for frames_per_gaussian frames while camera continues moving
    current_gaussian = None
    current_gaussian_path = None
    
    for frame_idx, eye_position in enumerate(full_trajectory):
        # Determine which Gaussian to use for this frame
        gaussian_idx = frame_idx // frames_per_gaussian
        scene_path = scene_paths[gaussian_idx]
        
        # Load new Gaussian only when it changes
        if scene_path != current_gaussian_path:
            # Free previous Gaussian
            if current_gaussian is not None:
                del current_gaussian
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            if frame_idx % 10 == 0 or scene_path != current_gaussian_path:
                LOGGER.info(
                    "Rendering frames %d-%d/%d with %s",
                    frame_idx + 1,
                    min(frame_idx + frames_per_gaussian, total_output_frames),
                    total_output_frames,
                    scene_path.name,
                )
            
            # Load new Gaussian file
            current_gaussian, metadata = load_ply(scene_path)
            current_gaussian_path = scene_path
        
        gaussians = current_gaussian
        
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
        
        # Free rendering outputs after each frame (but keep Gaussian loaded)
        del rendering_output
        del color
        del depth
    
    # Clean up final Gaussian
    if current_gaussian is not None:
        del current_gaussian
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    video_writer.close()
    LOGGER.info("Finished rendering %d frames to %s", total_output_frames, output_path)
