import argparse
from pathlib import Path

from units.visualize import load_xyz_trajectory, visualize_and_animate_trajectory


BOND_CONTROL_PARAMS = {
    "covalent_factor": 1.2,
    "force_bond": [],
    "force_nobond": [],
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a multi-frame xyz trajectory to a GIF."
    )
    parser.add_argument(
        "--xyz",
        type=str,
        required=True,
        help="Input xyz trajectory file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output gif path. Default: use the xyz stem in the same directory.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=0.0,
        help="Elevation angle for the 3D view.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=0.0,
        help="Azimuth angle for the 3D view.",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll angle for the 3D view.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output gif.",
    )
    parser.add_argument(
        "--selected_frames",
        type=int,
        default=1,
        help="Sample one frame every N frames from the trajectory.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.selected_frames <= 0:
        raise ValueError("selected_frames must be positive")

    xyz_path = Path(args.xyz).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else xyz_path.with_suffix(".gif")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = load_xyz_trajectory(str(xyz_path))
    frame_indices = list(range(0, len(trajectory), args.selected_frames))

    visualize_and_animate_trajectory(
        xyz_file_path=str(xyz_path),
        output_gif_path=str(output_path),
        bond_params=BOND_CONTROL_PARAMS,
        selected_frames=frame_indices,
        elev=args.elev,
        azim=args.azim,
        roll=args.roll,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
