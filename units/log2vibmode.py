import argparse
from pathlib import Path

import numpy as np
from molop import AutoParser
from molop.config import molopconfig
from qcbot.utils import multi_symbol_pos_to_xyz_file
from rdkit import Chem

from units.visualize import visualize_and_animate_trajectory

pt = Chem.GetPeriodicTable()



def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse a Gaussian log, verify TS, and generate vibrational xyz/gif."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Input Gaussian log file.",
    )
    parser.add_argument(
        "--xyz_output",
        type=str,
        default=None,
        help="Output vibrational xyz trajectory file. Default: <log_stem>_vib.xyz",
    )
    parser.add_argument(
        "--gif_output",
        type=str,
        default=None,
        help="Output vibrational gif file. Default: <log_stem>_vib.gif",
    )
    parser.add_argument(
        "--mode_index",
        type=int,
        default=0,
        help="Vibrational mode index to visualize. Default: 0.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames in the vibrational xyz trajectory.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Mode scaling factor applied to coords +/- amplitude * mode.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=0.0,
        help="Elevation angle for gif rendering.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=0.0,
        help="Azimuth angle for gif rendering.",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll angle for gif rendering.",
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
        help="Sample one frame every N frames when rendering the gif.",
    )
    parser.add_argument(
        "--covalent-factor",
        type=float,
        default=1.2,
        help="Covalent radius scaling factor used for bond detection in gif rendering.",
    )
    return parser


def parse_gaussian_log(log_path):
    molopconfig.quiet()
    opmol = AutoParser(str(log_path))
    last_frame = opmol[-1][-1]

    atoms = np.asarray(last_frame.atoms)
    symbols = [pt.GetElementSymbol(int(atom)) for atom in atoms]
    coords = np.asarray(last_frame.coords.m, dtype=float)
    frequencies = np.asarray(last_frame.vibrations.frequencies.m, dtype=float)
    vibration_modes = np.asarray(
        [np.asarray(mode.m, dtype=float) for mode in last_frame.vibrations.vibration_modes]
    )

    return {
        "is_ts": bool(last_frame.is_TS),
        "symbols": symbols,
        "coords": coords,
        "frequencies": frequencies,
        "modes": vibration_modes,
        "charge": getattr(last_frame, "charge", None),
        "multiplicity": getattr(last_frame, "multiplicity", None),
    }


def build_vibration_frames(coords, mode, num_frames, amplitude):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    center = np.asarray(coords, dtype=float)
    mode = np.asarray(mode, dtype=float)
    start = center - amplitude * mode
    end = center + amplitude * mode

    if num_frames == 1:
        return np.asarray([center])

    first_count = num_frames // 2
    second_count = num_frames - first_count

    first_half = np.linspace(start, center, num=first_count, endpoint=False)
    second_half = np.linspace(center, end, num=second_count, endpoint=True)
    return np.concatenate([first_half, second_half], axis=0)


def build_xyz_frames(symbols, frame_coords, title_prefix):
    return [
        (symbols, coords, f"{title_prefix}_frame_{frame_idx:04d}")
        for frame_idx, coords in enumerate(frame_coords)
    ]


def build_pingpong_indices(num_frames, stride):
    frame_indices = list(range(0, num_frames, stride))
    if frame_indices[-1] != num_frames - 1:
        frame_indices.append(num_frames - 1)
    if len(frame_indices) == 1:
        return frame_indices
    return frame_indices + frame_indices[-2:0:-1]


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode_index < 0:
        raise ValueError("mode_index must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.selected_frames <= 0:
        raise ValueError("selected_frames must be positive")
    if args.covalent_factor <= 0:
        raise ValueError("covalent_factor must be positive")

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"log file not found: {log_path}")

    xyz_output = (
        Path(args.xyz_output).expanduser().resolve()
        if args.xyz_output is not None
        else log_path.with_name(f"{log_path.stem}_vib.xyz")
    )
    gif_output = (
        Path(args.gif_output).expanduser().resolve()
        if args.gif_output is not None
        else log_path.with_name(f"{log_path.stem}_vib.gif")
    )
    xyz_output.parent.mkdir(parents=True, exist_ok=True)
    gif_output.parent.mkdir(parents=True, exist_ok=True)

    parsed = parse_gaussian_log(log_path)
    frequencies = parsed["frequencies"]
    modes = parsed["modes"]

    print(f"[INFO] log = {log_path}")
    print(f"[INFO] is transition state = {parsed['is_ts']}")
    print(f"[INFO] charge = {parsed['charge']}, multiplicity = {parsed['multiplicity']}")
    print(f"[INFO] num_atoms = {len(parsed['symbols'])}")
    print(f"[INFO] num_frequencies = {len(frequencies)}")

    if not parsed["is_ts"]:
        raise ValueError(f"optimized structure in {log_path} is not a transition state")
    if args.mode_index >= len(frequencies):
        raise ValueError(
            f"mode_index {args.mode_index} is out of range, available modes: 0-{len(frequencies) - 1}"
        )

    mode = modes[args.mode_index]
    frequency = frequencies[args.mode_index]
    print(f"[INFO] selected mode = {args.mode_index}, frequency = {frequency:.4f} cm^-1")

    frame_coords = build_vibration_frames(
        coords=parsed["coords"],
        mode=mode,
        num_frames=args.num_frames,
        amplitude=args.amplitude,
    )
    xyz_frames = build_xyz_frames(parsed["symbols"], frame_coords, log_path.stem)
    multi_symbol_pos_to_xyz_file(xyz_frames, str(xyz_output))
    print(f"[INFO] saved {xyz_output}")

    pingpong_indices = build_pingpong_indices(len(frame_coords), args.selected_frames)
    bond_control_params = {
        "covalent_factor": args.covalent_factor,
        "force_bond": [],
        "force_nobond": [],
    }
    visualize_and_animate_trajectory(
        xyz_file_path=str(xyz_output),
        output_gif_path=str(gif_output),
        bond_params=bond_control_params,
        selected_frames=pingpong_indices,
        elev=args.elev,
        azim=args.azim,
        roll=args.roll,
        fps=args.fps,
    )
    print(f"[INFO] saved {gif_output}")


if __name__ == "__main__":
    main()
