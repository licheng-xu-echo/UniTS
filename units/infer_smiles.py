import argparse
import tempfile
from pathlib import Path

import torch
from molop.config import molopconfig
from qcbot.utils import MolFormatConversion, multi_symbol_pos_to_xyz_file, symbol_pos_to_xyz_file
from rdkit import Chem
from torch_geometric.data import DataLoader

from units.data import gen_dataset_from_smiles
from units.generate import load_model
pt = Chem.GetPeriodicTable()


def parse_reactive_atom_idx(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("reactive atom indices must not be empty")
    try:
        return [int(idx.strip()) for idx in value.split(",") if idx.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "reactive atom indices must be a comma-separated list of integers"
        ) from exc


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("must be True/False")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate TS initial guesses from SMILES and reactive atom indices."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        nargs="+",
        required=True,
        help="One or more reaction SMILES.",
    )
    parser.add_argument(
        "--reactive_atom_idx",
        type=parse_reactive_atom_idx,
        nargs="+",
        required=True,
        help="One or more comma-separated reactive atom index lists, e.g. 5,12 3,8",
    )
    parser.add_argument(
        "--charge",
        type=int,
        nargs="+",
        default=[0],
        help="One or more total charges. A single value will be broadcast.",
    )
    parser.add_argument(
        "--multi",
        type=int,
        nargs="+",
        default=[1],
        help="One or more spin multiplicities. A single value will be broadcast.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_path/units_hiegnn",
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="best_full_model.pth",
        help="Checkpoint filename inside model_path.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of stochastic initial guesses to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ts_initial_guess",
        help="Directory for generated xyz files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="gen",
        help="Prefix for generated xyz filenames.",
    )
    parser.add_argument(
        "--save_combined_xyz",
        type=str_to_bool,
        default=True,
        help="Whether to save all generated guesses into one multi-frame xyz file.",
    )
    parser.add_argument(
        "--save_full_trajectory",
        type=str_to_bool,
        default=False,
        help="Whether to save the full generation trajectory for each sample as xyz and sdf.",
    )
    parser.add_argument(
        "--fix_noise",
        type=str_to_bool,
        default=False,
        help="Use the same initial diffusion noise across the batch.",
    )
    parser.add_argument(
        "--resample",
        type=str_to_bool,
        default=False,
        help="Enable the optional resampling loop during sampling.",
    )
    parser.add_argument(
        "--resample_steps",
        type=int,
        default=10,
        help="Number of resampling steps when resample=True.",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=40,
        help="Start reverse-step threshold for resampling.",
    )
    parser.add_argument(
        "--jump_len",
        type=int,
        default=2,
        help="Jump length for resampling.",
    )
    return parser


def sample_to_symbols_and_coords(mol_atoms_row, node_mask_row, pred_pos_row):
    node_mask = node_mask_row.squeeze(-1).bool().cpu()
    atom_numbers = mol_atoms_row.squeeze(-1).cpu()[node_mask].tolist()
    symbols = [pt.GetElementSymbol(int(atom_number)) for atom_number in atom_numbers]
    coords = pred_pos_row[node_mask.cpu().numpy()]
    return symbols, coords


def extract_trajectory_frames(symbols, node_mask_row, x_traj, batch_idx, title_prefix):
    valid_mask = node_mask_row.squeeze(-1).bool().cpu().numpy()
    traj_frames = []
    for frame_idx, frame in enumerate(x_traj):
        coords = frame[batch_idx][valid_mask]
        traj_frames.append((symbols, coords, f"{title_prefix}_frame_{frame_idx:04d}"))
    return traj_frames


def normalize_per_reaction_arg(values, num_reactions, arg_name):
    if len(values) == 1:
        return values * num_reactions
    if len(values) != num_reactions:
        raise ValueError(
            f"{arg_name} must have length 1 or match the number of SMILES "
            f"({num_reactions}), but got {len(values)}"
        )
    return values


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if cli_args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    molopconfig.quiet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args, model = load_model(cli_args.model_path, ckpt_file=cli_args.ckpt_file, device=device)

    num_reactions = len(cli_args.smiles)
    reactive_atom_idx_lst = normalize_per_reaction_arg(
        cli_args.reactive_atom_idx, num_reactions, "reactive_atom_idx"
    )
    charge_lst = normalize_per_reaction_arg(cli_args.charge, num_reactions, "charge")
    multi_lst = normalize_per_reaction_arg(cli_args.multi, num_reactions, "multi")

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_xyz_frames = []
    with torch.no_grad():
        for reaction_idx, (smiles, reactive_atom_idx, charge, multi) in enumerate(
            zip(cli_args.smiles, reactive_atom_idx_lst, charge_lst, multi_lst)
        ):
            smiles_react_atom_index_lst = [
                [smiles, reactive_atom_idx] for _ in range(cli_args.num_samples)
            ]
            reaction_output_dir = output_dir / f"reaction_{reaction_idx:03d}"
            reaction_output_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix=f"units_smiles_{reaction_idx:03d}_") as temp_root:
                dataset = gen_dataset_from_smiles(
                    None,
                    smiles_react_atom_index_lst=smiles_react_atom_index_lst,
                    args=args,
                    charge=charge,
                    multi=multi,
                    tag=f"infer_{reaction_idx:03d}",
                    root=temp_root,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=min(cli_args.batch_size, len(dataset)),
                    shuffle=False,
                    num_workers=args.num_workers,
                )

                sample_idx = 0
                for data in dataloader:
                    data = data.to(device)
                    x_traj, mol_atoms, node_mask = model.sample_traj(
                        data,
                        fix_noise=cli_args.fix_noise,
                        resample=cli_args.resample,
                        resample_steps=cli_args.resample_steps,
                        start_step=cli_args.start_step,
                        jump_len=cli_args.jump_len,
                    )
                    pred_final = x_traj[-1]

                    for batch_idx, pred_pos in enumerate(pred_final):
                        symbols, coords = sample_to_symbols_and_coords(
                            mol_atoms[batch_idx], node_mask[batch_idx], pred_pos
                        )

                        xyz_path = reaction_output_dir / f"{cli_args.output_prefix}_{sample_idx}.xyz"
                        symbol_pos_to_xyz_file(symbols, coords, str(xyz_path))
                        all_xyz_frames.append(
                            (symbols, coords, f"reaction_{reaction_idx:03d}_{cli_args.output_prefix}_{sample_idx}")
                        )
                        print(f"[INFO] saved {xyz_path}")

                        if cli_args.save_full_trajectory:
                            traj_title_prefix = (
                                f"reaction_{reaction_idx:03d}_{cli_args.output_prefix}_{sample_idx}"
                            )
                            traj_frames = extract_trajectory_frames(
                                symbols,
                                node_mask[batch_idx],
                                x_traj,
                                batch_idx,
                                traj_title_prefix,
                            )
                            traj_xyz_path = (
                                reaction_output_dir
                                / f"{cli_args.output_prefix}_{sample_idx}_traj.xyz"
                            )
                            traj_sdf_path = (
                                reaction_output_dir
                                / f"{cli_args.output_prefix}_{sample_idx}_traj.sdf"
                            )
                            multi_symbol_pos_to_xyz_file(traj_frames, str(traj_xyz_path))
                            MolFormatConversion(
                                str(traj_xyz_path),
                                str(traj_sdf_path),
                                input_format="xyz",
                                output_format="sdf",
                            )
                            print(f"[INFO] saved {traj_xyz_path}")
                            print(f"[INFO] saved {traj_sdf_path}")
                        sample_idx += 1

            if cli_args.save_combined_xyz:
                combined_xyz_path = reaction_output_dir / f"{cli_args.output_prefix}_all.xyz"
                reaction_frames = [
                    frame for frame in all_xyz_frames
                    if frame[2].startswith(f"reaction_{reaction_idx:03d}_")
                ]
                multi_symbol_pos_to_xyz_file(reaction_frames, str(combined_xyz_path))
                print(f"[INFO] saved {combined_xyz_path}")

    if cli_args.save_combined_xyz and num_reactions > 1 and all_xyz_frames:
        combined_xyz_path = output_dir / f"{cli_args.output_prefix}_all.xyz"
        multi_symbol_pos_to_xyz_file(all_xyz_frames, str(combined_xyz_path))
        print(f"[INFO] saved {combined_xyz_path}")


if __name__ == "__main__":
    main()
