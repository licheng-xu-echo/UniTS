import argparse
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from molop.config import molopconfig
from openbabel import openbabel as ob
from qcbot.utils import MolFormatConversion, multi_symbol_pos_to_xyz_file, symbol_pos_to_xyz_file
from rdkit import Chem
from torch_geometric.data import DataLoader

from units.data import (
    MultiDataset,
    MultiDataset1x,
    MultiDatasetV2,
    add_reactat_edge_info,
    mol2graphinfo,
    update_mol_info,
)
from units.generate import load_model
from units.infer_smiles import (
    extract_trajectory_frames,
    normalize_per_reaction_arg,
    parse_reactive_atom_idx,
    resolve_model_path,
    sample_to_symbols_and_coords,
    str_to_bool,
)
from units.utils import seed_worker, set_global_seed


def xyz2mol(xyz_path):
    xyz_path = Path(xyz_path)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    xyz_to_obmol = ob.OBConversion()
    if not xyz_to_obmol.SetInFormat("xyz"):
        raise ValueError("Open Babel does not support xyz input format")

    obmol = ob.OBMol()
    if not xyz_to_obmol.ReadFile(obmol, str(xyz_path)):
        raise ValueError(f"Failed to read xyz file with Open Babel: {xyz_path}")
    if obmol.NumAtoms() == 0:
        raise ValueError(f"No atoms found in xyz file: {xyz_path}")

    ob_to_sdf = ob.OBConversion()
    if not ob_to_sdf.SetOutFormat("sdf"):
        raise ValueError("Open Babel does not support sdf output format")
    mol_block = ob_to_sdf.WriteString(obmol)

    mol = Chem.MolFromMolBlock(
        mol_block,
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError(f"Failed to convert xyz file to an RDKit molecule: {xyz_path}")
    if mol.GetNumAtoms() != obmol.NumAtoms():
        raise ValueError(
            "Atom count mismatch after xyz conversion: "
            f"{xyz_path} (Open Babel={obmol.NumAtoms()}, RDKit={mol.GetNumAtoms()})"
        )
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    new_mol,prop_dict = update_mol_info(mol)
    #mol.UpdatePropertyCache(strict=False)
    return new_mol


def gen_dataset_from_mols(mol_react_atom_index_lst, args=None, charge=0, multi=1, tag="temp", root=".", ts_type="units"):
    ts_dataset = []
    for mol, reacting_atoms in mol_react_atom_index_lst:
        mol.UpdatePropertyCache(strict=False)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        x_edge_index_attr = list(mol2graphinfo(mol, chrg=charge, multi=multi))
        existed_edge_index = deepcopy(x_edge_index_attr[1])
        existed_edge_attr = deepcopy(x_edge_index_attr[2])
        if ts_type == "units":
            new_edge_index, new_edge_attr = torch.empty([2, 0]), torch.empty([0, 5])
        elif ts_type == "da":
            new_edge_index, new_edge_attr = add_reactat_edge_info(
                existed_edge_index, existed_edge_attr, reacting_atoms
            )
        else:
            raise ValueError("ts_type should be units or da")
        x_edge_index_attr.append(new_edge_index)
        x_edge_index_attr.append(new_edge_attr)
        blk_idxs = Chem.GetMolFrags(mol)
        ts_dataset.append(
            [atoms, np.random.rand(len(atoms), 3), x_edge_index_attr, mol, blk_idxs, reacting_atoms]
        )

    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, f"{tag}_ts_0.npy"), np.array(ts_dataset, dtype=object))
    processed_dir = os.path.join(root, "processed")
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    if not hasattr(args, "dataset_type") or args.dataset_type == 1:
        dataset = MultiDataset(root=root, name_regrex=f"{tag}_ts_0.npy")
    elif args.dataset_type == 2:
        dataset = MultiDatasetV2(
            root=root,
            name_regrex=f"{tag}_ts_0.npy",
            link_rc=args.link_rc,
            data_enhance=args.data_enhance if hasattr(args, "data_enhance") else False,
        )
    elif args.dataset_type == 3:
        dataset = MultiDataset1x(
            root=root,
            name_regrex=f"{tag}_ts_0.npy",
            link_rc=args.link_rc,
            iptgraph_type=args.iptgraph_type,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")
    return dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate TS initial guesses from xyz files and reactive atom indices."
    )
    parser.add_argument(
        "--xyz",
        type=str,
        nargs="+",
        required=True,
        help="One or more xyz files containing the input structures.",
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
        "--model_type",
        type=str,
        default="units_hiegnn",
        help="Bundled model ID under units/model_path, e.g. units_hiegnn.",
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
        "--seed",
        type=int,
        default=None,
        help="Global random seed for reproducible inference. By default no fixed seed is set.",
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


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if cli_args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cli_args.seed is not None:
        if cli_args.seed < 0 or cli_args.seed > 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        set_global_seed(cli_args.seed)
        print(f"[INFO] random seed = {cli_args.seed}")

    molopconfig.quiet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = resolve_model_path(cli_args.model_type)
    print(f"[INFO] model_type = {cli_args.model_type}")
    print(f"[INFO] model_path = {model_path}")
    args, model = load_model(str(model_path), ckpt_file=cli_args.ckpt_file, device=device)

    num_reactions = len(cli_args.xyz)
    reactive_atom_idx_lst = normalize_per_reaction_arg(
        cli_args.reactive_atom_idx, num_reactions, "reactive_atom_idx"
    )
    charge_lst = normalize_per_reaction_arg(cli_args.charge, num_reactions, "charge")
    multi_lst = normalize_per_reaction_arg(cli_args.multi, num_reactions, "multi")

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_xyz_frames = []
    with torch.no_grad():
        for reaction_idx, (xyz_path, reactive_atom_idx, charge, multi) in enumerate(
            zip(cli_args.xyz, reactive_atom_idx_lst, charge_lst, multi_lst)
        ):
            xyz_path = Path(xyz_path).expanduser().resolve()
            mol = xyz2mol(xyz_path)
            num_atoms = mol.GetNumAtoms()
            for atom_idx in reactive_atom_idx:
                if atom_idx < 0 or atom_idx >= num_atoms:
                    raise ValueError(
                        f"reactive atom index {atom_idx} is out of range for {xyz_path} "
                        f"(num_atoms={num_atoms})"
                    )
            mol_react_atom_index_lst = [
                [Chem.Mol(mol), list(reactive_atom_idx)] for _ in range(cli_args.num_samples)
            ]

            reaction_output_dir = output_dir / f"reaction_{reaction_idx:03d}"
            reaction_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] xyz = {xyz_path}")
            print(f"[INFO] reactive_atom_idx = {reactive_atom_idx}")

            with tempfile.TemporaryDirectory(prefix=f"units_xyz_{reaction_idx:03d}_") as temp_root:
                dataset = gen_dataset_from_mols(
                    mol_react_atom_index_lst=mol_react_atom_index_lst,
                    args=args,
                    charge=charge,
                    multi=multi,
                    tag=f"infer_{reaction_idx:03d}",
                    root=temp_root,
                    ts_type=cli_args.model_type.split("_")[0],
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=min(cli_args.batch_size, len(dataset)),
                    shuffle=False,
                    num_workers=args.num_workers,
                    worker_init_fn=seed_worker if cli_args.seed is not None else None,
                    generator=(
                        torch.Generator().manual_seed(cli_args.seed)
                        if cli_args.seed is not None
                        else None
                    ),
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

                        xyz_output_path = (
                            reaction_output_dir / f"{cli_args.output_prefix}_{sample_idx}.xyz"
                        )
                        symbol_pos_to_xyz_file(symbols, coords, str(xyz_output_path))
                        all_xyz_frames.append(
                            (
                                symbols,
                                coords,
                                f"reaction_{reaction_idx:03d}_{cli_args.output_prefix}_{sample_idx}",
                            )
                        )
                        print(f"[INFO] saved {xyz_output_path}")

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
