import argparse
from pathlib import Path


def parse_freeze_bond(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("freeze bond must not be empty")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "freeze bond must be a comma-separated atom pair, e.g. 6,12"
        )
    try:
        atom_i, atom_j = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "freeze bond atoms must be integers"
        ) from exc
    if atom_i <= 0 or atom_j <= 0:
        raise argparse.ArgumentTypeError(
            "Gaussian atom indices are 1-based and must be positive"
        )
    return atom_i, atom_j


def parse_xyz(xyz_path):
    xyz_path = Path(xyz_path)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"xyz file not found: {xyz_path}")

    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"invalid xyz file: {xyz_path}")

    frame_atoms = []
    line_idx = 0
    total_lines = len(lines)

    while line_idx < total_lines:
        while line_idx < total_lines and not lines[line_idx].strip():
            line_idx += 1
        if line_idx >= total_lines:
            break

        try:
            atom_count = int(lines[line_idx].strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid atom count at line {line_idx + 1} in xyz file: {xyz_path}"
            ) from exc

        frame_start_line = line_idx + 1
        if atom_count <= 0:
            raise ValueError(
                f"invalid atom count at line {frame_start_line} in xyz file: {xyz_path}"
            )

        if line_idx + 2 + atom_count > total_lines:
            raise ValueError(
                f"incomplete xyz frame starting at line {frame_start_line} in {xyz_path}"
            )

        coord_lines = lines[line_idx + 2 : line_idx + 2 + atom_count]
        atoms = []
        for coord_offset, line in enumerate(coord_lines, start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"invalid xyz coordinate line {line_idx + 2 + coord_offset} in {xyz_path}"
                )
            symbol = parts[0]
            try:
                x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError as exc:
                raise ValueError(
                    f"invalid xyz coordinate line {line_idx + 2 + coord_offset} in {xyz_path}"
                ) from exc
            atoms.append((symbol, x, y, z))
        frame_atoms.append(atoms)
        line_idx = line_idx + 2 + atom_count

    if not frame_atoms:
        raise ValueError(f"no valid xyz frame found in {xyz_path}")
    return frame_atoms[-1]


def normalize_dispersion(method, empirical_dispersion):
    method_lower = method.lower()
    dispersion = empirical_dispersion.strip()
    if dispersion.lower() == "none":
        return None
    if dispersion == "GD3BJ" and method_lower != "b3lyp":
        raise ValueError(
            "EmpiricalDispersion=GD3BJ can only be used with b3lyp"
        )
    return dispersion


def build_method_keyword(method, basis, empirical_dispersion):
    keyword = f"{method}/{basis}"
    if empirical_dispersion is not None:
        keyword += f" EmpiricalDispersion={empirical_dispersion}"
    return keyword


def format_coordinates(atoms):
    return "\n".join(
        f"{symbol:<2} {x:16.6f} {y:16.6f} {z:16.6f}"
        for symbol, x, y, z in atoms
    )


def build_direct_ts_gjf(
    atoms,
    charge,
    multiplicity,
    nproc,
    mem,
    method_keyword,
    title,
):
    coord_block = format_coordinates(atoms)
    return (
        f"%nproc={nproc}\n"
        f"%mem={mem}\n"
        f"# p opt(TS,calcfc,noeigen) freq {method_keyword} geom=PrintInputOrient\n\n"
        f"{title}\n\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n\n"
    )


def build_two_step_ts_gjf(
    atoms,
    charge,
    multiplicity,
    nproc,
    mem,
    method_keyword,
    title,
    chk_name,
    freeze_bonds,
):
    coord_block = format_coordinates(atoms)
    freeze_block = "\n".join(
        f"B {atom_i} {atom_j} F" for atom_i, atom_j in freeze_bonds
    )
    return (
        f"%chk={chk_name}\n"
        f"%nproc={nproc}\n"
        f"%mem={mem}\n"
        f"# p opt=modredundant freq {method_keyword} geom=PrintInputOrient\n\n"
        f"{title}\n\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n\n"
        f"{freeze_block}\n\n"
        "--link1--\n"
        f"%oldchk={chk_name}\n"
        f"%nproc={nproc}\n"
        f"%mem={mem}\n"
        f"# p opt=(ts,readfc,noeigen,nofreeze,notrust,maxcycle=250) freq "
        f"{method_keyword} geom=(allcheck,PrintInputOrient) guess=tcheck\n\n"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert an xyz file to a Gaussian input file."
    )
    parser.add_argument(
        "--xyz",
        type=str,
        required=True,
        help="Input xyz file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output gjf file path. Default: use the xyz stem in the same directory.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="direct_ts",
        choices=["direct_ts", "two_step_ts"],
        help="Gaussian task template.",
    )
    parser.add_argument(
        "--freeze_bond",
        type=parse_freeze_bond,
        nargs="*",
        default=None,
        help="Frozen bonds for two_step_ts, e.g. --freeze_bond 6,12 5,18",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=16,
        help="Number of CPU cores.",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="32GB",
        help="Gaussian memory setting.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="b3lyp",
        help="Electronic structure method.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="def2svp",
        help="Basis set.",
    )
    parser.add_argument(
        "--empirical_dispersion",
        type=str,
        default="GD3BJ",
        help="Empirical dispersion keyword value. Use 'none' to disable.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Generated by UniTS",
        help="Gaussian title line.",
    )
    parser.add_argument(
        "--chk_name",
        type=str,
        default=None,
        help="Checkpoint filename for two_step_ts. Default: <output_stem>_modts.chk",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.nproc <= 0:
        raise ValueError("nproc must be positive")
    if args.multiplicity <= 0:
        raise ValueError("multiplicity must be positive")

    xyz_path = Path(args.xyz).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else xyz_path.with_suffix(".gjf")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atoms = parse_xyz(xyz_path)
    empirical_dispersion = normalize_dispersion(args.method, args.empirical_dispersion)
    method_keyword = build_method_keyword(args.method, args.basis, empirical_dispersion)

    if args.task_type == "direct_ts":
        content = build_direct_ts_gjf(
            atoms=atoms,
            charge=args.charge,
            multiplicity=args.multiplicity,
            nproc=args.nproc,
            mem=args.mem,
            method_keyword=method_keyword,
            title=args.title,
        )
    else:
        if not args.freeze_bond:
            raise ValueError(
                "two_step_ts requires at least one --freeze_bond, e.g. --freeze_bond 6,12 5,18"
            )
        chk_name = (
            args.chk_name
            if args.chk_name is not None
            else f"{output_path.stem}.chk"
        )
        content = build_two_step_ts_gjf(
            atoms=atoms,
            charge=args.charge,
            multiplicity=args.multiplicity,
            nproc=args.nproc,
            mem=args.mem,
            method_keyword=method_keyword,
            title=args.title,
            chk_name=chk_name,
            freeze_bonds=args.freeze_bond,
        )

    output_path.write_text(content)
    print(f"[INFO] saved {output_path}")


if __name__ == "__main__":
    main()
