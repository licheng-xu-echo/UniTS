from rdkit import Chem
from rdkit.Chem import rdmolops
from .utils import create_reaction_with_atom_mapping
from qcbot.utils import symbol_pos_to_xyz_file,MolFormatConversion

def get_split_mol_idx(mol, pairs_to_remove):
    rw = Chem.RWMol(mol)

    for a, b in pairs_to_remove:
        a = int(a)
        b = int(b)
        if rw.GetBondBetweenAtoms(a, b):
            rw.RemoveBond(a, b)
    mol = rw.GetMol()
    split_mol_idx_lst = rdmolops.GetMolFrags(mol, asMols=False)
    split_mol_lst = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return mol,split_mol_idx_lst,split_mol_lst

def identify_reacting_atoms_by_vib_graph(atoms,coord,modes,ratio=0.5):
    coord_backward = coord - modes[0] * ratio
    coord_forward = coord + modes[0] * ratio
    symbol_pos_to_xyz_file(atoms,coord_backward,'tmp_back.xyz')
    MolFormatConversion("tmp_back.xyz","tmp_back.sdf")
    symbol_pos_to_xyz_file(atoms,coord_forward,'tmp_forw.xyz')
    MolFormatConversion("tmp_forw.xyz","tmp_forw.sdf")
    mol_back = Chem.MolFromMolFile('tmp_back.sdf',removeHs=False, sanitize=False)
    mol_forw = Chem.MolFromMolFile('tmp_forw.sdf',removeHs=False, sanitize=False)
    mol_back_bonum, mol_forw_bonum = mol_back.GetNumBonds(),mol_forw.GetNumBonds()
    Chem.rdmolops.AssignStereochemistryFrom3D(mol_back)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol_forw)
    if mol_back_bonum < mol_forw_bonum:
        molgraph = mol_back
    else:
        molgraph = mol_forw
    rxn = create_reaction_with_atom_mapping(mol_back,mol_forw)
    rxn.Initialize()
    rxn_rev = create_reaction_with_atom_mapping(mol_forw,mol_back)
    rxn_rev.Initialize()
    reacting_atoms = list(rxn.GetReactingAtoms()[0]) + list(rxn_rev.GetReactingAtoms()[0])
    reacting_atoms = list(set(reacting_atoms))
    return reacting_atoms,molgraph

def is_group_rot(atoms,coord,reacting_atoms):
    if len(reacting_atoms) == 0:
        return False, ''
    symbol_pos_to_xyz_file(atoms[reacting_atoms],coord[reacting_atoms],'tmp.xyz')
    MolFormatConversion("tmp.xyz","tmp.sdf")
    subgroup = Chem.MolFromMolFile("tmp.sdf",removeHs=False,sanitize=False)
    atom_symbol_set = sorted(list(set([atom.GetSymbol() for atom in subgroup.GetAtoms()])))
    #print(atom_symbol_set)
    vib_group_smi = Chem.MolToSmiles(subgroup)
    return atom_symbol_set==['C','H'] or atom_symbol_set==['H','O'] or atom_symbol_set==['H','N'], vib_group_smi

def identify_reacting_atoms_by_vib_graph_iter(atoms,coord,modes,freqs,file):
    reacting_atoms,molgraph = identify_reacting_atoms_by_vib_graph(atoms,coord,modes,ratio=0.5)
    if len(reacting_atoms) == 0:
        reacting_atoms,molgraph = identify_reacting_atoms_by_vib_graph(atoms,coord,modes,ratio=1.0)
        if len(reacting_atoms) == 0:
            reacting_atoms,molgraph = identify_reacting_atoms_by_vib_graph(atoms,coord,modes,ratio=1.5)
            if len(reacting_atoms) == 0:            
                return [],None
            else:
                is_me,vib_group_smi = is_group_rot(atoms,coord,reacting_atoms)
                print(reacting_atoms,is_me,freqs[0],vib_group_smi,file)
                if is_me:
                    return [],None
    return reacting_atoms,molgraph