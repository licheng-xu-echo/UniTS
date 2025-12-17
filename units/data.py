import torch,warnings,glob,os,itertools,logging
import numpy as np
from scipy.spatial import KDTree
from torch_geometric.data import InMemoryDataset,Data
from rdkit import Chem
from torch_geometric.data.separate import separate
from sklearn.utils import shuffle
from ase.io import read as ase_read
from scipy.spatial.distance import cdist
from openbabel import pybel
from openbabel import openbabel as ob
from scipy.spatial.transform import Rotation
from rdkit.Chem import rdmolops
from itertools import combinations
from molop import AutoParser
from copy import deepcopy

warnings.filterwarnings("ignore")
pt = Chem.GetPeriodicTable()

NUM_ATOM_TYPE = 65
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 7           # -3,-2,-1,0,1,2,3
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3
NUM_RADICAL_TYPES = 5           
MAX_REACTIVE_ATOMS = 20


NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_FEAT_DIMS = [NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,
                    NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE]

ATOM_LST = ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,Chem.rdchem.HybridizationType.SP3D2,Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC,
                 Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.UNSPECIFIED,
                 ]
NUM_BOND_TYPE = len(BOND_TYPE_LST)
BOND_DIR_LST = [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT,
                Chem.rdchem.BondDir.BEGINDASH,
                Chem.rdchem.BondDir.BEGINWEDGE,
                Chem.rdchem.BondDir.EITHERDOUBLE]
NUM_BOND_DIRECTION = len(BOND_DIR_LST)
BOND_STEREO_LST = [Chem.rdchem.BondStereo.STEREONONE,
                   Chem.rdchem.BondStereo.STEREOE,
                   Chem.rdchem.BondStereo.STEREOZ,
                   Chem.rdchem.BondStereo.STEREOANY,
                   Chem.rdchem.BondStereo.STEREOATROPCW,
                   Chem.rdchem.BondStereo.STEREOATROPCCW,
                   ]
NUM_BOND_STEREO = len(BOND_STEREO_LST)
BOND_FEAT_DIME = [NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R","S","None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}

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

def reorder_atoms_preserve_all_properties(mol, new_order):
    """
    重新排列原子顺序，同时保留所有分子属性
    
    参数:
    mol: 原始Mol对象
    new_order: 新的原子顺序列表
    
    返回:
    重新排序后的新Mol对象，原子映射字典
    """
    # 创建新的可写分子
    new_mol = Chem.RWMol()
    
    # 存储原子映射 (旧索引 -> 新索引)
    atom_mapping = {}
    
    # 1. 按照新顺序添加原子，并复制所有原子属性
    for new_idx, old_idx in enumerate(new_order):
        old_atom = mol.GetAtomWithIdx(old_idx)
        #new_atom = copy.deepcopy(old_atom)
        new_idx_in_mol = new_mol.AddAtom(old_atom)
        atom_mapping[old_idx] = new_idx_in_mol
    
    # 2. 添加键，并复制所有键属性
    for bond in mol.GetBonds():
        begin_old = bond.GetBeginAtomIdx()
        end_old = bond.GetEndAtomIdx()
        begin_new = atom_mapping[begin_old]
        end_new = atom_mapping[end_old]
        
        # 创建新的键对象并复制所有属性
        new_bond = new_mol.GetBondBetweenAtoms(begin_new, end_new)
        if new_bond is None:
            new_bond = new_mol.AddBond(begin_new, end_new, bond.GetBondType())
            new_bond = new_mol.GetBondWithIdx(new_mol.GetNumBonds()-1)
        
        # 复制键属性
        new_bond.SetBondType(bond.GetBondType())
        new_bond.SetIsAromatic(bond.GetIsAromatic())
        new_bond.SetIsConjugated(bond.GetIsConjugated())
        new_bond.SetStereo(bond.GetStereo())
        new_bond.SetBondDir(bond.GetBondDir())
        
        # 复制键的立体化学信息
        for prop_name in bond.GetPropNames():
            prop_value = bond.GetProp(prop_name)
            new_bond.SetProp(prop_name, prop_value)
    
    # 3. 转换为完整的Mol对象
    new_mol = new_mol.GetMol()
    
    # 4. 复制构象信息（如果有）
    if mol.GetNumConformers() > 0:
        for conf in mol.GetConformers():
            new_conf = Chem.Conformer(new_mol.GetNumAtoms())
            conf_id = conf.GetId()
            new_conf.SetId(conf_id)
            
            # 复制每个原子的3D坐标
            for old_idx, new_idx in atom_mapping.items():
                pos = conf.GetAtomPosition(old_idx)
                new_conf.SetAtomPosition(new_idx, pos)
            
            # 复制构象属性
            for prop_name in conf.GetPropNames():
                prop_value = conf.GetProp(prop_name)
                new_conf.SetProp(prop_name, prop_value)
            
            new_mol.AddConformer(new_conf)
    
    # 5. 复制分子级别的属性
    for prop_name in mol.GetPropNames():
        prop_value = mol.GetProp(prop_name)
        new_mol.SetProp(prop_name, prop_value)
    
    '''# 6. 复制环信息
    if mol.GetRingInfo().IsInitialized():
        new_mol.GetRingInfo().Initialize()
        rings = mol.GetRingInfo().AtomRings()
        for ring in rings:
            new_ring = [atom_mapping[atom_idx] for atom_idx in ring]
            new_mol.GetRingInfo().AddRing(new_ring)'''
    
    # 7. 复制手性信息
    try:
        # 尝试复制手性标签
        Chem.AssignStereochemistry(new_mol, cleanIt=False, force=False)
    except:
        pass
    
    # 8. 复制其他分子属性
    try:
        new_mol.SetNumExplicitHs(mol.GetNumExplicitHs())
    except:
        pass
    
    try:
        new_mol.SetNumImplicitHs(mol.GetNumImplicitHs())
    except:
        pass
    
    return new_mol, atom_mapping

def update_mol_info(init_mol):
    mol_blks = Chem.GetMolFrags(init_mol, asMols=True, sanitizeFrags=False)
    mol_idxs = Chem.GetMolFrags(init_mol, asMols=False)

    new_mol_ = Chem.RWMol()
    for idx,mol_ in enumerate(mol_blks):
        Chem.MolToXYZFile(mol_,f"tmp_mol_{idx}.xyz")
        mol_op = AutoParser(f"tmp_mol_{idx}.xyz")[-1][-1].rdmol
        if mol_op:
            new_mol_.InsertMol(Chem.RWMol(mol_op))
        else:
            new_mol_.InsertMol(Chem.RWMol(mol_))
    new_mol_ = new_mol_.GetMol()
    _ = np.concatenate(mol_idxs).tolist()
    new_mol,prop_dict = reorder_atoms_preserve_all_properties(new_mol_,[_.index(idx) for idx in range(len(_))])
    #get_split_mol_idx(new_mol,)
    return new_mol,prop_dict

def mol2graphinfo(mol,chrg,multi):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    """
    assert chrg in [-3,-2,-1,0,1,2,3] and multi in [1,2,3,4,5]
    atom_features_list = []
    atom_mass_list = []
    
    for atom in mol.GetAtoms():
        atom_feature = [ATOM_DICT.get(atom.GetSymbol(), ATOM_DICT["unk"]),
                        min(atom.GetDegree(),MAX_NEIGHBORS),
                        chrg+3,                                         # -3,-2,-1,0,1,2,3
                        HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
                        CHIRAL_TAG_DICT.get(atom.GetChiralTag(),2),
                        int(atom.GetIsAromatic()),
                        VALENCE_DICT.get(atom.GetTotalValence(), 6),
                        NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
                        RS_TAG_DICT.get(atom.GetPropsAsDict().get("_CIPCode", "None"), 2),
                        multi-1]
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list),dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))
    # bonds
    num_bond_features = 5   # bond type, bond direction, bond stereo, isinring, isconjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [BOND_TYPE_LST.index(bond.GetBondType()),
                            BOND_DIR_LST.index(bond.GetBondDir()),
                            BOND_STEREO_LST.index(bond.GetStereo()),
                            int(bond.IsInRing()),
                            int(bond.GetIsConjugated())]

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = np.array(edges_list).T
        
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)

    else:   # mol has no bonds
        edge_index = np.empty((2,0),dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x,edge_index,edge_attr,atom_mass

def generate_custom_list(intervals):
    result = []
    for idx, length in enumerate(intervals):
        result.extend([idx] * length)
    return result

class _Dataset(InMemoryDataset):
    # dropped, not used
    def __init__(self, root, name='dataset_0.npy',transform=None, pre_transform=None, train=True, multi_file=False):

        self.root = root
        self.name = name if name[-4:] != ".npy" else name[:-4]
        self.train = train
        self.multi_file = multi_file
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return self.name

    @property
    def processed_file_names(self):

        if self.multi_file:
            return [f'{self.name}_pure_feat_mol_multi.pt']
        else:
            return [f'{self.name}_pure_feat_mol.pt']


    def process(self):
        logging.info(f'[INFO] {self.name} Processing...')
        self.data_list = []
        if not self.multi_file:
            raw_data_arr = np.load(f"{self.root}/{self.name}.npy",allow_pickle=True)
        else:
            data_files = sorted(glob.glob(f"{self.root}/{self.name}_*.npy"),key=lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[-1]))
            raw_data_arr = []
            for data_file in data_files:
                logging.info(data_file)
                raw_data_arr += np.load(data_file,allow_pickle=True).tolist()
        for raw_data in raw_data_arr:
            sub_mol_inf_lst = raw_data[1:]
            mol_Atoms = []
            mol_Coords = []
            mol_Nodeattrs = []
            mol_Edge_indexs = []
            mol_Edge_attrs = []
            mol_Reactive_atoms = []
            edge_index_start = 0
            reactive_atoms_ = torch.zeros(MAX_REACTIVE_ATOMS,dtype=torch.long) - 1
            for sub_mol_inf in sub_mol_inf_lst:
                mol_atoms,mol_coords,mol_center,mol_pa1,mol_pa2,mol_pa3,node_attr,edge_index,edge_attr,atom_mass,reactive_atoms = sub_mol_inf
                mol_atnums = torch.tensor([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                node_attr_ = torch.concat([node_attr,mol_atnums.unsqueeze(-1)],dim=-1)
                mol_Atoms.append([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                mol_Coords.append(mol_coords)
                mol_Nodeattrs.append(node_attr_)
                mol_Edge_indexs.append(edge_index+edge_index_start)
                edge_index_start += len(mol_atnums)
                mol_Edge_attrs.append(edge_attr)
                mol_Reactive_atoms.append(reactive_atoms)
            sub_atom_num = torch.tensor([len(mol_Atoms)],dtype=torch.long)
            frame_batch = torch.tensor(generate_custom_list([len(coord) for coord in mol_Coords]),dtype=torch.long)
            mol_Atoms = torch.tensor(np.concatenate(mol_Atoms),dtype=torch.long)
            mol_Nodeattrs = torch.concatenate(mol_Nodeattrs).long()
            mol_Edge_indexs = torch.concatenate(mol_Edge_indexs,dim=1).long()
            mol_Edge_attrs = torch.concatenate(mol_Edge_attrs).long()
            mol_Reactive_atoms = torch.tensor(mol_Reactive_atoms,dtype=torch.long)
            mol_Coords = torch.tensor(np.concatenate(mol_Coords),dtype=torch.float32)
            assert len(mol_Reactive_atoms[0]) <= MAX_REACTIVE_ATOMS
            reactive_atoms_[:len(mol_Reactive_atoms[0])] = mol_Reactive_atoms[0]
            #x = torch.tensor([1]*len(mol_Atoms),dtype=torch.long)
            data = Data(x=mol_Nodeattrs,mol_atoms=mol_Atoms,
                        edge_index=mol_Edge_indexs,
                        edge_attr=mol_Edge_attrs,
                        mol_coords=mol_Coords,
                        frame_batch=frame_batch,
                        sub_atom_num=sub_atom_num,
                        reactive_atoms=reactive_atoms_.unsqueeze(0),)

            self.data_list.append(data)

        data, slices = self.collate(self.data_list)
        self.data = data
        self.slices = slices
        logging.info(f'[INFO] {len(self.data_list)} Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass

    def len(self):
        return len(self.slices["x"]) - 1
    
def generate_rc_geometry_indices(sample_indices, max_bond_num=25, max_angle_num=25):

    valid_indices = sample_indices[sample_indices >= 0].tolist()
    rc_bond_indices = torch.zeros((max_bond_num, 2), dtype=torch.long) - 1
    rc_angle_indices = torch.zeros((max_angle_num, 3), dtype=torch.long) - 1
    if len(valid_indices) < 2:
        return rc_bond_indices, rc_angle_indices

    bond_combinations = list(itertools.combinations(valid_indices, 2))
    
    if bond_combinations:
        rc_bond_indices_ = torch.tensor(bond_combinations)
        rc_bond_indices[:min(rc_bond_indices_.shape[0], max_bond_num)] = rc_bond_indices_[:min(rc_bond_indices_.shape[0], max_bond_num)]
    if len(valid_indices) >= 3:
        angle_combinations = list(itertools.combinations(valid_indices, 3))
        
        if angle_combinations:
            rc_angle_indices_ = torch.tensor(angle_combinations)
            rc_angle_indices[:min(rc_angle_indices_.shape[0], max_angle_num)] = rc_angle_indices_[:min(rc_angle_indices_.shape[0], max_angle_num)]
    

    return rc_bond_indices[:max_bond_num], rc_angle_indices[:max_angle_num]

class MultiDataset1x(InMemoryDataset):
    
    def __init__(self, root, name_regrex='dataset_0_*.npy',transform=None, 
                 pre_transform=None, train=True, contain_init_mol_coords=False, 
                 link_rc=False,iptgraph_type='rct+pdt'):

        self.root = root
        self.name_regrex = name_regrex
        self.raw_data_files = sorted(glob.glob(f"{root}/{self.name_regrex}"),key=lambda x:int(x.split('.')[-2].split('_')[-1]))
        logging.info(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.train = train
        self.contain_init_mol_coords = contain_init_mol_coords
        self.link_rc = link_rc
        self.iptgraph_type = iptgraph_type.lower()
        logging.info(f"[INFO] input graph type is {self.iptgraph_type}")
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = torch.load(processed_path)
            self.data_lst.append(data)
            self.slices_lst.append(slices)
            self.data_num_lst.append(self.data_num_lst[-1]+len(slices['x'])-1)
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]

    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):
        if not self.link_rc:
            return [f"{os.path.basename(file)[:-4]}_{self.iptgraph_type}_multi.pt" for file in self.raw_data_files]
        else:
            return [f"{os.path.basename(file)[:-4]}_{self.iptgraph_type}_link_rc_multi.pt" for file in self.raw_data_files]
            


    def process(self):

        for idx,raw_data_f in enumerate(self.raw_data_files):
            if os.path.exists(self.processed_paths[idx]):
                logging.info(f'[INFO] {self.processed_paths[idx]} already exists, skip it...')
                continue
            logging.info(f'[INFO] {raw_data_f} is processing...')
            data_list = []
            fn_set = []
            raw_data_arr = np.load(raw_data_f,allow_pickle=True)
            for raw_data in raw_data_arr:
                sub_mol_inf = raw_data
                mol_Atoms = []
                mol_Coords = []
                #init_mol_FPFH = []
                mol_Nodeattrs = []
                mol_Edge_indexs = []
                mol_Edge_attrs = []
                mol_Reactive_atoms = []
                edge_index_start = 0
                reactive_atoms_ = torch.zeros(MAX_REACTIVE_ATOMS,dtype=torch.long) - 1
                # for sub_mol_inf in sub_mol_inf_lst:

                mol_atoms,mol_coords,x_edge_index_attr,rdmol,blk_idxs,reactive_atoms,rct_x_edge_index_attr,\
                    rct_blk_idxs,pdt_x_edge_index_attr,pdt_blk_idxs = sub_mol_inf
                
 
                if self.iptgraph_type == 'ts':
                    node_attr,edge_index,edge_attr,atom_mass,new_edge_index,new_edge_attr = x_edge_index_attr
                elif self.iptgraph_type == 'rct':
                    node_attr,edge_index,edge_attr,atom_mass,new_edge_index,new_edge_attr = rct_x_edge_index_attr
                elif self.iptgraph_type == 'pdt':
                    node_attr,edge_index,edge_attr,atom_mass,new_edge_index,new_edge_attr = pdt_x_edge_index_attr
                elif self.iptgraph_type == 'rct+pdt':
                    rnode_attr,redge_index,redge_attr,ratom_mass,rnew_edge_index,rnew_edge_attr = rct_x_edge_index_attr
                    pnode_attr,pedge_index,pedge_attr,patom_mass,pnew_edge_index,pnew_edge_attr = pdt_x_edge_index_attr
                    node_attr = rnode_attr
                    edge_index = torch.cat([redge_index,pedge_index],dim=1)
                    edge_attr = torch.cat([redge_attr,pedge_attr],dim=0)
                    new_edge_index = torch.cat([rnew_edge_index,pnew_edge_index],dim=1)
                    new_edge_attr = torch.cat([rnew_edge_attr,pnew_edge_attr],dim=0)
                else:
                    raise ValueError(f'input graph type {self.iptgraph_type} is not supported')

                mol_atnums = torch.tensor([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                node_attr_ = torch.concat([node_attr,mol_atnums.unsqueeze(-1)],dim=-1)
                mol_Atoms.append([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                mol_Coords.append(mol_coords)
                mol_Nodeattrs.append(node_attr_)
                if not self.link_rc:
                    mol_Edge_indexs.append(edge_index+edge_index_start)
                    edge_index_start += len(mol_atnums)
                    mol_Edge_attrs.append(edge_attr)
                else:
                    mol_Edge_indexs.append(new_edge_index+edge_index_start)
                    edge_index_start += len(mol_atnums)
                    mol_Edge_attrs.append(new_edge_attr)
                mol_Reactive_atoms.append(reactive_atoms)
                
                sub_atom_num = torch.tensor([len(mol_Atoms)],dtype=torch.long)
                frame_batch = torch.tensor(generate_custom_list([len(coord) for coord in mol_Coords]),dtype=torch.long)
                mol_Atoms = torch.tensor(np.concatenate(mol_Atoms),dtype=torch.long)
                mol_Nodeattrs = torch.concatenate(mol_Nodeattrs).long()
                mol_Edge_indexs = torch.concatenate(mol_Edge_indexs,dim=1).long()
                mol_Edge_attrs = torch.concatenate(mol_Edge_attrs).long()
                mol_Reactive_atoms = torch.tensor(mol_Reactive_atoms,dtype=torch.long)
                mol_Coords = torch.tensor(np.concatenate(mol_Coords),dtype=torch.float32)
                assert len(mol_Reactive_atoms[0]) <= MAX_REACTIVE_ATOMS
                reactive_atoms_[:len(mol_Reactive_atoms[0])] = mol_Reactive_atoms[0]
                reactive_atoms_ = reactive_atoms_.unsqueeze(0)
                rc_bond_indices, rc_angle_indices = generate_rc_geometry_indices(reactive_atoms_)

                data = Data(x=mol_Nodeattrs,
                            mol_atoms=mol_Atoms,
                            edge_index=mol_Edge_indexs,
                            edge_attr=mol_Edge_attrs,
                            mol_coords=mol_Coords,
                            frame_batch=frame_batch,
                            sub_atom_num=sub_atom_num,
                            reactive_atoms=reactive_atoms_,
                            rc_bond_indices=rc_bond_indices.unsqueeze(0),
                            rc_angle_indices=rc_angle_indices.unsqueeze(0)
                            )
                
                data_list.append(data)
            data, slices = self.collate(data_list)
            logging.info(f'[INFO] {len(data_list)} data index {idx} is saving...')
            torch.save((data, slices), self.processed_paths[idx])

    def download(self):
        pass

    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += len(slices['x']) - 1
        return ct
    
    def get(self,idx):
        for blk_i,data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        self.data,self.slices = self.data_lst[blk_i],self.slices_lst[blk_i]
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        return data

class MultiDatasetV2(InMemoryDataset):
    
    def __init__(self, root, name_regrex='dataset_0_*.npy',transform=None, 
                 pre_transform=None, train=True, contain_init_mol_coords=False, 
                 link_rc=False,data_enhance=False):

        self.root = root
        self.name_regrex = name_regrex
        self.raw_data_files = sorted(glob.glob(f"{root}/{self.name_regrex}"),key=lambda x:int(x.split('.')[-2].split('_')[-1]))
        logging.info(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.train = train
        self.contain_init_mol_coords = contain_init_mol_coords
        self.link_rc = link_rc
        self.data_enhance = data_enhance
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = torch.load(processed_path)
            self.data_lst.append(data)
            self.slices_lst.append(slices)
            self.data_num_lst.append(self.data_num_lst[-1]+len(slices['x'])-1)
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]

    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):
        if not self.link_rc:
            if not self.data_enhance:
                return [f"{os.path.basename(file)[:-4]}_multi.pt" for file in self.raw_data_files]
            else:
                return [f"{os.path.basename(file)[:-4]}_dataenh_multi.pt" for file in self.raw_data_files]
        else:
            if not self.data_enhance:
                return [f"{os.path.basename(file)[:-4]}_link_rc_multi.pt" for file in self.raw_data_files]
            else:
                return [f"{os.path.basename(file)[:-4]}_link_rc_dataenh_multi.pt" for file in self.raw_data_files]


    def process(self):

        for idx,raw_data_f in enumerate(self.raw_data_files):
            if os.path.exists(self.processed_paths[idx]):
                logging.info(f'[INFO] {self.processed_paths[idx]} already exists, skip it...')
                continue
            logging.info(f'[INFO] {raw_data_f} is processing...')
            data_list = []
            fn_set = []
            raw_data_arr = np.load(raw_data_f,allow_pickle=True)
            for raw_data in raw_data_arr:
                sub_mol_inf = raw_data
                mol_Atoms = []
                mol_Coords = []
                #init_mol_FPFH = []
                mol_Nodeattrs = []
                mol_Edge_indexs = []
                mol_Edge_attrs = []
                mol_Reactive_atoms = []
                edge_index_start = 0
                reactive_atoms_ = torch.zeros(MAX_REACTIVE_ATOMS,dtype=torch.long) - 1
                # for sub_mol_inf in sub_mol_inf_lst:
                if not self.data_enhance:
                    mol_atoms,mol_coords,x_edge_index_attr,rdmol,blk_idxs,reactive_atoms = sub_mol_inf
                else:
                    mol_atoms,mol_coords,x_edge_index_attr,rdmol,blk_idxs,reactive_atoms,fn,frame_idx = sub_mol_inf
                    if not fn in fn_set:
                        fn_set.append(fn)

                node_attr,edge_index,edge_attr,atom_mass,new_edge_index,new_edge_attr = x_edge_index_attr

                mol_atnums = torch.tensor([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                node_attr_ = torch.concat([node_attr,mol_atnums.unsqueeze(-1)],dim=-1)
                mol_Atoms.append([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                mol_Coords.append(mol_coords)
                mol_Nodeattrs.append(node_attr_)
                if not self.link_rc:
                    mol_Edge_indexs.append(edge_index+edge_index_start)
                    edge_index_start += len(mol_atnums)
                    mol_Edge_attrs.append(edge_attr)
                else:
                    mol_Edge_indexs.append(new_edge_index+edge_index_start)
                    edge_index_start += len(mol_atnums)
                    mol_Edge_attrs.append(new_edge_attr)
                mol_Reactive_atoms.append(reactive_atoms)
                
                sub_atom_num = torch.tensor([len(mol_Atoms)],dtype=torch.long)
                frame_batch = torch.tensor(generate_custom_list([len(coord) for coord in mol_Coords]),dtype=torch.long)
                mol_Atoms = torch.tensor(np.concatenate(mol_Atoms),dtype=torch.long)
                mol_Nodeattrs = torch.concatenate(mol_Nodeattrs).long()
                mol_Edge_indexs = torch.concatenate(mol_Edge_indexs,dim=1).long()
                mol_Edge_attrs = torch.concatenate(mol_Edge_attrs).long()
                mol_Reactive_atoms = torch.tensor(mol_Reactive_atoms,dtype=torch.long)
                mol_Coords = torch.tensor(np.concatenate(mol_Coords),dtype=torch.float32)
                #init_mol_Coords = torch.tensor(np.concatenate(init_mol_Coords),dtype=torch.float32)
                #init_mol_FPFH = torch.tensor(np.concatenate(init_mol_FPFH),dtype=torch.float32)
                assert len(mol_Reactive_atoms[0]) <= MAX_REACTIVE_ATOMS
                reactive_atoms_[:len(mol_Reactive_atoms[0])] = mol_Reactive_atoms[0]
                reactive_atoms_ = reactive_atoms_.unsqueeze(0)
                rc_bond_indices, rc_angle_indices = generate_rc_geometry_indices(reactive_atoms_)
                #x = torch.tensor([1]*len(mol_Atoms),dtype=torch.long)
                if not self.data_enhance:
                    data = Data(x=mol_Nodeattrs,mol_atoms=mol_Atoms,
                                edge_index=mol_Edge_indexs,
                                edge_attr=mol_Edge_attrs,
                                mol_coords=mol_Coords,
                                frame_batch=frame_batch,
                                sub_atom_num=sub_atom_num,
                                reactive_atoms=reactive_atoms_,
                                rc_bond_indices=rc_bond_indices.unsqueeze(0),
                                rc_angle_indices=rc_angle_indices.unsqueeze(0)
                                )
                else:
                    data = Data(x=mol_Nodeattrs,mol_atoms=mol_Atoms,
                                edge_index=mol_Edge_indexs,
                                edge_attr=mol_Edge_attrs,
                                mol_coords=mol_Coords,
                                frame_batch=frame_batch,
                                sub_atom_num=sub_atom_num,
                                reactive_atoms=reactive_atoms_,
                                rc_bond_indices=rc_bond_indices.unsqueeze(0),
                                rc_angle_indices=rc_angle_indices.unsqueeze(0),
                                fn_id=torch.tensor([fn_set.index(fn)]),
                                frame_idx=torch.tensor([frame_idx])
                                )
                data_list.append(data)
            data, slices = self.collate(data_list)
            logging.info(f'[INFO] {len(data_list)} data index {idx} is saving...')
            torch.save((data, slices), self.processed_paths[idx])

    def download(self):
        pass

    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += len(slices['x']) - 1
        return ct
    
    def get(self,idx):
        for blk_i,data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        self.data,self.slices = self.data_lst[blk_i],self.slices_lst[blk_i]
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        return data

class MultiDataset(InMemoryDataset):
    def __init__(self, root, name_regrex='dataset_0_*.npy',transform=None, pre_transform=None, train=True, contain_init_mol_coords=False):

        self.root = root
        self.name_regrex = name_regrex
        self.raw_data_files = sorted(glob.glob(f"{root}/{self.name_regrex}"),key=lambda x:int(x.split('.')[-2].split('_')[-1]))
        logging.info(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.train = train
        self.contain_init_mol_coords = contain_init_mol_coords
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = torch.load(processed_path)
            self.data_lst.append(data)
            self.slices_lst.append(slices)
            self.data_num_lst.append(self.data_num_lst[-1]+len(slices['x'])-1)
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]

    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):

        return [f"{os.path.basename(file)[:-4]}_multi.pt" for file in self.raw_data_files]


    def process(self):

        for idx,raw_data_f in enumerate(self.raw_data_files):
            if os.path.exists(self.processed_paths[idx]):
                logging.info(f'[INFO] {self.processed_paths[idx]} already exists, skip it...')
                continue
            logging.info(f'[INFO] {raw_data_f} is processing...')
            data_list = []
            raw_data_arr = np.load(raw_data_f,allow_pickle=True)
            for raw_data in raw_data_arr:
                sub_mol_inf_lst = raw_data[1:]
                mol_Atoms = []
                mol_Coords = []
                init_mol_FPFH = []
                mol_Nodeattrs = []
                mol_Edge_indexs = []
                mol_Edge_attrs = []
                mol_Reactive_atoms = []
                init_mol_Coords = []
                edge_index_start = 0
                reactive_atoms_ = torch.zeros(MAX_REACTIVE_ATOMS,dtype=torch.long) - 1
                for sub_mol_inf in sub_mol_inf_lst:

                    mol_atoms,mol_coords,mol_center,mol_pa1,mol_pa2,mol_pa3,fpfh,init_coords,node_attr,edge_index,edge_attr,atom_mass,reactive_atoms,rdmol = sub_mol_inf
                    #fpfh = compute_fpfh(init_mol_coords)

                    mol_atnums = torch.tensor([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                    node_attr_ = torch.concat([node_attr,mol_atnums.unsqueeze(-1)],dim=-1)
                    mol_Atoms.append([pt.GetAtomicNumber(atom) if isinstance(atom, str) else atom for atom in mol_atoms])
                    mol_Coords.append(mol_coords)
                    init_mol_FPFH.append(fpfh)
                    init_mol_Coords.append(init_coords)
                    mol_Nodeattrs.append(node_attr_)
                    mol_Edge_indexs.append(edge_index+edge_index_start)
                    edge_index_start += len(mol_atnums)
                    mol_Edge_attrs.append(edge_attr)
                    mol_Reactive_atoms.append(reactive_atoms)
                sub_atom_num = torch.tensor([len(mol_Atoms)],dtype=torch.long)
                frame_batch = torch.tensor(generate_custom_list([len(coord) for coord in mol_Coords]),dtype=torch.long)
                mol_Atoms = torch.tensor(np.concatenate(mol_Atoms),dtype=torch.long)
                mol_Nodeattrs = torch.concatenate(mol_Nodeattrs).long()
                mol_Edge_indexs = torch.concatenate(mol_Edge_indexs,dim=1).long()
                mol_Edge_attrs = torch.concatenate(mol_Edge_attrs).long()
                mol_Reactive_atoms = torch.tensor(mol_Reactive_atoms,dtype=torch.long)
                mol_Coords = torch.tensor(np.concatenate(mol_Coords),dtype=torch.float32)
                init_mol_Coords = torch.tensor(np.concatenate(init_mol_Coords),dtype=torch.float32)
                init_mol_FPFH = torch.tensor(np.concatenate(init_mol_FPFH),dtype=torch.float32)
                assert len(mol_Reactive_atoms[0]) <= MAX_REACTIVE_ATOMS
                reactive_atoms_[:len(mol_Reactive_atoms[0])] = mol_Reactive_atoms[0]
                reactive_atoms_ = reactive_atoms_.unsqueeze(0)
                rc_bond_indices, rc_angle_indices = generate_rc_geometry_indices(reactive_atoms_) # V7 New
                #x = torch.tensor([1]*len(mol_Atoms),dtype=torch.long)
                data = Data(x=mol_Nodeattrs,mol_atoms=mol_Atoms,
                            edge_index=mol_Edge_indexs,
                            edge_attr=mol_Edge_attrs,
                            mol_coords=mol_Coords,
                            init_mol_fpfh=init_mol_FPFH,
                            frame_batch=frame_batch,
                            sub_atom_num=sub_atom_num,
                            reactive_atoms=reactive_atoms_,
                            init_mol_coords=init_mol_Coords,
                            rc_bond_indices=rc_bond_indices.unsqueeze(0),
                            rc_angle_indices=rc_angle_indices.unsqueeze(0)
                            )
                data_list.append(data)
            data, slices = self.collate(data_list)
            logging.info(f'[INFO] {len(data_list)} data index {idx} is saving...')
            torch.save((data, slices), self.processed_paths[idx])



    def download(self):
        pass

    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += len(slices['x']) - 1
        return ct
    
    def get(self,idx):
        for blk_i,data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        self.data,self.slices = self.data_lst[blk_i],self.slices_lst[blk_i]
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        return data
    
def get_idx_split(data_size, train_size, valid_size, seed):
    ids = shuffle(range(data_size), random_state=seed)
    if abs(train_size + valid_size - data_size) < 2:
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:])
        test_idx = val_idx
    else:    
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
    split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    return split_dict

def get_fnidx_split(fn_id, train_ratio, valid_ratio, seed):

    fn_id_set = torch.unique(fn_id)
    shuffle_fn_id_set = shuffle(fn_id_set, random_state=seed)
    train_size = int(len(shuffle_fn_id_set) * train_ratio)
    valid_size = int(len(shuffle_fn_id_set) * valid_ratio)
    if abs(train_size + valid_size - len(shuffle_fn_id_set)) < 2:
        train_fn_idx, val_fn_idx = torch.tensor(shuffle_fn_id_set[:train_size]), torch.tensor(shuffle_fn_id_set[train_size:])
        test_fn_idx = val_fn_idx
    else:    
        train_fn_idx, val_fn_idx, test_fn_idx = torch.tensor(shuffle_fn_id_set[:train_size]), torch.tensor(shuffle_fn_id_set[train_size:train_size + valid_size]), torch.tensor(shuffle_fn_id_set[train_size + valid_size:])
    
    train_idx = []
    val_idx = []
    test_idx = []

    for idx,id_ in enumerate(fn_id):
        if id_ in train_fn_idx:
            train_idx.append(idx)
        elif id_ in val_fn_idx:
            val_idx.append(idx)
        elif id_ in test_fn_idx:
            test_idx.append(idx)
    train_idx = shuffle(torch.tensor(train_idx), random_state=seed)
    val_idx = shuffle(torch.tensor(val_idx), random_state=seed)
    test_idx = shuffle(torch.tensor(test_idx), random_state=seed)    
    
    split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx, 'train_fn':train_fn_idx, 'valid_fn':val_fn_idx, 'test_fn':test_fn_idx}
    
    return split_dict

def rotate_molecule(coords, rotation_angles):

    center = np.mean(coords, axis=0)
    centered_coords = coords - center

    rotation = Rotation.from_euler('xyz', rotation_angles, degrees=True)
    rotated_coords = rotation.apply(centered_coords)
    
    return rotated_coords + center


def generate_rotations():
    rotations = []
    angles = [0, 60, 120, 180, 240, 300]
    
    # 生成一些有代表性的旋转组合
    for x_angle in angles:
        for y_angle in angles:
            for z_angle in angles:
                rotations.append([x_angle, y_angle, z_angle])
    
    return rotations

ROT_CAND = generate_rotations()

def generate_sphere_points(n, radius):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    
    for i in range(n):
        y = radius * (1.0 - (i / float(n - 1)) * 2.0)
        current_radius = np.sqrt(radius**2 - y**2)
        
        theta = phi * i
        x = np.cos(theta) * current_radius
        z = np.sin(theta) * current_radius
        
        points.append([x, y, z])
    
    return np.array(points)

def find_best_position(existing_coords, new_coords, min_distance):
    """Find best position for new_coords to avoid overlapping with existing_coords."""
    new_center = np.mean(new_coords, axis=0)
    
    radius = min_distance * 2 
    candidate_positions = generate_sphere_points(50, radius)
    rotation_combinations = ROT_CAND
    
    best_pos = None
    best_score = float('inf')
    best_rotation = None
    
    for rotation_angles in rotation_combinations:
        rotated_coords = rotate_molecule(new_coords, rotation_angles)
        rotated_center = np.mean(rotated_coords, axis=0)
        
        for candidate in candidate_positions:
            translated = rotated_coords - rotated_center + candidate
            
            dist_matrix = cdist(existing_coords, translated)
            min_dist = np.min(dist_matrix)
            if min_dist < min_distance:
                continue
            

            move_distance = np.linalg.norm(candidate - rotated_center)
            current_score = move_distance
            
            if current_score < best_score:
                best_score = current_score
                best_pos = candidate - rotated_center
                best_rotation = rotation_angles
    
    if best_pos is None:
        return find_best_position(existing_coords, new_coords, min_distance * 1.5)
    

    return best_pos, best_rotation

def merge_positions(coords_lst, min_distance=1.0):
    """merge multiple molecules and ensure no overlap"""
    combined_coords = coords_lst[0].copy()
    warn_tag = False
    
    for i in range(1, len(coords_lst)):
        coords = coords_lst[i]
        best_pos, rotation_angles = find_best_position(combined_coords, coords, min_distance)
        rotated_coords = rotate_molecule(coords, rotation_angles)
        translated_coords = rotated_coords + best_pos
        
        move_distance = np.linalg.norm(best_pos)
        if move_distance > 5:
            warn_tag = True
            print(f"Warning: Molecule {i+1} moved by {move_distance:.2f} units, just move 0.5 unit along x axis")
            combined_x_axis_max = combined_coords[:,0].max()
            rot_x_axis_min = rotated_coords[:,0].min()
            translated_coords = rotated_coords + np.array([combined_x_axis_max-rot_x_axis_min+0.5, 0, 0])

        
        combined_coords = np.vstack([combined_coords, translated_coords])
    
    return combined_coords, warn_tag

def optimize_structure(input_file, output_file, forcefield="uff", steps=500):

    mol = next(pybel.readfile(input_file.split('.')[-1], input_file))
    mol.localopt(forcefield=forcefield, steps=steps)
    mol.write(output_file.split('.')[-1], output_file, overwrite=True)

def get_reorder_mol(src_dir):
    split_mol_files = glob.glob(f"{src_dir}/*/*_openbabelopt.xyz")
    split_positions = []
    split_atom_idx_lst = []
    split_symbols = []
    for split_mol_file in split_mol_files:
        split_atoms = ase_read(split_mol_file)
        with open(split_mol_file,"r") as fr:
            lines = fr.readlines()
        split_atom_idx = list(eval(lines[1].strip()))
        split_positions.append(split_atoms.positions)
        split_atom_idx_lst.append(split_atom_idx)
        split_symbols.append(list(split_atoms.symbols))
    split_positions,merg_warn_tag = merge_positions(split_positions)
    split_positions = split_positions - np.mean(split_positions,axis=0)
    split_atom_idx_lst = np.concatenate(split_atom_idx_lst,axis=0)
    split_symbols = np.concatenate(split_symbols,axis=0)
    assert len(split_atom_idx_lst) == len(set(split_atom_idx_lst))
    reorder_symbols = np.zeros((len(split_atom_idx_lst),),dtype=np.dtype('U20'))
    reorder_positions = np.zeros((len(split_atom_idx_lst),3))
    for i,atom_idx in enumerate(split_atom_idx_lst):
        reorder_symbols[atom_idx] = split_symbols[i]
        reorder_positions[atom_idx] = split_positions[i]
    return reorder_symbols,reorder_positions

def run_md_conformational_search(input_file, output_file, 
                                 md_steps=5000, md_timestep=0.001, 
                                 md_temperature=300, sampling_interval=100,
                                 optimization_steps=2000, forcefield="uff"):
    """
    conformer search and optimization based on MD
    """
    ob_log_handler = ob.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)
    ob.obErrorLog.SetOutputLevel(0)
    
    conv = ob.OBConversion()
    mol = ob.OBMol()
    
    input_format = input_file.split('.')[-1]
    output_format = output_file.split('.')[-1]
    
    if not conv.SetInFormat(input_format):
        raise ValueError(f"Unsupported input format: {input_format}")
    if not conv.SetOutFormat(output_format):
        raise ValueError(f"Unsupported output format: {output_format}")
    
    if not conv.ReadFile(mol, input_file):
        raise IOError(f"Unable to read file: {input_file}")

    ff = ob.OBForceField.FindForceField(forcefield)
    if not ff:
        raise RuntimeError(f"Cannot find force field: {forcefield}")
    
    ff_setup = ff.Setup(mol)
    if not ff_setup:
        raise RuntimeError("Force field setting failed")
    
    ff.DynamicsInitialize(md_timestep, md_temperature)
    
    best_conformer = None
    lowest_energy = float('inf')
    
    for step in range(0, md_steps + 1, sampling_interval):
        ff.DynamicsRunNSteps(sampling_interval)

        current_energy = ff.Energy()
        

        if current_energy < lowest_energy:
            lowest_energy = current_energy
            best_conformer = mol.Copy()  
    

    if best_conformer:
        mol = best_conformer
        
    else:
        print("original structure would be used")
    

    ff = ob.OBForceField.FindForceField(forcefield)
    ff.Setup(mol)
    
    ff.SteepestDescentInitialize(optimization_steps)
    ff.SteepestDescentTakeNSteps(optimization_steps)
    
    final_energy = ff.Energy()
    conv.WriteFile(mol, output_file)

    return mol, final_energy

def compute_fpfh(points, k_neighbors=15, n_bins=11):
    N, _ = points.shape
    if N == 0:
        return np.array([])
    elif N == 1:
        return np.zeros(3 * n_bins).reshape(1, -1) + 1/(3*n_bins)
    elif N == 2:
        return np.zeros((2,3 * n_bins)) + 1/(3*n_bins)
    elif N == 3:
        return np.zeros((3,3 * n_bins)) + 1/(3*n_bins)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    if N == 1:
        return np.zeros(3 * n_bins).reshape(1, -1)
    
    tree = KDTree(points_centered)

    normals = np.zeros((N, 3))
    for i in range(N):
        actual_k = min(k_neighbors + 1, N)
        
        distances, indices = tree.query(points_centered[i], k=actual_k)
        
        if actual_k > 1:
            neighbors = points_centered[indices[1:]] 
        else:
            neighbors = np.empty((0, 3)) 
            
        if len(neighbors) < 2:

            normals[i] = np.random.randn(3)
            normals[i] /= np.linalg.norm(normals[i]) + 1e-8
            continue
        
        cov_matrix = np.cov(neighbors, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]
        
        centroid_neighbors = np.mean(neighbors, axis=0)
        if np.dot(normal, centroid_neighbors) < 0:
            normal = -normal
        normals[i] = normal
    
    spfh_all = np.zeros((N, 3 * n_bins))
    bin_edges = {
        'alpha': np.linspace(-1, 1, n_bins + 1), 
        'phi': np.linspace(-1, 1, n_bins + 1),
        'theta': np.linspace(-np.pi, np.pi, n_bins + 1) 
    }
    
    for i in range(N):
        actual_k = min(k_neighbors + 1, N)
        
        _, indices = tree.query(points_centered[i], k=actual_k)
        
        if actual_k > 1:
            neighbor_indices = indices[1:]
            neighbors = points_centered[neighbor_indices]
            neighbor_normals = normals[neighbor_indices]
        else:
            neighbors = np.empty((0, 3))
            neighbor_normals = np.empty((0, 3))

        hist_alpha = np.zeros(n_bins)
        hist_phi = np.zeros(n_bins)
        hist_theta = np.zeros(n_bins)
        
        for j, (q, n_q) in enumerate(zip(neighbors, neighbor_normals)):
            diff = q - points_centered[i]
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            u = diff / diff_norm

            n_p = normals[i]
            
            alpha = np.dot(n_p, n_q)
            
            phi = np.dot(n_p, u)
            
            v = np.cross(n_p, u)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v /= v_norm
                w = np.cross(n_p, v)
                theta = np.arctan2(np.dot(w, n_q), np.dot(n_p, n_q))
            else:
                theta = 0.0
        
            hist_alpha += np.histogram([alpha], bins=bin_edges['alpha'])[0]
            hist_phi += np.histogram([phi], bins=bin_edges['phi'])[0]
            hist_theta += np.histogram([theta], bins=bin_edges['theta'])[0]

        spfh_all[i] = np.concatenate([hist_alpha, hist_phi, hist_theta])

    fpfh_features = np.zeros((N, 3 * n_bins))
    for i in range(N):
        actual_k = min(k_neighbors + 1, N)

        _, indices = tree.query(points_centered[i], k=actual_k)
        
        if actual_k > 1:
            neighbor_indices = indices[1:]
        else:
            neighbor_indices = np.array([], dtype=int)
        
       
        fpfh = spfh_all[i].copy()
        if len(neighbor_indices) > 0:
            distances = np.linalg.norm(
                points_centered[neighbor_indices] - points_centered[i], 
                axis=1
            )
            weights = 1.0 / (distances + 1e-8)
            weights /= np.sum(weights) 
            
            neighbor_spfh = spfh_all[neighbor_indices]
            weighted_sum = np.sum(neighbor_spfh * weights[:, None], axis=0)
            fpfh += weighted_sum
    
        if np.sum(fpfh) > 0:
            fpfh /= np.sum(fpfh)
        fpfh_features[i] = fpfh
    
    return fpfh_features

def get_reorder_mol_inf(src_dir, verbose=False):
    split_mol_dirs = glob.glob(f"{src_dir}/*")
    split_mol_files = glob.glob(f"{src_dir}/*/*_openbabelopt.xyz")
    assert len(split_mol_dirs) == len(split_mol_files) and len(split_mol_dirs) > 0
    split_atom_idx_lst = []
    split_symbols = []
    split_fpfh = []
    split_positions = []
    for split_mol_file in split_mol_files:
            split_atoms = ase_read(split_mol_file)
            with open(split_mol_file,"r") as fr:
                lines = fr.readlines()
            tmp_positions = split_atoms.positions
            tmp_positions -= np.mean(tmp_positions,axis=0)
            tmp_atom_idx = list(eval(lines[1].strip()))
            tmp_fpfh = compute_fpfh(tmp_positions)
            split_atom_idx_lst.append(tmp_atom_idx)
            split_fpfh.append(tmp_fpfh)
            split_symbols.append(list(split_atoms.symbols))
            split_positions.append(tmp_positions)
    split_atom_idx_lst = np.concatenate(split_atom_idx_lst,axis=0)
    split_fpfh = np.concatenate(split_fpfh,axis=0)
    split_symbols = np.concatenate(split_symbols,axis=0)
    split_positions,merg_warn_tag = merge_positions(split_positions)
    split_positions = split_positions - np.mean(split_positions,axis=0)
    reorder_symbols = np.zeros((len(split_atom_idx_lst),),dtype=np.dtype('U20'))
    reorder_fpfh = np.zeros(split_fpfh.shape)
    reorder_positions = np.zeros((len(split_atom_idx_lst),3))
    for i,atom_idx in enumerate(split_atom_idx_lst):
        reorder_fpfh[atom_idx] = split_fpfh[i]
        reorder_symbols[atom_idx] = split_symbols[i]
        reorder_positions[atom_idx] = split_positions[i]
    if verbose and merg_warn_tag:
        print(f"[WARN] {src_dir} merge positions warning")
    return reorder_symbols,reorder_positions,reorder_fpfh

def add_reactat_edge_info(existed_edge_index,existed_edge_attr,reacting_atoms):
    react_atom_pairs = torch.tensor([pair for pair in combinations(reacting_atoms,2)])
    react_atom_pairs_ = [pair for pair in react_atom_pairs if not torch.all(pair == existed_edge_index.T,dim=1).any()]
    if len(react_atom_pairs_) == 0:
        return existed_edge_index,existed_edge_attr
    react_atom_pairs_new = torch.stack(react_atom_pairs_).T

    react_atom_edge_index = torch.concat([torch.stack([react_atom_pairs_new[1],react_atom_pairs_new[0]]),react_atom_pairs_new],dim=1)
    react_atom_edge_attr = torch.tensor([[BOND_TYPE_LST.index(Chem.rdchem.BondType.UNSPECIFIED),0,0,0,0] for _ in range(react_atom_edge_index.shape[1])])

    new_edge_index = torch.concat([existed_edge_index,react_atom_edge_index],dim=1)
    new_edge_attr = torch.concat([existed_edge_attr,react_atom_edge_attr],dim=0)
    return new_edge_index,new_edge_attr

def load_dataset_from_args(args,only_test=True,shuffle=True):
    if not hasattr(args,'dataset_type') or args.dataset_type == 1:
        dataset = MultiDataset(root=args.dataset_path,name_regrex=args.name_regrex)
        dataset_to_use_graph = deepcopy(dataset)
    elif args.dataset_type == 2:
        dataset = MultiDatasetV2(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=args.link_rc,
                                    data_enhance=args.data_enhance if hasattr(args,'data_enhance') else False)
        dataset_to_use_graph = MultiDatasetV2(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=False,
                                data_enhance=args.data_enhance if hasattr(args,'data_enhance') else False)
    elif args.dataset_type == 3:
        dataset = MultiDataset1x(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=args.link_rc,
                                    iptgraph_type=args.iptgraph_type)
        dataset_to_use_graph = MultiDataset1x(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=False,
                                iptgraph_type=args.iptgraph_type)
    sf_index = list(range(len(dataset)))
    np.random.seed(args.seed)
    if shuffle:
        np.random.shuffle(sf_index)
    if args.data_truncated > 0:
        dataset = dataset[sf_index[:args.data_truncated]]
        dataset_to_use_graph = dataset_to_use_graph[sf_index[:args.data_truncated]]
    else:
        dataset = dataset[sf_index]
        dataset_to_use_graph = dataset_to_use_graph[sf_index]

    if not args.specific_test:
        train_ratio = args.train_ratio if hasattr(args,'train_ratio') else 0.8
        valid_ratio = args.valid_ratio if hasattr(args,'valid_ratio') else 0.1
        if not hasattr(args,"data_enhance") or not args.data_enhance:
            split_ids_map = get_idx_split(len(dataset), 
                                            int(train_ratio*len(dataset)), 
                                            int(valid_ratio*len(dataset)), 
                                            args.seed)
        else:
            dataset[0]
            split_ids_map = get_fnidx_split(dataset.data.fn_id,
                                            train_ratio,
                                            valid_ratio,
                                            args.seed)
        test_dataset = dataset[split_ids_map['test']]
        test_dataset_to_use_graph = dataset_to_use_graph[split_ids_map['test']]
    else:
        assert args.test_name_regrex != ''
        print("[INFO] Splitting dataset into train, valid. Test is specified")
        assert abs(args.train_ratio + args.valid_ratio - 1.0) < 1e-6
        if not hasattr(args,'dataset_type') or args.dataset_type == 1:
            test_dataset = MultiDataset(root=args.dataset_path, name_regrex=args.test_name_regrex)
            test_dataset_to_use_graph = deepcopy(test_dataset)
        elif args.dataset_type == 2:
            test_dataset = MultiDatasetV2(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc)
            test_dataset_to_use_graph = MultiDatasetV2(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=False)
        elif args.dataset_type == 3:
            test_dataset = MultiDataset1x(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc, 
                                            iptgraph_type=args.iptgraph_type)
            test_dataset_to_use_graph = MultiDataset1x(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=False, 
                                        iptgraph_type=args.iptgraph_type)
    if only_test:
        return test_dataset, test_dataset_to_use_graph
    else:
        return dataset, dataset_to_use_graph