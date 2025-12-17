import torch,os,logging,sys,argparse
from .optimizer import Muon
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import networkx as nx
from tqdm import tqdm
from .egnn.egnn_new import calc_angles
from collections import Counter
from torch_scatter import scatter_mean
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
pt = Chem.GetPeriodicTable()

class Args:
    def __init__(self,**entries):
        self.__dict__.update(entries)

def str_to_bool(value):
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("must be True/False")

def find_subgraphs(atoms, bonds):

    G = nx.Graph()
    
    G.add_nodes_from(atoms)
    
    G.add_edges_from(bonds)
    
    components = [sorted(c) for c in nx.connected_components(G)]
    
    components.sort(key=lambda x: x[0])
    return components

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def split_and_padding(coords, batch, size=3):

    num_molecules = torch.max(batch) + 1


    atom_counts = torch.bincount(batch)


    max_atoms = torch.max(atom_counts)


    padded_coords = torch.zeros((num_molecules, max_atoms, size))
    mask = torch.zeros((num_molecules, max_atoms), dtype=torch.bool)

    for mol_id in range(num_molecules):

        mol_coords = coords[batch == mol_id]

        padded_coords[mol_id, :len(mol_coords)] = mol_coords
        
        mask[mol_id, :len(mol_coords)] = True
    return padded_coords, mask

'''
def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask, remove_mean=True):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    if remove_mean:
        x_projected = remove_mean_with_mask(x_masked, node_mask)
    else:
        x_projected = x_masked
    return x_projected
'''

def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def generate_index_tensor(tensor):
    """
    index tensor generation function
    Args:
        tensor (torch.Tensor): input tensor

    Return:
        torch.Tensor
    """
    indices = torch.arange(len(tensor), device=tensor.device)
    index_tensor = torch.repeat_interleave(indices,tensor)
    return index_tensor


def split_and_padding(coords, batch, size=3):
    """
    Split molecular coordinates into corresponding molecules based on batch indices, 
    and pad molecules with fewer atoms to match the maximum atom count.

    Args:
        coords (np.ndarray): Molecular coordinates with shape (X, 3)
        batch (np.ndarray): Batch indices indicating molecule membership for each atom, shape (X,)

    Returns:
        tuple: Contains two elements:
            - padded_coords (np.ndarray): Padded coordinates array with shape (num_molecules, max_atoms, 3)
            - mask (torch.Tensor): Boolean mask indicating valid atoms, shape (num_molecules, max_atoms)
    """
    num_molecules = torch.max(batch) + 1
    atom_counts = torch.bincount(batch)
    max_atoms = torch.max(atom_counts)
    padded_coords = torch.zeros((num_molecules, max_atoms, size)).to(coords.device)
    mask = torch.zeros((num_molecules, max_atoms), dtype=torch.bool).to(coords.device)
    for mol_id in range(num_molecules):
        mol_coords = coords[batch == mol_id]
        padded_coords[mol_id, :len(mol_coords)] = mol_coords
        
        mask[mol_id, :len(mol_coords)] = True
    return padded_coords, mask

def get_optim(lr, model, optimizer_type="adamw"):
    if optimizer_type.lower() == 'adamw':
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=lr, amsgrad=True,
            weight_decay=1e-12)
    elif optimizer_type.lower() == 'muon':
        optim = Muon(model.parameters(),
                     lr=lr,weight_decay=1e-12)
    else:
        raise NotImplementedError("Optimizer {} not implemented".format(optimizer_type))

    return optim

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)
    
def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))
    
    #if float(grad_norm) > max_grad_norm:
    #    print(f'Clipped gradient with value {grad_norm:.1f} '
    #          f'while allowed {max_grad_norm:.1f}')
    return grad_norm

def setup_logger(save_dir):

    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(f"{config.model.save_dir}/{config.data.data_path.split('/')[-1]}", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{save_dir}/{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def str_to_bool(value):
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("must be True/False")
    
def random_transform(coordinates, max_translation=5.0, max_rotation_deg=360.0, 
                    seed=None, rotate_around_center=True):
    center = coordinates.mean(axis=0)
    if max_translation < 0 or max_rotation_deg < 0:
        raise ValueError("The parameters for translation and rotation must be non-negative")

    rng = np.random.default_rng(seed)

    # translation
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    translation = max_translation * direction

    # rotation
    rotation = Rotation.from_rotvec(
        rng.uniform(-1, 1, 3) * np.radians(max_rotation_deg)
    )


    new_coords = np.copy(coordinates)

    if rotate_around_center:
        centered = new_coords - center
        rotated = rotation.apply(centered)
        new_coords = rotated + center
    else:
        new_coords = rotation.apply(new_coords)

    new_coords += translation

    return new_coords

def gather_selected_nodes_and_compute_mean(context, select_node_index):
    """
    Args:
        context: torch.Tensor, shape [batch_size, node_num, emb_dim]
        select_node_index: torch.Tensor, shape [batch_size, 10] (padding index = -1)
    
    Returns:
        selected_mean: torch.Tensor, shape [batch_size, emb_dim]
    """
    batch_size, node_num, emb_dim = context.shape
    _, select_num = select_node_index.shape
    
    padded_index = select_node_index.clone()
    padded_index[padded_index == -1] = 0  
    
    index_expanded = padded_index.unsqueeze(-1).expand(-1, -1, emb_dim)  # [batch_size, 10, emb_dim]
    

    selected = context.gather(
        dim=1,
        index=index_expanded
    )  
    
    
    mask = (select_node_index != -1).float()  
    mask_expanded = mask.unsqueeze(-1)  
    

    masked_selected = selected * mask_expanded  
    sum_selected = masked_selected.sum(dim=1)  
    valid_counts = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
    
    valid_counts = torch.clamp(valid_counts, min=1e-6)
    selected_mean = sum_selected / valid_counts
    
    return selected_mean

def calc_angle_over_dataset(dataset):
    angle_query_map = {}
    for data in tqdm(dataset):
        coords = data.mol_coords
        edge_index = data.edge_index
        at_types = data.x[:,-1]
        bond_types = data.edge_attr
        angles = calc_angles(coords,edge_index).squeeze()
        for eg_idx, bd_type,angle in zip(edge_index.T, bond_types, angles):
            key = tuple(map(int,[*bd_type, at_types[eg_idx[0]], at_types[eg_idx[1]]]))
            if key not in angle_query_map:
                angle_query_map[key] = [float(angle)]
            else:
                angle_query_map[key].append(float(angle))
    return angle_query_map

def read_freq_mode_from_lines(lines,freq_start_idx_lst,atom_num):
    freqs = np.concatenate([list(map(float,lines[idx].strip().split()[2:])) for idx in freq_start_idx_lst])
    modes = []
    for idx in freq_start_idx_lst:
        modes_0 = []
        modes_1 = []
        modes_2 = []
        for line in lines[idx+5:idx+5+atom_num]:
            mode_0 = list(map(float,line.strip().split()[2:5]))
            mode_1 = list(map(float,line.strip().split()[5:8]))
            mode_2 = list(map(float,line.strip().split()[8:11]))
            if len(mode_0) != 0:
                modes_0.append(mode_0)
            if len(mode_1) != 0:
                modes_1.append(mode_1)
            if len(mode_2) != 0:
                modes_2.append(mode_2)
        if len(modes_0) != 0:
            modes.append(modes_0)
        if len(modes_1) != 0:
            modes.append(modes_1)
        if len(modes_2) != 0:
            modes.append(modes_2)
    modes = np.array(modes)
    modes_shape = modes.shape
    if modes_shape[1]*3-6 != modes_shape[0]:
        print(f"[WARN] The number of modes is not equal to the number of atoms * 3 - 6, current atom num is {atom_num}, modes shape is {modes_shape}")
    return freqs,modes

def read_optlog(log_file):
    with open(log_file,'r',errors='ignore') as fr:
        lines = fr.readlines()
    if "Normal termination of Gaussian 16" in lines[-1]:
        
        atom_num,chrg,mult = None,None,None
        freq_start_idx_lst = []
        geom_info_lst = []
        geom_start = False
        coord_start_idx_lst = []
        TCG = None
        TCH = None
        for i,line in enumerate(lines):
            if 'NAtoms=' in line:
                atom_num = eval(line.split()[1])
            elif ' Charge =' in line and 'Multiplicity ='  in line:
                chrg,mult = eval(line.strip().split()[2]),eval(line.strip().split()[5])
            elif 'Input orientation:' in line:
                geom_start = True
                tmp_geom_inf = {'input_orientation':i+5}
                coord_start_idx_lst.append(i+5)
            elif 'Standard orientation:' in line and geom_start and len(tmp_geom_inf) == 1:
                tmp_geom_inf['standard_orientation'] = i+5
            elif len(line.strip().split()) == 9 and line.strip().split()[0] == 'SCF' and line.strip().split()[1] == 'Done:' and geom_start and len(tmp_geom_inf) == 2:
                tmp_geom_inf['scf_energy'] = eval(line.strip().split()[4])

            elif 'Center     Atomic                   Forces (Hartrees/Bohr)' in line and geom_start and len(tmp_geom_inf) == 3:
                #print("NAtom:", atom_num)
                tmp_geom_inf['forces'] = i+3
                geom_start = False
                ipt_ori_start_idx = tmp_geom_inf['input_orientation']
                std_ori_start_idx = tmp_geom_inf['standard_orientation']
                force_start_idx = tmp_geom_inf['forces']
                tmp_geom_inf['input_orientation'] = np.array([list(map(float,line.strip().split()[-3:])) for line in lines[ipt_ori_start_idx:ipt_ori_start_idx+atom_num]])
                tmp_geom_inf['standard_orientation'] = np.array([list(map(float,line.strip().split()[-3:])) for line in lines[std_ori_start_idx:std_ori_start_idx+atom_num]])
                tmp_geom_inf['forces'] = np.array([list(map(float,line.strip().split()[-3:])) for line in lines[force_start_idx:force_start_idx+atom_num]])
                geom_info_lst.append(tmp_geom_inf)
            elif 'Frequencies --' in line:
                freq_start_idx_lst.append(i)
            elif line.strip().split()[:6] == ['Thermal', 'correction', 'to', 'Gibbs', 'Free', 'Energy='] and len(line.strip().split()) == 7:
                TCG = eval(line.strip().split()[6])
            elif line.strip().split()[:4] == ['Thermal', 'correction', 'to', 'Enthalpy='] and len(line.strip().split()) == 5:
                TCH = eval(line.strip().split()[4])
        freqs,modes = read_freq_mode_from_lines(lines,freq_start_idx_lst,atom_num)
        
        atom_coords_string = [lines[idx:idx+atom_num] for idx in coord_start_idx_lst]
        atoms = np.array([pt.GetElementSymbol(int(line.strip().split()[1])) for line in atom_coords_string[0]])
        coords = np.array([[list(map(float,line.strip().split()[-3:])) for line in string_] for string_ in atom_coords_string])
        assert np.max(np.abs(coords[-1] - geom_info_lst[-1]['input_orientation'])) < 1e-6
        
        data = {'atoms':atoms,'geom_info':geom_info_lst,'opted_geom':coords[-1],'freqs':freqs,'modes':modes,'TCG':TCG,'TCH':TCH,'charge':chrg,'mult':mult}
        return data, atoms, coords, chrg, mult
    else:
        return None, None, None, None, None

def create_reaction_with_atom_mapping(rct_mol, pdt_mol):
    for atom in rct_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    for atom in pdt_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    rxn = AllChem.ChemicalReaction()
    rxn.AddReactantTemplate(rct_mol)
    rxn.AddProductTemplate(pdt_mol)
    return rxn

def calculate_gaussview_spin_multiplicity(element_list, charge=0):
    
    if isinstance(element_list[0],str):
        element_list = [pt.GetAtomicNumber(element) for element in element_list]
    
    if not element_list:
        return 1  # 

    element_counts = Counter(element_list)
    total_valence_electrons = 0
    for element, count in element_counts.items():
        
        total_valence_electrons += pt.GetNOuterElecs(element) * count
    total_electrons_in_system = total_valence_electrons - charge

    if total_electrons_in_system % 2 == 0:
        return 1
    else:
        return 2

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps    


def compute_angles(vec1, vec2):
    vec1_norm = vec1 / (torch.norm(vec1, dim=1, keepdim=True) + 1e-8)
    vec2_norm = vec2 / (torch.norm(vec2, dim=1, keepdim=True) + 1e-8)
    cos_angles = torch.sum(vec1_norm * vec2_norm, dim=1)
    cos_angles = torch.clamp(cos_angles, -0.999999, 0.999999)
    return torch.acos(cos_angles)

def str_to_bool(value):
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("must be True/False")

def atoms_to_formula(atoms_list):
    atom_count = Counter(atoms_list)
    sorted_keys = sorted(atom_count.keys())
    formula = ""
    for atom in sorted_keys:
        count = atom_count[atom]
        if count == 1:
            formula += atom
        else:
            formula += f"{atom}{count}"
    
    return formula

def clear_atom_map(mol):
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return mol