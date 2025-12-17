import torch,logging,time
import numpy as np
from itertools import combinations
from rdkit.Chem.Draw import MolDrawOptions
from rdkit.Chem import Draw
from rdkit import Chem
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, HungarianOrderMatcher, KabschMatcher
from pymatgen.io.xyz import XYZ
from pymatgen.core import Molecule
pt = Chem.GetPeriodicTable()

class GeneticOrderMatcher(KabschMatcher):
    """This method was inspired by genetic algorithms and tries to match molecules
    based on their already matched fragments.

    It uses the fact that when two molecule is matching their sub-structures have to match as well.
    The main idea here is that in each iteration (generation) we can check the match of all possible
    fragments and ignore those which are not feasible.

    Although in the worst case this method has N! complexity (same as the brute force one),
    in practice it performs much faster because many of the combination can be eliminated
    during the fragment matching.

    Notes:
        This method very robust and returns with all the possible orders.

        There is a well known weakness/corner case: The case when there is
        a outlier with large deviation with a small index might be ignored.
        This happens due to the nature of the average function
        used to calculate the RMSD for the fragments.

        When aligning molecules, the atoms of the two molecules **must** have the
        same number of atoms from the same species.
    """

    def __init__(self, target: Molecule, threshold: float, timeout: float = None):
        """Constructor of the matcher object.

        Args:
            target: a `Molecule` object used as a target during the alignment
            threshold: value used to match fragments and prune configuration
        """
        super().__init__(target)
        self.threshold = threshold
        self.timeout = timeout
        self.N = len(target)

    def match(self, p: Molecule):
        """Similar as `KabschMatcher.match` but this method also finds all of the
        possible atomic orders according to the `threshold`.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            Array of the possible matches where the elements are:
                inds: The indices of atoms
                U: 3x3 rotation matrix
                V: Translation vector
                rmsd: Root mean squared deviation between P and Q
        """
        out = []
        start_time = time.time()
        for inds in self.permutations(p):
            p_prime = p.copy()
            p_prime._sites = [p_prime[idx] for idx in inds]

            U, V, rmsd = super().match(p_prime)

            out.append((inds, U, V, rmsd))
            end_time = time.time()
            if self.timeout is not None and end_time - start_time > self.timeout:
                return []
        return out

    def fit(self, p: Molecule):
        """Order, rotate and transform all of the matched `p` molecule
        according to the given `threshold`.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            list[tuple[Molecule, float]]: possible matches where the elements are:
                p_prime: Rotated and translated of the `p` `Molecule` object
                rmsd: Root-mean-square-deviation between `p_prime` and the `target`
        """
        out: list[tuple[Molecule, float]] = []
        start_time = time.time()
        for inds in self.permutations(p):
            
            p_prime = p.copy()
            p_prime._sites = [p_prime[idx] for idx in inds]

            U, V, rmsd = super().match(p_prime)

            # Rotate and translate matrix `p` onto the target molecule.
            # P' = P * U + V
            for site in p_prime:
                site.coords = np.dot(site.coords, U) + V

            out += [(p_prime, rmsd)]
            end_time = time.time()
            if self.timeout is not None and end_time - start_time > self.timeout:
                print(f"GeneticOrderMatcher.fit timeout {self.timeout} seconds")
                return []
        return out

    def permutations(self, p: Molecule):
        """Generate all of possible permutations of atom order according the threshold.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            Array of index arrays
        """
        # caching atomic numbers and coordinates
        p_atoms, q_atoms = p.atomic_numbers, self.target.atomic_numbers
        p_coords, q_coords = p.cart_coords, self.target.cart_coords

        if sorted(p_atoms) != sorted(q_atoms):
            raise ValueError("The number of the same species aren't matching!")

        # starting matches (only based on element)
        partial_matches = [[j] for j in range(self.N) if p_atoms[j] == q_atoms[0]]
        matches: list = []
        start_time = time.time()
        for idx in range(1, self.N):
            end_time = time.time()
            if self.timeout and end_time - start_time > self.timeout:
                print(f"GeneticOrderMatcher.permutations timeout {self.timeout} seconds")
                return []
            #print(idx,self.N)
            # extending the target fragment with then next atom
            f_coords = q_coords[: idx + 1]
            f_atom = q_atoms[idx]

            f_trans = f_coords.mean(axis=0)
            f_centroid = f_coords - f_trans

            matches = []
            for indices in partial_matches:
                for jdx in range(self.N):
                    # skipping if the this index is already matched
                    if jdx in indices:
                        continue

                    # skipping if they are different species
                    if p_atoms[jdx] != f_atom:
                        continue

                    inds = [*indices, jdx]
                    P = p_coords[inds]

                    # Both sets of coordinates must be translated first, so that
                    # their centroid coincides with the origin of the coordinate system.
                    p_trans = P.mean(axis=0)
                    p_centroid = P - p_trans

                    # The optimal rotation matrix U using Kabsch algorithm
                    U = self.kabsch(p_centroid, f_centroid)

                    p_prime_centroid = np.dot(p_centroid, U)
                    rmsd = np.sqrt(np.mean(np.square(p_prime_centroid - f_centroid)))

                    # rejecting if the deviation is too large
                    if rmsd > self.threshold:
                        continue
                    matches.append(inds)
            partial_matches = matches

        return matches

def display_atom_indices(mol, size=(300, 300)):
    """
    show atom index
    """
    opts = MolDrawOptions()
    opts.addAtomIndices = True
    opts.bondLineWidth = 2
    opts.annotationFontScale = 1.0
    img = Draw.MolToImage(mol, size=size, options=opts)
    return img

def compute_distance_matrix(coords, atom_mask):
    batch_size, max_atom_num, _ = coords.shape
    coords_expanded1 = coords.unsqueeze(2) 
    coords_expanded2 = coords.unsqueeze(1)
    coord_diff = coords_expanded1 - coords_expanded2
    distance_sq = torch.sum(coord_diff ** 2, dim=-1)
    distance_matrix = torch.sqrt(distance_sq + 1e-10)  
    atom_mask_squeezed = atom_mask.squeeze(-1)
    valid_mask = atom_mask_squeezed.unsqueeze(2) & atom_mask_squeezed.unsqueeze(1)
    distance_matrix = distance_matrix * valid_mask.float()
    
    return distance_matrix, valid_mask

def get_reactive_edge_index(data):
    all_reactive_pairs = []
    for idx in range(len(data.reactive_atoms)):
        ptr = data.ptr[idx]
        reactive_atoms_ = data.reactive_atoms[idx][data.reactive_atoms[idx] != -1]
        react_atom_pairs = torch.tensor([pair for pair in combinations(reactive_atoms_,2)])
        react_atom_pairs = react_atom_pairs + ptr
        all_reactive_pairs.append(react_atom_pairs)
    all_reactive_edge_index = torch.concatenate(all_reactive_pairs,dim=0).T
    all_reactive_edge_index_rev = torch.stack([all_reactive_edge_index[1],all_reactive_edge_index[0]])
    return torch.concat([all_reactive_edge_index,all_reactive_edge_index_rev], dim=1)

def calc_truth_false_bond_dist(x_truth,x_pred,node_mask,edge_index,batch):
    edge_index = edge_index.long()
    x_final_pad = x_pred[node_mask.squeeze().bool()]
    pred_final_i = x_final_pad[edge_index[0]]
    pred_final_j = x_final_pad[edge_index[1]]
    x_truth_pad = x_truth[node_mask.squeeze().bool()]
    truth_i = x_truth_pad[edge_index[0]]
    truth_j = x_truth_pad[edge_index[1]]
    pred_final_ij_dist = np.linalg.norm(pred_final_i-pred_final_j,axis=1)
    truth_ij_dist = np.linalg.norm(truth_i-truth_j,axis=1)

    dist_diff_abs = np.abs(pred_final_ij_dist - truth_ij_dist)
    
    edge_batch = batch[edge_index[0]]
    dist_diff_abs_norm_in_mol_batch = []
    #dist_diff_abs_norm_in_mol_batch_mean = []
    for mol_idx in range(batch.max()+1):
        mol_edge_mask = (edge_batch == mol_idx)
        dist_diff_abs_norm_in_mol = dist_diff_abs[mol_edge_mask]
        dist_diff_abs_norm_in_mol_batch.append(dist_diff_abs_norm_in_mol)
        #dist_diff_abs_norm_in_mol_batch_mean.append(dist_diff_abs_norm_in_mol.mean())
    return dist_diff_abs_norm_in_mol_batch

def sc2pmg(species,coords):
    mol = Molecule(
        species=species,
        coords=coords,
    )
    return mol

def rmsd_core(mol1, mol2, threshold=0.5, same_order=False, timeout=None):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
        
    #print("total_permutations:",total_permutations)
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold, timeout=timeout)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd

def calc_rmsd(mol_at,pos1,pos2,threshold=0.5,same_order=False,ignore_chirality=True,timeout=None):
    mol1 = sc2pmg(mol_at,pos1)
    mol2 = sc2pmg(mol_at,pos2)
    
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order, timeout=timeout)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order, timeout=timeout)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd

def calc_rmsd_batch(mol_atoms, node_mask, mol_xyz_truth_batch, mol_xyz_pred_batch,reactive_atom_rmsd,
                    reactive_atoms,rmsd_threshold,same_order,ignore_chirality,verbose=True):
    rmsd_batch = []
    for data_idx in range(len(mol_atoms)):
        mol_at = mol_atoms[data_idx]
        mol_xyz_pred = mol_xyz_pred_batch[data_idx]
        mol_xyz_truth = mol_xyz_truth_batch[data_idx]
        mol_node_mask = node_mask[data_idx]
        mol_at = [pt.GetElementSymbol(int(at)) for at in mol_at[mol_node_mask.bool().squeeze()].squeeze()]
        mol_xyz_pred = mol_xyz_pred[mol_node_mask.bool().squeeze()]
        assert len(mol_at) == len(mol_xyz_pred) == len(mol_xyz_truth)
        if not reactive_atom_rmsd:
            rmsd = calc_rmsd(mol_at,mol_xyz_pred,mol_xyz_truth,threshold=rmsd_threshold,same_order=same_order,ignore_chirality=ignore_chirality)
        else:
            mol_reactat = reactive_atoms[data_idx].cpu().numpy()
            mol_reactat = mol_reactat[mol_reactat!=-1]
            reactat_mol_at = np.array(mol_at)[mol_reactat]
            reactat_mol_xyz_pred = mol_xyz_pred[mol_reactat]
            reactat_mol_xyz_truth = mol_xyz_truth[mol_reactat]
            rmsd = calc_rmsd(reactat_mol_at,reactat_mol_xyz_pred,reactat_mol_xyz_truth,threshold=rmsd_threshold,same_order=same_order,ignore_chirality=ignore_chirality)
        if verbose:
            logging.info(f"[{data_idx+1} / {len(mol_atoms)}] RMSD: {rmsd}")
        rmsd_batch.append(rmsd)
    return rmsd_batch

def refine_sample_result_batch(pred_final_lst,node_mask,mol_atoms,edge_index,batch,type_="not too close",th=1.1):

    min_max_ratio_repeat = []
    ratio_bias_repeat = []
    for pred_final in pred_final_lst:
        pred_final_pad = pred_final[node_mask.squeeze().bool()]
        mol_atoms_pad = mol_atoms[node_mask.squeeze().bool()].squeeze()
        pred_final_i = pred_final_pad[edge_index[0]]
        pred_final_j = pred_final_pad[edge_index[1]]
        pred_final_ij_dist = np.linalg.norm(pred_final_i-pred_final_j,axis=1)
        mol_atom_i = mol_atoms_pad[edge_index[0]]
        mol_atom_j = mol_atoms_pad[edge_index[1]]
        edge_with_nH_mask = torch.stack([mol_atom_i!=1,mol_atom_j!=1]).all(dim=0)
        mol_atom_i_r = np.array([pt.GetRcovalent(int(at)) for at in mol_atom_i])
        mol_atom_j_r = np.array([pt.GetRcovalent(int(at)) for at in mol_atom_j])
        mol_atom_ij_default_r = mol_atom_i_r+mol_atom_j_r
        pred_vs_default = pred_final_ij_dist/mol_atom_ij_default_r
        
        mol_ratio_dict,mol_ratio_arr = assign_edgeattr_to_mol(batch, edge_index, pred_vs_default, torch.ones_like(edge_with_nH_mask).bool())
        min_max_ratio = [[mol_ratio_dict[key].min(),mol_ratio_dict[key].max()] for key in mol_ratio_dict]
        ratio_bias = [np.abs(mol_ratio_dict[key] - th).mean() for key in mol_ratio_dict]
        min_max_ratio_repeat.append(min_max_ratio)
        ratio_bias_repeat.append(ratio_bias)
        
    min_max_ratio_repeat = np.array(min_max_ratio_repeat)
    if type_ == "not too close":
        sel_repeat_id = min_max_ratio_repeat[:,:,1].argmin(axis=0)
    elif type_ == "not too far":
        sel_repeat_id = min_max_ratio_repeat[:,:,0].argmax(axis=0)
    elif type_ == "div close to 1":
        sel_repeat_id = np.array(ratio_bias_repeat).argmin(axis=0)
    else:
        raise NotImplementedError
    refined_final_preds = []
    for batch_idx, sel_idx in enumerate(sel_repeat_id):
        sel_final_pred = pred_final_lst[sel_idx][batch_idx]
        refined_final_preds.append(sel_final_pred)
    refined_final_preds = np.array(refined_final_preds)
    return refined_final_preds,sel_repeat_id

def assign_edgeattr_to_mol(batch, edge_index, edge_attr, edgeH_mask):
    molecules_ratios = {}
    
    edge_batch = batch[edge_index[0]] 
    max_edge_num = 0
    for mol_idx in range(batch.max() + 1):
        mol_edge_mask = (edge_batch == mol_idx)
        mol_edge_mask = mol_edge_mask & edgeH_mask
        mol_ratios = edge_attr[mol_edge_mask]
        molecules_ratios[mol_idx] = mol_ratios
        max_edge_num = max(max_edge_num, len(mol_ratios))
    mol_ratio_arr = np.zeros((len(molecules_ratios), max_edge_num))
    for i, ratios in enumerate(molecules_ratios.values()):
        mol_ratio_arr[i, :len(ratios)] = ratios
    return molecules_ratios,mol_ratio_arr

def refine_sample_result(pred_xyz_lst,atoms,graph,sel_type="not too close",th=1.1):

    assert sel_type in ["not too close", "not too far", "close to threshold"]

    min_max_ratio_repeat = []
    ratio_bias_repeat = []

    for pred_final in pred_xyz_lst:

        pred_final_i = pred_final[graph.edge_index[0]]
        pred_final_j = pred_final[graph.edge_index[1]]
        pred_final_ij_dist = np.linalg.norm(pred_final_i-pred_final_j,axis=1)
        mol_atom_i = atoms[graph.edge_index[0]]
        mol_atom_j = atoms[graph.edge_index[1]]
        mol_atom_i_r = np.array([pt.GetRcovalent(int(at)) for at in mol_atom_i])
        mol_atom_j_r = np.array([pt.GetRcovalent(int(at)) for at in mol_atom_j])
        mol_atom_ij_default_r = mol_atom_i_r+mol_atom_j_r
        pred_vs_default = pred_final_ij_dist/mol_atom_ij_default_r
        min_max_ratio_repeat.append([pred_vs_default.min(), pred_vs_default.max()])
        ratio_bias_repeat.append(np.abs(pred_vs_default-th).mean())
    min_max_ratio_repeat = np.array(min_max_ratio_repeat)
    ratio_bias_repeat = np.array(ratio_bias_repeat)

    if sel_type == "not too far":
        sel_repeat_id = np.argmin(min_max_ratio_repeat[:, 1])
    elif sel_type == "not too close":
        sel_repeat_id = np.argmax(min_max_ratio_repeat[:, 0])
    elif sel_type == "close to threshold":
        sel_repeat_id = np.argmin(ratio_bias_repeat)
        
    return sel_repeat_id, pred_xyz_lst[sel_repeat_id]

def compute_pairwise_distances(coords,atom_mask=None):

    delta = coords.unsqueeze(1) - coords.unsqueeze(0)  # [X, X, 3]
    
    dist_sq = torch.sum(delta ** 2, dim=-1)
    distance_matrix = torch.sqrt(dist_sq) 
    if atom_mask is not None:
        dist_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(0)  # [X, X]
        distance_matrix = distance_matrix * dist_mask
    return distance_matrix

def calc_dist_error(xyz_truth,xyz_pred,atom_mask=None):
    dist_error = []
    dist_error_max = []
    for _idx in range(len(xyz_pred)):
        if atom_mask is not None:
            at_mask = atom_mask[_idx]
        else:
            at_mask = None
        dist_mat_pred = compute_pairwise_distances(xyz_pred[_idx],atom_mask=at_mask)[0]
        dist_mat_truth = compute_pairwise_distances(xyz_truth[_idx],atom_mask=at_mask)[0]
        #print(dist_mat_pred.shape,dist_mat_truth.shape)
        _dist_error = torch.mean(torch.abs(dist_mat_pred-dist_mat_truth))
        _dist_error_max = torch.max(torch.abs(dist_mat_pred-dist_mat_truth))
        dist_error.append(_dist_error)
        dist_error_max.append(_dist_error_max)
    dist_error_mean = np.mean(dist_error)
    dist_error_max_ = np.max(dist_error_max)
    return dist_error_mean,dist_error,dist_error_max_,dist_error_max

def atom_coords2pmg(atoms,coords):
    mol = Molecule(
        species=atoms,
        coords=coords,
    )
    return mol

def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol

def pymatgen_rmsd(mol1,mol2,ignore_chirality=False,threshold=0.5,same_order=False):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold, same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd