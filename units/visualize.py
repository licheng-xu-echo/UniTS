import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'Li': '#CC80FF', 'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 
    'F': '#90E050', 'Na': '#AA55FF', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C85A', 'P': '#FF8000', 
    'S': '#FFFF30', 'Cl': '#1FF01F', 'K': '#8F3FD6', 'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 
    'Mn': '#9C7AC7', 'Fe': '#800000', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#AA8080', 'Zn': '#C88033', 
    'Ga': '#C28F8F', 'Se': '#FFA000', 'Br': '#A62929', 'Ru': '#248F8F', 'Rh': '#0A7D87', 'Pd': '#006986', 
    'Ag': '#848484', 'In': '#A67573', 'Sn': '#668080', 'Sb': '#9966CC', 'I': '#940094', 'Cs': '#550080', 
    'La': '#70D4FF', 'W': '#2194D6', 'Os': '#668080', 'Ir': '#178F8F', 'Pt': '#D0D0E0', 'Au': '#FFD700'
}
pt = Chem.GetPeriodicTable()

def load_xyz_trajectory(file_path: str) -> list:
    """
    Loads a multi-frame XYZ trajectory file.
    Returns: A list of frames: [{'atom_types': [], 'positions': np.array}, ...]
    """
    trajectory = []
    with open(file_path, encoding='utf8') as f:
        while True:
            try:
                line = f.readline()
                if not line: break
                n_atoms = int(line.strip())
            except ValueError:
                break
                
            f.readline() # Skip comment line
            
            atom_types = []
            positions = np.zeros((n_atoms, 3))
            
            for i in range(n_atoms):
                atom_line = f.readline().split()
                if not atom_line: raise EOFError
                
                atom_types.append(atom_line[0])
                positions[i, 0] = float(atom_line[1])
                positions[i, 1] = float(atom_line[2])
                positions[i, 2] = float(atom_line[3])
                
            trajectory.append({
                'atom_types': np.array(atom_types),
                'positions': positions
            })
            
    if not trajectory:
        raise ValueError(f"No valid molecule frames found in file: {file_path}")
        
    return trajectory

def get_bonds_for_frame(
    atom_types: np.ndarray, 
    positions: np.ndarray, 
    covalent_factor: float = 1.2,
    force_bond: list = None, 
    force_nobond: list = None
) -> list:
    """
    Determines bonds based on distance and custom rules.
    Distance threshold: dist <= (R_i + R_j) * covalent_factor.
    
    Args:
        atom_types (np.ndarray): List of element symbols (e.g., ['C', 'H', ...])
        positions (np.ndarray): Atomic coordinates (N x 3)
        covalent_factor (float): Tolerance factor for the sum of covalent radii.
        force_bond (list): List of atomic index pairs to force as bonded, e.g., [(0, 1)]
        force_nobond (list): List of atomic index pairs to force as non-bonded, e.g., [(2, 5)]
        
    Returns:
        list: List of bonded atomic index pairs, e.g., [(0, 1), (0, 2), ...]
    """
    n_atoms = len(atom_types)
    bonds = []
    
    force_bond_set = {tuple(sorted(p)) for p in (force_bond or [])}
    force_nobond_set = {tuple(sorted(p)) for p in (force_nobond or [])}

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pair = tuple(sorted((i, j)))

            # 1. Check override rules
            if pair in force_nobond_set:
                continue
            if pair in force_bond_set:
                bonds.append(pair)
                continue

            # 2. Get RDKit Covalent Radii (Å)
            R_i = pt.GetRcovalent(atom_types[i])
            R_j = pt.GetRcovalent(atom_types[j])
            
            # Bonding distance threshold
            threshold_sq = ((R_i + R_j) * covalent_factor) ** 2

            # 3. Calculate actual distance squared
            dist_sq = np.sum((positions[i] - positions[j]) ** 2)

            # 4. Distance check
            if dist_sq <= threshold_sq:
                bonds.append(pair)
                
    return bonds
def plot_molecule(
    ax: Axes3D, 
    atom_types: np.ndarray, 
    positions: np.ndarray, 
    bonds: list
):
    """Plots a single molecular structure on a Matplotlib 3D axis."""
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    # --- Atom Drawing ---
    colors = [ELEMENT_COLORS.get(t, '#808080') for t in atom_types]
    
    # Use RDKit radius for size calculation
    radii = np.array([pt.GetRcovalent(t) for t in atom_types])
    sizes = radii * 100 # Scale size for visibility
    
    # Using 2D scatter plot for performance
    ax.scatter(x, y, z, s=sizes, c=colors, depthshade=False) 
    
    # --- Bond Drawing ---
    bond_color = '#FFFFFF'
    bond_linewidth = 2.0
    
    for i, j in bonds:
        p1 = positions[i]
        p2 = positions[j]
        
        ax.plot(
            [p1[0], p2[0]], 
            [p1[1], p2[1]], 
            [p1[2], p2[2]], 
            color=bond_color, 
            linewidth=bond_linewidth,
            alpha=0.9
        )

def plot_data3d_frame(
    frame_data: dict, 
    frame_idx: int, 
    save_dir: str, 
    bond_params: dict, 
    axis_lim: float,
    elev: float = 0.0,
    azim: float = 0.0,
    roll: float = 0.0,
) -> str:
    """Plots and saves a single frame."""
    
    atom_types = frame_data['atom_types']
    positions = frame_data['positions']
    
    # Center coordinates
    positions -= positions.mean(axis=0)
    
    # Get dynamic bonds
    bonds = get_bonds_for_frame(atom_types, positions, **bond_params)
    
    # Create Matplotlib figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    
    # --- Plotting Style ---
    ax.set_aspect('auto')
    ax.set_facecolor('black')
    ax._axis3don = False
    
    # Set view and fixed axis limits for stable animation
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    # Plot the molecule
    plot_molecule(ax, atom_types, positions, bonds)

    # Save image
    save_path = os.path.join(save_dir, f'frame_{frame_idx:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=150)
    plt.close(fig)
    
    return save_path

def visualize_and_animate_trajectory(
    xyz_file_path: str, 
    output_gif_path: str,
    bond_params: dict,
    selected_frames: list = None,
    elev: float = 0,
    azim: float = 0,
    roll: float = 0,
    fps: int = 30
):
    """
    Main function: Loads trajectory, plots frame-by-frame, and generates a GIF.
    """
    print(f"Loading trajectory file: {xyz_file_path}")
    trajectory = load_xyz_trajectory(xyz_file_path)
    if selected_frames is not None:
        trajectory = [trajectory[i] for i in selected_frames]
    temp_dir = 'temp_png_frames'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    all_save_paths = []
    
    # 1. Determine maximum coordinate extent for stable axis limits
    max_coords = max(np.abs(frame['positions']).max() for frame in trajectory)
    plot_axis_lim = max(max_coords * 1.5, 3.5) # Set a fixed limit for all frames


    print(f"Total {len(trajectory)} frame(s), Starting frame-by-frame plotting...")
    for i, frame_data in enumerate(trajectory):
        if (i + 1) % 100 == 0:
            print(f"  - Processing frame {i+1}/{len(trajectory)}...")
            
        save_path = plot_data3d_frame(
            frame_data, i, temp_dir, bond_params, axis_lim=plot_axis_lim, elev=elev, azim=azim, roll=roll
        )
        all_save_paths.append(save_path)
        
    print(f"All {len(all_save_paths)} frames plotted. Generating GIF...")

    # 2. Generate GIF using imageio
    imgs = [imageio.imread(fn) for fn in all_save_paths]
    
    imageio.mimsave(output_gif_path, imgs, fps=fps)
    
    print(f"🎉 Animation successfully saved to: {output_gif_path}")
    
    # 3. Cleanup temporary files
    for fn in all_save_paths:
        os.remove(fn)
    os.rmdir(temp_dir)
    print("Temporary files cleaned up.")
    return imgs[-1]