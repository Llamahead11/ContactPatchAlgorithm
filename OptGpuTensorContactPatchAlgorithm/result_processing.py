import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import matplotlib as mpl

import pyvista as pv
mpl.rcParams.update({
    # --- Figure setup ---
    'figure.figsize': [7.5, 5],       # MATLAB default aspect
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',

    # --- Font and text ---
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'mathtext.default': 'regular',    # cleaner like MATLAB (not italic math)

    # --- Axes appearance ---
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.titlepad': 10,
    'axes.prop_cycle': plt.cycler('color', [
        (0, 0.4470, 0.7410),   # blue
        (0.8500, 0.3250, 0.0980),  # orange
        (0.9290, 0.6940, 0.1250),  # yellow
        (0.4940, 0.1840, 0.5560),  # purple
        (0.4660, 0.6740, 0.1880),  # green
        (0.3010, 0.7450, 0.9330),  # light blue
        (0.6350, 0.0780, 0.1840)   # dark red
    ]),

    # --- Grid style ---
    'grid.color': '0.85',
    'grid.linewidth': 0.9,
    'grid.linestyle': '-',

    # --- Tick style ---
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    # --- Lines ---
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.5,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',

    # --- Legend ---
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': True,
    'legend.edgecolor': '0.85',
    'legend.loc': 'best',

    # --- Figure background ---
    'figure.facecolor': 'white',
})

from app_vis import Viewer3D
import vidVisualiser as vV

#===========================================================================================================

# Path to your saved files
folder = "saved_arrays"

# All .npz files sorted
files = sorted(glob.glob(os.path.join(folder, "iteration_unc*.npz")))

# Parameters
batch_size = 50
start_iter = 200   # <-- change this to resume from any iteration number

#============================================================================================================

# Filter files to start from a specific iteration
files = [f for f in files if (int(os.path.basename(f).split('_')[-1].split('.')[0]) >= start_iter) & (int(os.path.basename(f).split('_')[-1].split('.')[0]) <= start_iter+batch_size)]

print(f"Starting from iteration {start_iter}, total files to load: {len(files)}")

# Example: global results list
data = []

# def process_batch(batch_data):
#     """Example batch processing function."""
#     batch_means = []
#     for data in batch_data:
#         # Example operation (replace with your real analysis)
#         deform = np.linalg.norm(data["pcd_inner_deform"], axis=1)
#         batch_means.append(deform.mean())
#     return batch_means

# Batch loop
# for i in range(0, len(files), batch_size):
#     batch_files = files[i:i+batch_size]
#     print(f"Loading batch {i//batch_size + 1}: {len(batch_files)} files")

#     batch_data = [np.load(f, allow_pickle=True) for f in batch_files]

#     results = process_batch(batch_data)
#     global_results.extend(results)

#     # Free memory
#     for d in batch_data:
#         d.close()
#     del batch_data

# current_idx = 0
# data = [np.load(f, allow_pickle=True) for f in files]

# print(data[0]["pcd_inner_deform"])

# inner, outer = data[current_idx]["pcd_inner_deform"],data[current_idx]["pcd_outer_deform"]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")

# # plot initial iteration
# inner_scatter = ax.scatter(inner[:, 0], inner[:, 1], inner[:, 2], s=1, c='b', label="inner")
# outer_scatter = ax.scatter(outer[:, 0], outer[:, 1], outer[:, 2], s=1, c='r', label="outer")
# ax.legend()
# ax.set_title(f"Iteration {current_idx}")

# # === UPDATE FUNCTION ===
# def update_plot(new_idx):
#     global current_idx, inner_scatter, outer_scatter

#     # Clamp index
#     new_idx = max(0, min(len(files) - 1, new_idx))
#     if new_idx == current_idx:
#         return

#     # Load new data
#     inner, outer = inner, outer = data[new_idx]["pcd_inner_deform"],data[new_idx]["pcd_outer_deform"]

#     # Update scatter plots
#     inner_scatter._offsets3d = (inner[:, 0], inner[:, 1], inner[:, 2])
#     outer_scatter._offsets3d = (outer[:, 0], outer[:, 1], outer[:, 2])

#     ax.set_title(f"Iteration {new_idx}")
#     plt.draw()

#     current_idx = new_idx

# # === KEYBOARD HANDLER ===
# def on_key(event):
#     if event.key == 'right':
#         update_plot(current_idx + 1)
#     elif event.key == 'left':
#         update_plot(current_idx - 1)

# fig.canvas.mpl_connect('key_press_event', on_key)
# plt.show()

#==============================================================================================

# Config
folder = "saved_arrays"
# files = sorted(glob.glob(os.path.join(folder, "iteration_*.npz")))
current_idx = 0
global idx
idx = current_idx

# Load first iteration
data = [np.load(f, allow_pickle=True) for f in files]
inner = data[current_idx]["pcd_inner_deform"]
outer = data[current_idx]["pcd_outer_deform"]
# data.close()

# Create PyVista plotter
plotter = pv.Plotter()
plotter.add_axes()  # Shows axes with numeric tick labels

# Add point clouds
pc_inner = pv.PolyData(inner)
pc_outer = pv.PolyData(outer)
plotter.add_points(pc_inner, color='blue', point_size=2)
plotter.add_points(pc_outer, color='red', point_size=2)

# Store iteration
# plotter.user_data = {"idx": current_idx}

# Function to update iteration
def update_iteration(key):
    global idx
    # idx = plotter.user_data["idx"]
    if key == "Right":
        idx = min(idx+1, len(files)-1)
    elif key == "Left":
        idx = max(idx-1, 0)
    else:
        return

    # data = np.load(files[idx])
    inner = data[idx]["pcd_inner_deform"]
    outer = data[idx]["pcd_outer_deform"]
    # data.close()

    # Update points
    plotter.clear()  # Remove previous points
    plotter.add_axes()
    plotter.add_points(pv.PolyData(inner), color='blue', point_size=2)
    plotter.add_points(pv.PolyData(outer), color='red', point_size=2)

    # plotter.user_data["idx"] = idx
    # print(f"Showing iteration {idx}")

# Bind keys
plotter.add_key_event("Right", lambda: update_iteration("Right"))
plotter.add_key_event("Left", lambda: update_iteration("Left"))

plotter.show()