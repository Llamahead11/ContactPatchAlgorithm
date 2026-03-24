import open3d as o3d
import numpy as np
from replay_realsense_tensor import read_RGB_D_folder
import time 
import cv2
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
import optix
import cupy as cp
import csv

import scipy.io as sio
from sklearn.decomposition import PCA

from scipy.optimize import minimize,least_squares

#
#find exact dimension of the tyre uninflated and inflated 
# 
#find move each point along its normal by that difference to simulate inflated tyre?

def fit_radial_center(pts):
    # PCA find radial plane first
    pca = PCA(n_components=3).fit(pts)
    radial_axis = pca.components_[-1]  # smallest eigenvector = radial

    # Project all points onto radial plane
    projection = pts - np.dot(pts, radial_axis)[:,None] * radial_axis

    # Fit circle in 2D projected coordinates
    mean_xy = np.median(projection, axis=0)
    return mean_xy, radial_axis

def fit_radial_center_circle(pts):
    """
    pts: (N,3) numpy array of points
    Returns:
        center_3d: 3D coordinates of radial center
        radial_axis: unit vector along radial axis
    """
    # PCA to get radial axis (smallest eigenvector)
    pca = PCA(n_components=3).fit(pts)
    radial_axis = pca.components_[-1]

    # Project points onto plane perpendicular to radial axis
    proj_pts = pts - np.dot(pts, radial_axis)[:,None] * radial_axis
    # proj_pts = proj_pts[proj_pts[:,1]**2 + proj_pts[:,2]**2 > 10**2]
    # Fit circle in projected plane to get center
    # For simplicity, approximate by centroid of projected points
    # (you can do real circle fit if needed)
    center_proj = np.median(proj_pts, axis=0)

    # Reconstruct 3D center
    center_3d = center_proj  # already in 3D (projection removes only radial component)
    return center_3d, radial_axis

def pca_align_pointcloud(pcd):
    # Convert to numpy (N,3)
    pts = pcd.point.positions.numpy()

    # 1 Subtract centroid
    # centroid = np.mean(pts, axis=0)
    # pts_centered = pts - centroid
    center, radial_axis = fit_radial_center_circle(pts)
    center[2] = 0.14297444/0.03912
    print(center,center*0.03912)
    pts_centered = pts - center

    proj_pts = pts_centered - np.dot(pts_centered, radial_axis)[:,None] * (radial_axis)

    pca_plane = PCA(n_components=2).fit(proj_pts)
    axis1, axis2 = pca_plane.components_  # two axes in rolling & width plane

    # Construct rotation matrix
    # Columns = desired X, Y, Z
    R = np.column_stack([-radial_axis, axis2, axis1])
    print(R)

    pts_aligned = (R.T @ pts_centered.T).T
    aligned_pcd = o3d.t.geometry.PointCloud()
    aligned_pcd.point.positions = o3d.core.Tensor(pts_aligned)

    # Save or visualize
    # o3d.io.write_point_cloud("tyre_aligned.ply", aligned_pcd)
    

    # # 2 PCA using SVD
    # H = np.dot(pts_centered.T, pts_centered)
    # w, v = np.linalg.eig(H)  # v: eigenvectors (3x3)

    # # Sort eigenvectors by eigenvalues descending
    # idx = np.argsort(w)[::-1]
    # v = v[:, idx]  # Columns = PCA axes

    # # 3 Rotation to world coordinate system
    # R = v.T  # align PCA principal axes to X, Y, Z

    # # 4 Apply transform
    # pts_aligned = (R @ pts_centered.T).T

    # # Create new aligned point cloud
    # aligned_pcd = o3d.geometry.PointCloud()
    # aligned_pcd.points = o3d.utility.Vector3dVector(pts_aligned)

    return aligned_pcd, R, center
radius = 0.4
height = 1.0
resolution = 100  # number of segments around circumference

# 1️⃣ Create cylinder (aligned with Z-axis by default)
cylinder = o3d.geometry.TriangleMesh.create_cylinder(
    radius=radius,
    height=height,
    resolution=resolution
)

# 2️⃣ Rotate from Z-axis → X-axis
# Rotation about Y-axis by -90° (so new axis is X)
R = cylinder.get_rotation_matrix_from_xyz((0, -np.pi / 2,0))
cylinder.rotate(R, center=(0, 0, 0))

# 3️⃣ Optional: center the cylinder at origin
cylinder.translate([-height / 2, 0, 0])

# 4️⃣ Add color for visibility
cylinder.paint_uniform_color([0.2, 0.7, 1.0])
t_cylinder = o3d.t.geometry.TriangleMesh.from_legacy(cylinder)
t_cylinder.compute_vertex_normals()
t_cylinder.compute_triangle_normals()
# 5️⃣ Visualize
# o3d.visualization.draw_geometries([cylinder])

# Usage Example:
pcd_outer = o3d.t.io.read_point_cloud("full_outer_outer_part_only.ply")
pcd_inner = o3d.t.io.read_point_cloud("full_outer_inner_smoothed_part_only.ply")
pcd_outer_copy = pcd_outer.clone()
pcd_inner_copy = pcd_inner.clone()
pcd_outer_copy.scale(scale = 0.0375, center = [0,0,0])
pcd_inner_copy.scale(scale = 0.0375, center = [0,0,0])
pcd_outer_down = pcd_outer.voxel_down_sample(voxel_size=0.3)
pcd_inner_down = pcd_inner.voxel_down_sample(voxel_size=0.3)
#pcd = pcd_outer #pcd_inner.append(pcd_outer)
aligned_pcd_i, R_i, centroid_i = pca_align_pointcloud(pcd_inner_down)
aligned_pcd_o, R_o, centroid_o = pca_align_pointcloud(pcd_outer_down)
aligned_pcd_i.point.colors = pcd_inner.point.colors
aligned_pcd_o.point.colors = pcd_outer.point.colors

Ry = o3d.core.Tensor(np.array([[-1,0,0],[0,1,0],[0,0,-1]]))

pcd_inner.translate(o3d.core.Tensor(-centroid_i))
pcd_inner.rotate(o3d.core.Tensor(R_o.T), center=[0,0,0])
pcd_inner.translate(-o3d.core.Tensor([0,0.0948+0.0355+0.0139 + 0.0001/0.0375, 0.0944-0.0216+0.0109-0.0018/0.0375]))
pcd_inner.rotate(Ry, center = [0,0,0])
pcd_inner.scale(scale = 0.0375, center = [0,0,0])
pcd_outer.translate(o3d.core.Tensor(-centroid_o))
pcd_outer.rotate(o3d.core.Tensor(R_o.T), center=[0,0,0])
pcd_outer.translate(-o3d.core.Tensor([0,0.0948+0.0015+0.0025/0.0375, 0.0944-0.001-0.002/0.0375]))
pcd_outer.rotate(Ry, center = [0,0,0])
pcd_outer.scale(scale = 0.0375, center = [0,0,0])
pcd_inner_down.translate(o3d.core.Tensor(-centroid_i))
pcd_inner_down.rotate(o3d.core.Tensor(R_o.T), center=[0,0,0])
pcd_inner_down.translate(-o3d.core.Tensor([0,0.0948+0.0355+0.0139+ 0.0001/0.0375, 0.0944-0.0216+0.0109-0.0018/0.0375]))
pcd_inner_down.rotate(Ry, center = [0,0,0])
pcd_inner_down.scale(scale = 0.0375, center = [0,0,0])
pcd_outer_down.translate(o3d.core.Tensor(-centroid_o))
pcd_outer_down.rotate(o3d.core.Tensor(R_o.T), center=[0,0,0])
pcd_outer_down.translate(-o3d.core.Tensor([0,0.0948+0.0015+0.0025/0.0375, 0.0944-0.001-0.002/0.0375]))
pcd_outer_down.rotate(Ry, center = [0,0,0])
pcd_outer_down.scale(scale = 0.0375, center = [0,0,0])
o3d.visualization.draw([pcd_outer_copy, pcd_inner_copy, pcd_inner,pcd_outer, t_cylinder])
# .rotate(Ry, center = [0,0,0])


def tyre_radial_profile(pts):
    # Assume pts are aligned and centered: shape (N, 3)
    x = pts[:, 0]  # width or axial coordinate
    y = pts[:, 1]  # lateral
    z = pts[:, 2]  # circumferential
    
    r = np.sqrt(y**2 + z**2)
    theta = np.arctan2(z, y)  # in radians (-π, π)
    
    # Wrap to 0–2π if needed
    theta = np.mod(theta, 2*np.pi)
    
    return x, theta, r

angles = (np.linspace(0,360,18)*(np.pi/180))[:-1] + 0.174
print(angles)

x, theta, r = tyre_radial_profile(pcd_inner.point.positions.numpy())
print(x.shape)
idx_x_slice = np.flatnonzero(np.abs(x-0.0308) < 0.002)
idx_closest = np.argmin(np.abs(theta[idx_x_slice][:, None] - angles[None, :]), axis=0)
print(idx_closest)
lug_pts = pcd_inner.point.positions.numpy()[idx_x_slice[idx_closest]]
lug_radius = r[idx_x_slice[idx_closest]]
lug_theta = theta[idx_x_slice[idx_closest]]
# idx_closest = [18837,1884,21244,17853,16633,15413,13697,13181,4727,2529,4276,3374
#   12290,8444,10649,12185,18498]
print(lug_theta)
print(lug_radius)
plt.scatter(theta, x, c=r, s=1, cmap='viridis')
plt.xlabel("Circumferential angle θ (rad)")
plt.ylabel("Width (x)")
plt.colorbar(label="Radial distance r (m)")
plt.show()



def fit_equal_radius_center(xy):
    # xy: array of shape (N, 2)
    x, y = xy[:, 0], xy[:, 1]
    print(xy)
    def residuals(c):
        xc, yc = c
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        print(r-np.mean(r))
        return r - np.mean(r)

    # Initial guess: centroid
    c0 = np.mean(xy, axis=0)
    print(c0)
    # c0 = np.array([0,0])
    res = least_squares(residuals, c0, method='lm', ftol=1e-15, xtol=1e-15, gtol=1e-15)
    xc, yc = res.x

    # Optional: estimate best-fit radius
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    r_mean = np.mean(r)
    r_std = np.std(r)
    print(r)

    return xc, yc, r_mean, r_std, res

# pts = np.array([
#     [1.0, 0.0],
#     [0.0, 1.0],
#     [-1.0, 0.1],
#     [0.1, -1.1]
# ])
lug_pts = lug_pts.reshape(-1, 3)
print(lug_pts)
# print(lug_pts[0][0][:][0])
xc, yc, r, spread, res = fit_equal_radius_center(lug_pts[:,1:])
print(f"Center: ({xc:.4f}, {yc:.4f}), Radius: {r:.4f}, Spread: {spread:.4e}")

#===============================================================================
# start=np.array([0.0043582423,0.0006682535,0.2466])
# end=np.array([0.0038174386,0.0005853315,0.216])

# print(np.linalg.norm(end-start))

#================================================================

# estimated = np.array([7223.37, 7279.34, 8217.7, 9016.58])
# measured  = np.array([6506.11, 7321.94, 9027.52, 9755.13])

# # Absolute error
# abs_error = np.abs(estimated - measured)

# # Relative error (%)
# rel_error = abs_error / measured * 100

# # RMSE
# rmse = np.sqrt(np.mean((estimated - measured)**2))

# # MAPE
# mape = np.mean(rel_error)

# print("Absolute Error:", abs_error)
# print("Relative Error (%):", rel_error)
# print("RMSE:", rmse)
# print("MAPE (%):", mape)

# data7 = sio.loadmat("contact_patch_points_test_7.mat")
# data8 = sio.loadmat("contact_patch_points_test_8.mat")
# data9 = sio.loadmat("contact_patch_points_test_9.mat")
# data21 = sio.loadmat("contact_patch_points_test_21.mat")

# #boundary = data["boundary_mm_all"]    
# interior7 = data7["interior_mm"] 
# interior8 = data8["interior_mm"] 
# interior9 = data9["interior_mm"] 
# interior21 = data21["interior_mm"] 

# #boundary_3d = np.hstack([boundary, np.zeros((boundary.shape[0], 1))])
# interior_3d7 = np.hstack([interior7, np.zeros((interior7.shape[0], 1))])
# interior_3d8 = np.hstack([interior8, np.zeros((interior8.shape[0], 1))])
# interior_3d9 = np.hstack([interior9, np.zeros((interior9.shape[0], 1))])
# interior_3d21 = np.hstack([interior21, np.zeros((interior21.shape[0], 1))])

# # boundary_pcd = o3d.t.geometry.PointCloud()
# #boundary_pcd.point.positions = o3d.core.Tensor(boundary_3d)
# interior_pcd7 = o3d.t.geometry.PointCloud()
# interior_pcd7.point.positions = o3d.core.Tensor(interior_3d7)
# interior_pcd8 = o3d.t.geometry.PointCloud()
# interior_pcd8.point.positions = o3d.core.Tensor(interior_3d8)
# interior_pcd9 = o3d.t.geometry.PointCloud()
# interior_pcd9.point.positions = o3d.core.Tensor(interior_3d9)
# interior_pcd21 = o3d.t.geometry.PointCloud()
# interior_pcd21.point.positions = o3d.core.Tensor(interior_3d21)

# o3d.visualization.draw([interior_pcd7, interior_pcd8, interior_pcd9, interior_pcd21])





#==================================================================================================

# depth = cv2.imread(r".\DATA\test_9_sine_0_2Hz_3_cycles_80000_right_tyre_d3_c18_2bar\depth\000026.png", cv2.IMREAD_UNCHANGED)
# color = cv2.imread(r".\DATA\test_9_sine_0_2Hz_3_cycles_80000_right_tyre_d3_c18_2bar\color\000026.jpg", cv2.IMREAD_UNCHANGED)
# # ensure it's float for scaling
# color = color.astype(np.float32)/255
# depth = depth.astype(np.float32)/10000

# plt.figure(figsize=(8,6))
# img_c = plt.imshow(color)
# plt.show()
# img_m = plt.imshow(depth, cmap='viridis')


# # normalize depth for visualization (0 → min depth, 1 → max depth)
# depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

# # plot with colormap
# plt.figure(figsize=(8,6))
# img = plt.imshow(depth_norm, cmap='viridis')   # try 'plasma' or 'turbo' for other looks
# cbar = plt.colorbar(img_m, fraction=0.027, pad=0.04)
# cbar.set_label("Depth [m]", rotation=270, labelpad=15)

# plt.axis('off')
# plt.show()

# --- optional: save colorized depth map as PNG ---
#color_mapped = (plt.cm.viridis(depth_norm)[:,:,:3] * 255).astype(np.uint8)
#cv2.imwrite("depth_colored.png", cv2.cvtColor(color_mapped, cv2.COLOR_RGB2BGR))


#===============================================


# box = o3d.geometry.TriangleMesh.create_box()
# cone = o3d.geometry.TriangleMesh.create_cone()
# cone.compute_vertex_normals()
# cone.compute_triangle_normals()
# box.compute_vertex_normals()
# box.compute_triangle_normals()
# box_pcd = box.sample_points_poisson_disk(5000)
# box_pcd.estimate_normals()
# cone_pcd = cone.sample_points_poisson_disk(5000)
# cone_pcd.estimate_normals()
# o3d.visualization.draw_geometries([cone_pcd])
# fig, ax = plt.subplots(figsize=(4, 6))

# # Define colormap and normalization
# cmap = plt.cm.get_cmap('jet')
# norm = plt.Normalize(vmin=-3, vmax=3)

# # Create a normal vertical colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # required for matplotlib < 3.1
# cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
# # Label
# cbar.set_label("Displacement [mm]", labelpad=15)

# plt.show()

#==========================================================

# def elapsed_time_until_frame(time_file, frame_number):
#     """
#     Calculate elapsed time until a given frame number.

#     Parameters:
#         time_file (str): Path to the .npy file containing frame-to-frame times.
#         frame_number (int): The frame index (inclusive) to calculate elapsed time until.

#     Returns:
#         float: Total elapsed time until the given frame.
#     """
#     # Load time differences (between consecutive frames)
#     time_diffs = np.load(time_file) * (0.001)

#     # Ensure frame_number is valid
#     if frame_number < 0 or frame_number > len(time_diffs):
#         raise ValueError(f"Frame number {frame_number} out of range. "
#                          f"Valid range is 0 to {len(time_diffs)}.")

#     # Sum elapsed time up to the frame (exclusive)
#     total_seconds = np.sum(time_diffs[:frame_number])

#     # Convert to minutes and seconds
#     minutes = int(total_seconds // 60)
#     seconds = total_seconds % 60  # keeps fractional seconds

#     return minutes, seconds, total_seconds


# # Example usage
# if __name__ == "__main__":
#     time_file = "./DATA/STTR_test_8_1000mm_min_500kg_10sec_d3_c18_2bar/time.npy"
#     frame_number = 270  # Change this to the frame you want
#     minutes, seconds, total_seconds = elapsed_time_until_frame(time_file, frame_number)
#     print(f"Elapsed time until frame {frame_number}: "
#           f"{minutes} min {seconds:.3f} sec (total {total_seconds:.3f} sec)")

#=======================

# def load_rc_control_points(file_path="./4_row_model_control_points.csv",scale_factor=0.019390745853434508):
#     """
#     Load control points from a CSV file.

#     Args:
#         file_path (str): Path to the CSV file containing control points.
#         scale_factor (float): Scaling factor for the control point coordinates.

#     Returns:
#         tuple: 
#             markers (list): List of marker HEX names from the CSV.
#             m_points (list): List of scaled 3D points (x, y, z).
#             numeric_markers (list): List of hex to dec marker ids from RC.
#     """
#     markers = []
#     numeric_markers = []
#     m_points = []

#     try:
#         with open(file_path, 'r') as file:
#             csv_reader = csv.reader(file)
            
#             # Iterate through rows
#             for row in csv_reader:
#                 #print(row)
#                 markers.append(row[0])
#                 m_points.append([
#                     scale_factor*float(row[1]),
#                     scale_factor*float(row[2]),
#                     scale_factor*float(row[3])
#                 ])  
#                 numeric_markers.append(row[5])

#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {file_path}")

#     return markers, m_points, numeric_markers

# ply_path = 'full_outer_inner_part_only.ply'
# pcd_1 = o3d.t.io.read_point_cloud(ply_path)
# pcd_2 = o3d.t.io.read_point_cloud(ply_path)
# pcd_1.scale(scale = 0.03912, center = [0,0,0])
# pcd_2.scale(scale = 0.03912, center = [0,0,0])

# pcd_3 = o3d.t.io.read_point_cloud('full_outer_treads_part_only.ply')
# pcd_3.scale(scale = 0.03912, center = [0,0,0])
# o3d.visualization.draw([pcd_3])

# markers, m_points, numeric_markers = load_rc_control_points('./full_outer.csv',0.03912)
#print(m_points)

# mean = np.mean(m_points, axis=0)
# print(mean)
# centroid = o3d.core.Tensor(mean) #pcd_1.get_center()
# pcd_2.translate(-centroid)

# april_tag_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
# april_tag_pcd.point.positions = o3d.core.Tensor(m_points)
# april_tag_pcd.translate(-centroid)

# points_centered = april_tag_pcd.point.positions.numpy()  # pcd can be legacy or tensor, just convert

# Covariance matrix
# cov = np.cov(points_centered.T)

# # SVD
# eigvecs, _,_ = np.linalg.svd(cov)

# v1 = eigvecs[:, 0]
# v2 = eigvecs[:, 1]
# v3 = eigvecs[:, 2]

# # Reorder so that v2 becomes the first axis
# reordered_basis = np.column_stack([v3, v1, v2])

# # Rotation from PCA → global
# #R = reordered_basis.T  

# R_pca_to_global = reordered_basis.T
# print(R_pca_to_global)

# pcd_2.rotate(R_pca_to_global, center=(0,0,0))
# april_tag_pcd.rotate(R_pca_to_global, center=(0,0,0))

# plane = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
# #print(pcd_centre.point.positions.shape[0]) 
# #print(VT)
# #print(VT.shape)
# normal = o3d.core.Tensor([1,0,0],dtype=o3d.core.float32).cuda()# VT[-1]
# #print(normal)
# #print(centroid)
# A, B, C = normal 
# #print("A:",A,"B:",B,"C:",C)
# D = 0 # (-normal.mul(centroid)).sum(dim=0)
# #print("D:",D)

# #A*x+B*y+C*z+D = 0
# y = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float32).cuda()
# z = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float32).cuda()
# x = - (B*y + C*z ) / A
# #print((x.append(y,axis = 0)).append(z,axis = 0).T())

# plane.vertex.positions = (x.append(y,axis = 0)).append(z,axis = 0).T()
# plane.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()

# #dist_to_plane = ((masked_pcd.point.positions).matmul(normal) + D).flatten()

# o3d.visualization.draw([april_tag_pcd,pcd_2,plane.cpu()])


# #fit a plane to april tags and with principle coordinate normals

# # fit a plane to t2cam points

# #find angle between planes = slip angles

#================================================================================================================

# from optix_castrays import OptiXRaycaster
# #import kaolin as kao
# import torch
# import torch.nn.functional as F
# s = cp.cuda.Stream()
# with s:
#     cp_start = cp.array([[0.01,0.3,0.45],[0.03,0.6,0.75]])
#     cp_end = cp.array([[0.1,0.4,0.55],[0.5,0.7,0.85]])
#     repeats = 407040//2

#     cp_big_start = cp.tile(cp_start, (repeats, 1)).reshape(480,848,3)
#     cp_big_end = cp.tile(cp_end, (repeats, 1)).reshape(480,848,3)
#     for i in range(20):
#         e1 = cp.cuda.Event()
#         e2 = cp.cuda.Event()
#         e1.record()
#         ray_points = cp.linspace(cp_big_start,cp_big_end,10,True,False,cp.float32,axis = 0)

#         # steps = cp.linspace(0, 1, 10, dtype=cp.float32)  # shape: (10,)
#         # steps = steps[None, :, None]  # shape: (1, 10, 1)
#         # result = (1 - steps) * cp_big_start[:, None, :] + steps * cp_big_end[:, None, :]

#         e2.record()
#         e2.synchronize()
#         t = cp.cuda.get_elapsed_time(e1, e2)
#         print(t)
#         print(ray_points.shape)

# ply_path = 'full_outer_inner_part_only.ply'
# # pcd = o3d.t.io.read_point_cloud(ply_path)
# # pcd = o3d.t.io.read_point_cloud(ply_path)
# # pcd.scale(scale = 0.03912, center = [0,0,0])
# # print("1")
# # pcd.estimate_normals()
# # print("2")
# # pcd.orient_normals_consistent_tangent_plane(k = 10)
# # normals = pcd.point.normals.numpy()
# file_name = 'inner_normals_oriented.npy'
# #np.save(file_name, normals)

# normals = o3d.core.Tensor.load(file_name)
# mesh = o3d.t.io.read_triangle_mesh(ply_path)
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.vertex.normals = normals #pcd.point.normals
# #mesh.compute_vertex_normals()
# print("3")
# mesh.compute_triangle_normals()
# print("4")
# mesh.normalize_normals()
# print("5")

# width = 848
# height = 480
# border = 3 #pixel border
# np_mask_invalid_points = np.ones((height,width), dtype = np.uint8)
# np_mask_invalid_points[border:height-border,border:width-border] = 0
# mask_invalid_points = cv2.cuda.GpuMat(rows = height, cols = width,type = cv2.CV_8U)
# mask_invalid_points.upload(np_mask_invalid_points)

# img = mask_invalid_points.download() * 255
# cv2.imshow("mask",img)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()

# ply_path = 'full_outer_inner_part_only.ply'
# pcd = o3d.t.io.read_point_cloud(ply_path)
# pcd.scale(scale = 0.03912, center = [0,0,0])
# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(k = 10)
# mesh = o3d.t.io.read_triangle_mesh(ply_path)
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.vertex.normals = pcd.point.normals
# #mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()
# mesh.normalize_normals()

# o3d.visualization.draw_geometries([pcd.to_legacy()])

# plane_coords = np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]], dtype=np.float32)
# plane = o3d.t.geometry.TriangleMesh()
# plane.vertex.positions = o3d.core.Tensor.from_numpy(plane_coords)
# plane.triangle.indices = o3d.core.Tensor(np.array([[0,1,2],[1,2,3]]),dtype = o3d.core.int32)
# mesh = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()

# mesh1 = o3d.t.io.read_triangle_mesh("full_outer_inner_part_only.ply")
# mesh1.scale(scale = 0.03912, center = [0,0,0])
# mesh1.compute_vertex_normals()
# mesh1.compute_triangle_normals()
# # o3d.visualization.draw([mesh1, mesh])
# # o3d.visualization.draw([plane, mesh])

# pcd = o3d.t.io.read_point_cloud("full_outer_outer_part_only.ply")
# pcd.estimate_normals()
# print(pcd.covariances)
# pcd = pcd.cuda()
# pcd.scale(scale = 0.03912, center = [0,0,0])



# #pcd = pcd.uniform_down_sample(every_k_points = 10)
# points = torch.utils.dlpack.from_dlpack(pcd.point.positions.contiguous().to_dlpack())

# def split_pointcloud_into_batches(points, batch_size):
#     """
#     Splits a (M, 3) point cloud into batches of shape (B, N, 3).

#     Args:
#         points (torch.Tensor): shape (M, 3)
#         batch_size (int): number of points per batch (N)

#     Returns:
#         torch.Tensor: shape (B, N, 3)
#     """
#     M = points.shape[0]
#     # Truncate M so it's divisible by batch_size
#     M_trunc = (M // batch_size) * batch_size
#     points = points[:M_trunc]

#     # Reshape to (B, N, 3)
#     batched_points = points.view(-1, batch_size, 3)
#     return batched_points

# # Example usage
# #points = torch.randn(100000, 3)  # M = 100,000
# batch_size = 2048*2*2*2*2*2*2*2*2 *2*2         # N = 2048 points per batch

# batched = split_pointcloud_into_batches(points, batch_size)  # shape (B, 2048, 3)
# print(batched.shape) 
# points = points.unsqueeze(0)  # shape: [1, N, 3]

# # Set voxel size
# voxel_size = 0.003  # in meters

# # Compute resolution from bounding box
# min_bound = points.min(dim=1)[0]
# max_bound = points.max(dim=1)[0]
# extent = max_bound - min_bound
# resolution = 128 #int(torch.ceil(extent.max() / voxel_size *extent.max()).item())
# print(resolution)

# def create_3d_sobel_kernels():

#     sobel_x = torch.tensor([
#         [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
#         [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
#         [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
#     ], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)

#     sobel_y = sobel_x.permute(0,1,3,2,4)  # Rotate to Y
#     sobel_z = sobel_x.permute(0,1,4,3,2)  # Rotate to Z

#     return sobel_x, sobel_y, sobel_z

# sobel_x, sobel_y, sobel_z = create_3d_sobel_kernels()
# # Get kernels
# Gx, Gy, Gz = create_3d_sobel_kernels()

# for i in range(10):
#     t1 = time.perf_counter()
#     # spc = kao.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(pointcloud=points,
#     #                                                                     level=11,
#     #                                                                     )
#     # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(leg_pcd,
#     #                                                             voxel_size=0.05)
#     vox = kao.ops.conversions.pointclouds_to_voxelgrids(batched,
#                                                         resolution=resolution,
#                                                         origin=None,
#                                                         scale=None,
#                                                         return_sparse=False
#                                                         )   
    
#     # Dummy voxel data
#     voxel = vox

    

#     # Convolve
#     grad_x = F.conv3d(voxel, Gx, padding=1)
#     grad_y = F.conv3d(voxel, Gy, padding=1)
#     grad_z = F.conv3d(voxel, Gz, padding=1)

#     # Gradient magnitude
#     grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

#     print("Gradient magnitude shape:", grad_mag.shape)

#     t2 = time.perf_counter()
#     #print(vox)
#     # print(f'SPC keeps track of the following cells in levels of detail (parents + leaves):\n'
#     #       f' {spc.point_hierarchies}\n')
#     print("Voxel time:", t2-t1)
#     threshold = grad_mag.mean() + 2 * grad_mag.std() # tune this threshold
#     mask = grad_mag.squeeze() > threshold

#     coords = torch.nonzero(mask).float()  # shape (K, 3)

#     # Convert to Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())

#     o3d.visualization.draw_geometries([pcd])
# #o3d.visualization.draw_geometries([voxel_grid])
# num_partitions = pcd.pca_partition(max_points=40000)

# # print the partition ids and the number of points for each of them.
# print(np.unique(pcd.point.partition_ids.numpy(), return_counts=True))

# pcd1 = pcd.select_by_mask(pcd.point.partition_ids == 1)
# pcd2 = pcd.select_by_mask(pcd.point.partition_ids == 2)
# pcd3 = pcd.select_by_mask(pcd.point.partition_ids == 3)
# pcd4 = pcd.select_by_mask(pcd.point.partition_ids == 4)
# pcd5 = pcd.select_by_mask(pcd.point.partition_ids == 160)
# pcd6 = pcd.select_by_mask(pcd.point.partition_ids == 161)
# pcd7 = pcd.select_by_mask(pcd.point.partition_ids == 162)
# pcd8 = pcd.select_by_mask(pcd.point.partition_ids == 163)

# o3d.visualization.draw([pcd1,pcd2,pcd3,pcd4,pcd5,pcd6,pcd7,pcd8])

# # 1. Initialize once (outside your frame loop)
# raycaster = OptiXRaycaster("full_outer_inner_part_only.ply","full_outer_outer_part_only.ply", "raycast.cu", stream=cp.cuda.Stream())
# scale = 0.03912
# # outer = o3d.t.io.read_triangle_mesh("full_outer_inner_part_only.ply")
# # outer.scale(scale = scale, center = [0,0,0])
# # outer.compute_vertex_normals()
# # origins = outer.vertex.positions.numpy()
# # directions = outer.vertex.normals.numpy()
# outer = o3d.t.io.read_point_cloud("full_outer_inner_part_only.ply")
# mesh = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()
# o3d.visualization.draw_geometries([mesh.to_legacy()])
# outer.scale(scale = scale, center = [0,0,0])
# outer.estimate_normals()
# #outer.orient_normals_consistent_tangent_plane(k = 10)

# outer = outer.cuda()
# origins = cp.from_dlpack(outer.point.positions.to_dlpack())
# directions = cp.from_dlpack(outer.point.normals.to_dlpack())
# # origins = np.asarray(outer.points)
# # directions = -np.asarray(outer.normals)
# # o3d.visualization.draw_geometries([outer,mesh.to_legacy()])

# rays = cp.concatenate([origins, -directions], axis=1)
# print(rays)
# hit_point, tri_id, t_hit = raycaster.cast(rays)
# print(hit_point)
# print(tri_id)
# print(t_hit)

# r_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
# r_pcd.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack())
# o3d.visualization.draw([r_pcd.cpu()])
# 2. In your frame loop

# imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
# while imageStream.has_next():
#     start_cv2 = cv2.getTickCount()
#     if imageStream.has_next():
#         count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu= imageStream.get_next_frame()
#         origins = t2cam_pcd_cuda.point.positions.cpu().numpy()
#         directions = t2cam_pcd_cuda.point.normals.cpu().numpy()
#         rays = np.concatenate([origins, directions], axis=1)  # (N, 6)
#         hit_point, tri_id, t_hit = raycaster.cast(rays)
#     #t2cam_pcd.to_legacy()
#     end_cv2 = cv2.getTickCount()
#     time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     print("FPS:", time_sec)


# def load_mesh_from_ply(path):
#     mesh = o3d.io.read_triangle_mesh(path)
#     mesh.compute_vertex_normals()
#     vertices = np.asarray(mesh.vertices, dtype=np.float32)
#     triangles = np.asarray(mesh.triangles, dtype=np.uint32)
#     return vertices, triangles


# def prepare_rays(origins, directions):
#     n = origins.shape[0]
#     rays = np.zeros(n, dtype=optix.RayHit.dtype)
#     rays['origin'] = origins
#     rays['direction'] = directions
#     rays['tmin'] = 0.0
#     rays['tmax'] = 1e10
#     return rays


# def raycast_with_optix(vertices, triangles, rays):
#     ctx = optix.DeviceContext()

#     # Acceleration structure (BVH)
#     gas = optix.GeometryAccelerationStructure(ctx)
#     gas.set_triangles(vertices, triangles)
#     gas.build()

#     # Minimal raygen, miss, and hit programs
#     ptx = optix.utils.minimal_ptx()
#     module = ctx.create_module(ptx)
#     raygen = module.create_raygen_program("__raygen__minimal")
#     miss = module.create_miss_program("__miss__default")
#     hit = module.create_hitgroup_program("__closesthit__default")

#     pipeline = ctx.create_pipeline([raygen], miss, [hit])

#     # Output buffer
#     hit_results = np.zeros(len(rays), dtype=np.float32)

#     # Launch
#     sbt = pipeline.create_shader_binding_table()
#     pipeline.launch(sbt, rays, output=hit_results, width=len(rays), height=1)

#     return hit_results


# if __name__ == "__main__":
#     # Load mesh
#     mesh_path = "your_mesh.ply"
#     vertices, triangles = load_mesh_from_ply(mesh_path)

#     # Generate 400,000 rays (pointing along Z)
#     N = 400_000
#     origins = np.random.uniform(-1, 1, size=(N, 3)).astype(np.float32)
#     directions = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (N, 1))

#     rays = prepare_rays(origins, directions)

#     # Run OptiX raycasting
#     hit_dists = raycast_with_optix(vertices, triangles, rays)

#     # Inspect results
#     print("Hits:", np.count_nonzero(np.isfinite(hit_dists)))
#     print("Closest hit distance:", hit_dists[np.isfinite(hit_dists)].min())

# file_name = 'inner_to_outer_correspondences.npy'
# with open(file_name, 'rb') as f:
#     rays_hit_start_io = np.load(f)
#     rays_hit_end_io = np.load(f)

# model_pcd = o3d.t.geometry.PointCloud()
# model_pcd.point.positions = o3d.core.Tensor(rays_hit_start_io)

# scale = 0.03912
# file_path = 'full_outer_inner_part_only.ply'
# mesh = o3d.io.read_triangle_mesh(filename=file_path, print_progress = True)
# mesh.scale(scale = scale, center = [0,0,0])

# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(mesh)

# imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
# while imageStream.has_next():
#     start_cv2 = cv2.getTickCount()
#     if imageStream.has_next():
#         count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu= imageStream.get_next_frame()
#     #t2cam_pcd.to_legacy()
#     end_cv2 = cv2.getTickCount()
#     time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     print("FPS:", time_sec)

# intrinsic = o3d.core.Tensor([[433.4320983886719,0,416.3398132324219],[0,432.88079833984375,238.8269500732422],[0,0,1]]).cuda()
# #print(np.asarray(intrinsic))
# intrinsic_nt = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
# #print(intrinsic_nt)

# if imageStream.has_next():
#     count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
# if imageStream.has_next():
#     count1, depth_image1, color_image1, rgbd_image1, t2cam_pcd1 = imageStream.get_next_frame()

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image).cuda()
# print(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image).cuda()
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cpu = pcd.cpu()
# #pcd.estimate_normals() #bad for gpu
# #pcd.orient_normals_consistent_tangent_plane(k=10)
# #pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.clone()
# pcd2.translate(o3d.core.Tensor([0.01,0.1,0.01]))
# pcd2.estimate_normals()
# t1 = time.time()
# #filtered_depth = depth_o3d.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
# vertex_map = depth_o3d.create_vertex_map(intrinsic)
# normal_map = vertex_map.create_normal_map()
# print((normal_map.as_tensor().cpu().numpy())[240,422]) #480, 848
# #o3d.visualization.draw([pcd,pcd2])
# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # metric_params = o3d.t.geometry.MetricParameters()
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.ChamferDistance],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# p = o3d.t.geometry.PointCloud()
# p.point.positions = vertex_map.as_tensor().cpu().reshape((-1, 3))
# p.point.normals = normal_map.as_tensor().cpu().reshape((-1, 3))
# # o3d.visualization.draw([p])

# legacy_pcd = p.to_legacy()
# o3d.visualization.draw_geometries([legacy_pcd], point_show_normal=True)

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image)
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cuda = pcd.cuda()
# pcd.uniform_down_sample(every_k_points=10)
# #pcd.estimate_normals()
# pcd2 = pcd.translate([0.01,0.1,0.01])
# pcd2.estimate_normals()
# t1 = time.time()


# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # pcd2 = pcd.translate([0.1,0.1,0.1])
# # metric_params = o3d.t.geometry.MetricParameters(fscore_radius=o3d.utility.DoubleVector((0.01, 0.11, 0.15, 0.18)))
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.FScore],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# t0 = time.time()
# depth_o3d = o3d.geometry.Image(depth_image)
# color_o3d = o3d.geometry.Image(color_image)
# RGBD = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d,depth_o3d,10000)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic_nt)
# #pcd.estimate_normals()
# pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.translate([0.1,0.1,0.1])
# pcd2.estimate_normals()
# t1 = time.time()

# # reg_res = o3d.pipelines.registration.registration_icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1.000000e-08, max_iteration=1000))
# # print(reg_res)
# t2 = time.time()

# #print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

