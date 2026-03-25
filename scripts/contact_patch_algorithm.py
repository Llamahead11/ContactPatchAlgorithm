import cv2
import open3d as o3d
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
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
#matplotlib.use("Agg")
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from contact_patch.capture_realsense_tensor import RealSenseManager
from contact_patch.replay_realsense_tensor import read_RGB_D_folder
from contact_patch.Dense_Opt_Flow import DenseOptFlow
from contact_patch.Sparse_Opt_Flow import SparseOptFlow
from contact_patch.app_vis import Viewer3D
from contact_patch.optix_castrays import OptiXRaycaster
# import vidVisualiser as vV
import time
import yaml
from scipy.spatial import cKDTree
from numba import njit
import os
import cupy as cp
import sys
import copy
import nvtx
import threading
import scipy.io as sio
from scipy.io import savemat

plt.ioff() 

def load_rc_control_points(file_path="./4_row_model_control_points.csv",scale_factor=0.019390745853434508):
    """
    Load control points from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing control points.
        scale_factor (float): Scaling factor for the control point coordinates.

    Returns:
        tuple: 
            markers (list): List of marker HEX names from the CSV.
            m_points (list): List of scaled 3D points (x, y, z).
            numeric_markers (list): List of hex to dec marker ids from RC.
    """
    markers = []
    numeric_markers = []
    m_points = []

    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            
            # Iterate through rows
            for row in csv_reader:
                print(row)
                markers.append(row[0])
                m_points.append([
                    scale_factor*float(row[1]) - (0.42841208)*scale_factor,#0.02758715*(scale_factor/0.03912),
                    scale_factor*float(row[2]) - (-1.6929364)*scale_factor,#0.07112041*(scale_factor/0.03912),
                    scale_factor*float(row[3]) - (3.6547658)*scale_factor,#0.14297444*(scale_factor/0.03912)
                ])  
                numeric_markers.append(row[5])

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    centroid = np.array([0.42841208,-1.6929364,3.6547658 ])*scale_factor
    R = np.array([[ -0.03135062 , 0.35852575 , 0.93299323], #should not have the third column negative
    [-0.03143202 , 0.9326368 , -0.35944495],
    [ -0.9990141 , -0.04059484 , -0.01796954]]).T
    # pcd.translate(o3d.core.Tensor(-np.array([0,0.0948+0.0355+0.0139, 0.0944-0.0216+0.0109])))
    
    # centroid = [0.02758715, -0.07112041, 0.14297444]
    # R = np.array([[2.52815128e-02, 3.33760291e-02, 9.99123058e-01], 
    #     [-7.85843857e-01, 6.18424477e-01, -7.73910520e-04], 
    #     [-6.17907985e-01, -7.85135152e-01,  4.18630537e-02]])
    
    #0.9247612953 0.3793223500 0.0305141602 0.5241848230
    # -0.3801788390 0.9244197011 0.0302038081 -0.0352334976
    # -0.0167509131 -0.0395321511 0.9990778565 -0.0503222942
    # 0.0000000000 0.0000000000 0.0000000000 1.0000000000
    
    m_points = np.array(m_points)

    m_points = m_points @ R.T
    m_points = m_points - np.array([0,0.0948+0.0355+0.0139, 0.0944-0.0216+0.0109])*scale_factor 
    m_points = m_points @ np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T
    m_points = m_points.tolist()

    return markers, m_points, numeric_markers

def convert_rc_apriltag_hex_ids(file_path="./assets/RCtoTag.csv"):
    """
    Converts the AprilTag Ids from Reality Capture Model 
    corresponding to the conventional IDs. 
    Id 0 to 587 conversion 

    Args:
         file_path (str): Path to the CSV file containing RC apriltag IDs.
    
    Returns:
        tuple:
            normalTag (list): 0-587 - Correct April Tag Convention list
            RCcorTag (list): hex2demical list of april tags from RC
    """
    normalTag = []
    RCcorTag = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            
            # Iterate through rows
            for row in csv_reader:
                print(row)
                RCcorTag.append(row[0])
                normalTag.append(row[2])
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return normalTag, RCcorTag

def convert_rc_control_points(normalTag, numeric_markers):
    """
    Find correct convention Apriltag IDs in RC 3D inner tyre model

    Args:
        normalTag (list): corresponding correct 0 - 587 list 
        numeric_markers (list): hex2dec list of RC IDs in model 

    Returns:
        correctTags (list): conventional AprilTag IDs in the 3D model 
    """ 
    correctTags = []; 
    for n_m in numeric_markers:
        c_t = normalTag[int(n_m)-1]
        correctTags.append(c_t)
    
    return correctTags

def load_model_pcd(file_path="./4_row_model/4_row_model_HighPoly_Smoothed.ply", scale = 0.019390745853434508):
    '''
    Load Inner Tyre Model Point Cloud and Estimate Normals

    Args:
        file_path (str): Path to the CSV file containing control points.
        scale (float): Scaling factor for the control point coordinates.
    
    Returns:
        pcd (o3d.geometry.PointCloud): Inner Tyre Model Point Cloud
    '''
    pcd = o3d.io.read_point_cloud(filename=file_path, format = 'auto',remove_nan_points=True, remove_infinite_points=True, print_progress = True)
    pcd.scale(scale = scale, center = [0,0,0])
    # centroid = np.array([0.02758715, -0.07112041, 0.14297444])*(scale/0.03912)
    centroid = np.array([0.42841208,-1.6929364,3.6547658 ])*scale
    pcd.translate(-centroid)
    R = np.array([[-0.03135062 , 0.35852575 , 0.93299323],
    [-0.03143202 , 0.9326368 , -0.35944495],
    [ -0.9990141 , -0.04059484 , -0.01796954]]).T
    pcd.rotate(R, center = [0,0,0])
    pcd.translate(-np.array([0,0.0948+0.0355+0.0139, 0.0944-0.0216+0.0109])*scale)
    pcd.rotate(np.array([[-1,0,0],[0,1,0],[0,0,-1]]), center = [0,0,0])
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    tensor_pcd.normalize_normals()
    return tensor_pcd


def load_model_ply(file_path="./4_row_model/4_row_model_HighPoly_Smoothed.ply", scale = 0.019390745853434508):
    '''
    Load Inner Tyre Model Point Cloud and Estimate Normals

    Args:
        file_path (str): Path to the CSV file containing control points.
        scale (float): Scaling factor for the control point coordinates.
    
    Returns:
        pcd (o3d.geometry.PointCloud): Inner Tyre Model Point Cloud
    '''
    mesh = o3d.io.read_triangle_mesh(filename=file_path, print_progress = True)
    mesh.scale(scale = scale, center = [0,0,0])
    centroid = np.array([0.42841208,-1.6929364,3.6547658 ])*scale
    mesh.translate(-centroid)
    R = np.array([[ -0.03135062 , 0.35852575 , 0.93299323],
    [-0.03143202 , 0.9326368 , -0.35944495],
    [ -0.9990141 , -0.04059484 , -0.01796954]]).T
    # centroid = [0.02758715, -0.07112041, 0.14297444]
    # 
    # R = [[2.52815128e-02, 3.33760291e-02, 9.99123058e-01], 
    #     [-7.85843857e-01, 6.18424477e-01, -7.73910520e-04], 
    #     [-6.17907985e-01, -7.85135152e-01,  4.18630537e-02]]
    mesh.rotate(R, center = [0,0,0])
    mesh.translate(-np.array([0,0.0948+0.0355+0.0139, 0.0944-0.0216+0.0109])*scale)
    mesh.rotate(np.array([[-1,0,0],[0,1,0],[0,0,-1]]), center = [0,0,0])
    return mesh

def find_tag_point_ID_correspondence(pcd,m_points):
    """
    Find point and normal at the center of each identified marker

    Args:
        pcd (o3d.geometry.PointCloud): inner tyre model point cloud
        m_points (list): List of (x, y, z) points at the center of each Tag in the model.

    Returns:
        Tuple:
            tag_norm (numpy.ndarray): a NumPy array of shape (N,3) containing [x,y,z] normals for each tag
            model_correspondences (list): a list of Point IDs Correspondences on the model for each tag
    """
    print("Finding tag-point correspondences in model")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd.cpu().to_legacy())
    tag_norm = []
    model_correspondence = []
    for tag in range(0,len(m_points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(m_points[tag],1)
        model_correspondence.append(idx[0])
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        m_norm = pcd.point.normals.cpu().numpy()[idx[1:], :]
        m_norm_ave = np.mean(m_norm, axis=0)
        tag_norm.append(m_norm_ave)
    print("Completed")

    tag_norm = np.array(tag_norm)    

    return tag_norm, model_correspondence

def draw_lines(points, normals):
    """
    Makes a set of lines using open3d

    Args:
        points (list): A list of (x, y, z) starting points
        normals (list): A list of (x, y, z) directional normals

    Returns:
        line_set (o3d.geometry.LineSet): A o3d object containing a set of lines 
    """
    normal_length = 0.1
    line_starts = points
    line_ends = points + normals * normal_length

    lines = [[i, i + len(points)] for i in range(len(points))]
    line_points = np.vstack((line_starts, line_ends))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
    return line_set

def vis_window(*geometries):
    """
    Visualise Statically any amount of geometries
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window() 

def find_t2cam_correspondence(t2cam_cuda,tag_locations,debug_mode=True):
    """
    Function to find t2cam point IDs from tag locations 
    using np.where

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): The PointCloud captured by t2cam
        tag_locations (list): list of tag [x,y,z] locations

    Returns:
        indices (o3d.core.Tensor)(numpy.ndarray) A NumPy array of correspondence Point IDs
    """

    # indices = []
    # for xyz in tag_locations:
    #     idx = np.where((t2cam_cuda.point.positions.cpu().numpy() == xyz).all(axis=1))[0]
    #     if debug_mode: print(t2cam_cuda.point.positions[idx],xyz)
    #     indices.append(idx)

    # return o3d.core.Tensor.from_numpy(np.array(indices).flatten()).cuda()
    t2cam_tensor = t2cam_cuda.point.positions
    tags_tensor = o3d.core.Tensor(tag_locations, dtype = o3d.core.float32 ,device=o3d.core.Device("CUDA:0"))

    tags_reshaped = tags_tensor.reshape((-1, 1, 3))
    t2cam_reshaped = t2cam_tensor.reshape((1, -1, 3))

    # Compute squared distances: [M, N]
    diff = tags_reshaped - t2cam_reshaped
    dist2 = diff * diff
    dist2_sum = dist2.sum(2)
    
    # Get index of nearest point in t2cam for each tag location
    nearest_indices_gpu = dist2_sum.argmin(1)

    if debug_mode:
        nearest_points = t2cam_tensor[nearest_indices_gpu]
        for i in range(len(tag_locations)):
            print(f"Tag: {tag_locations[i]}, Match: {nearest_points[i].cpu().numpy()}")

    return nearest_indices_gpu

def make_correspondence_vector(t2cam_correspondence,model_correspondence,tag_IDs,correctTags,debug_mode):
    """
    Determines Correspondence Vector between T2cam Point Cloud and Inner Model

    Args:
        t2cam_correspondence (numpy.ndarray): a NumPy array of t2cam Point ID correspondence
        model_correspondence (list): a list of inner tyre model point ID correspondence
        tag_IDs (list): a list of detected tag IDs in t2Cam Point Cloud
        correctTags (list): a list of tag IDs in the inner tyre model 
        debug_mode (boolean): a boolean whether debug mode is enabled

    Returns:
        corres (o3d.utility.Vector2iVector): Correspondence array for registration 
    """
    str_tags = [str(e) for e in tag_IDs]
    p = t2cam_correspondence
    q = cp.array([model_correspondence[correctTags.index(tag)] for tag in str_tags], dtype=cp.int64)
    n = p.shape[0]
    corr = cp.arange(n, dtype=np.int64)
    dl_corr = corr.toDlpack()
    corres = o3d.core.Tensor.from_dlpack(dl_corr)
    dl_q = q.toDlpack()
    q_tensor = o3d.core.Tensor.from_dlpack(dl_q)
    return corres, p, q_tensor

def register_t2cam_with_model(t2cam_pcd_cuda, tracked_t2cam, model_pcd_cuda,corres_vector,p,q, curr_t):
    """
    Function that transforms the T2Cam point cloud to register with the inner tyre model point cloud

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): open3d Point Cloud of T2Cam 
        model_pcd (o3d.geometry.PointCloud): open3d Point Cloud of inner tyre model
        corres_vector (o3d.utility.Vector2iVector): Correspondence array for registration 
    """
    T = o3d.core.Tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],dtype=o3d.core.float32).cuda()
    invT = o3d.core.Tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],dtype=o3d.core.float32).cuda()
    alpha = 19.2*(np.pi/180) #14.2
    theta  = np.arctan2((0.07254*np.sin(alpha) + 0.0393*np.cos(alpha)),(0.2162+(0.07254*np.cos(alpha))-0.0393*np.sin(alpha)))
    x_translate = 0#+0.009
    y_translate = 0.2162 + 0.07254 #0.2162+(0.07254*np.cos(alpha))-0.0393*np.sin(alpha) # 0.2162 + 0.07254 #rotate about  [0,0.2162,0]
    z_translate = 0.0393 #0.07254*np.sin(alpha) + 0.0393*np.cos(alpha) # 0.0393 
    xyz_translate = np.array([x_translate,y_translate,z_translate]) #+ np.array([0,0.01,0])
    xyz_tensor = o3d.core.Tensor(xyz_translate,dtype = o3d.core.float32).cuda()

    #xyz_tensor
    Rx = np.array([[1, 0, 0], 
        [0, np.cos(alpha), -np.sin(alpha)], 
        [0, np.sin(alpha),  np.cos(alpha)]])
    Rx_tensor = o3d.core.Tensor(Rx,dtype = o3d.core.float32).cuda()
    # print(xyz_tensor, Rx_tensor)


    estimator = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    icp_t2cam = t2cam_pcd_cuda.select_by_index(p)
    
    icp_model = model_pcd_cuda.select_by_index(q)
    
    # # mask = t2cam_pcd_cuda.point.positions[:,2] > 0.07
    # # t2cam_pcd_cuda = t2cam_pcd_cuda.select_by_mask(mask)
    # o3d.visualization.draw([t2cam_pcd_cuda,model_pcd_cuda,icp_t2cam,icp_model])
    # t2cam_pcd_cuda.transform(T)
    # icp_t2cam.transform(T)
   
    # t2cam_pcd_cuda.rotate(Rx_tensor, center=[0,0,0])
    # t2cam_pcd_cuda.translate(xyz_tensor)
    tracked_t2cam.translate(xyz_tensor)
    tracked_t2cam.rotate(Rx_tensor, center=[0,0.2162,0])
    
    icp_t2cam.translate(xyz_tensor)
    icp_t2cam.rotate(Rx_tensor, center=[0,0.2162,0])
    initial = icp_t2cam.point.positions[[1,-1],:].cpu().numpy()
    final = icp_model.point.positions[[1,-1],:].cpu().numpy()
    # print(initial)
    # print(final)
    T_est = estimator.compute_transformation(icp_t2cam,icp_model,corres_vector)
    # print(T_est)
    # gamma = np.arctan2(T_est[2,1].cpu().numpy(), T_est[2,2].cpu().numpy()) #-np.arccos(T_est[2,2].cpu().numpy())
    gamma = np.arctan2(np.sum(initial[:,1]*final[:,2]-initial[:,2]*final[:,1]),np.sum(initial[:,1]*final[:,1]+initial[:,2]*final[:,2]))
    Rx_gamma = o3d.core.Tensor(np.array([[1, 0, 0], 
        [0, np.cos(gamma), -np.sin(gamma)], 
        [0, np.sin(gamma),  np.cos(gamma)]]),dtype = o3d.core.float32).cuda()
    # t2cam_pcd_cuda.rotate(Rx_gamma, center=[0,0,0])
    tracked_t2cam.rotate(Rx_gamma, center=[0,0,0])
    icp_t2cam.rotate(Rx_gamma, center=[0,0,0])
    # # print(T_est)
    # tracked_t2cam.translate(T_est[:3,3])
    # icp_t2cam.translate(T_est[:3,3])
    cam_pos = [0,0.2162+(0.07254*np.cos(alpha))-0.0393*np.sin(alpha),0.07254*np.sin(alpha) + 0.0393*np.cos(alpha)] 
    t1 = o3d.core.Tensor([[0], [0.07254], [0.0393]], dtype=o3d.core.float32).cuda()
    t2 = o3d.core.Tensor([[0], [0.2162], [0]], dtype=o3d.core.float32).cuda()
    print((Rx_tensor.matmul(t1)).add(t2))
    T[:3,3] = (Rx_gamma.matmul(Rx_tensor.matmul(t1).add(t2))).flatten()
    T[:3,:3] = Rx_gamma.matmul(Rx_tensor)
    invT[:3,:3] = (Rx_gamma.matmul(Rx_tensor)).T()
    invT[:3,3] = ((-t1).add(-Rx_tensor.T().matmul(t2))).flatten()

    # T[:3,3] = T[:3,3] + T_est[:3,3].cuda()
    # o3d.visualization.draw([tracked_t2cam,model_pcd_cuda,icp_t2cam,icp_model])
    #T without the y and z rotation, but include the x rotation?
    # if tracked_t2cam != 1:
    #     tracked_t2cam.transform(T)
        #mesh.transform(T)

        # plane = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
        # normal_p = o3d.core.Tensor([1,0,0],dtype=o3d.core.float32).cuda()
        # A_p, B_p, C_p = normal_p #D = 0

        # #A*x+B*y+C*z+D = 0
        # y_p = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float32).cuda()
        # z_p = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float32).cuda()
        # x_p = - (B_p*y_p + C_p*z_p ) / A_p

        # plane.vertex.positions = (x_p.append(y_p,axis = 0)).append(z_p,axis = 0).T()
        # plane.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()

        # dist_to_plane_twist = ((icp_t2cam.point.positions).matmul(normal_p)).flatten()
        # dist_to_plane_model = ((icp_model.point.positions).matmul(normal_p)).flatten()
        # mask_slice_twist = (dist_to_plane_twist > 0) & (dist_to_plane_twist < 0.015)
        # mask_slice_model = (dist_to_plane_model > 0) & (dist_to_plane_model < 0.015)
    
        # centroid = icp_t2cam.select_by_mask(mask_slice_twist).get_center()
        # pcd_centre = icp_t2cam.select_by_mask(mask_slice_twist).clone()
        # pcd_centre.translate(-centroid)

        # plane_twist = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
        # if pcd_centre.point.positions.shape[0] != 0:
        #     U,S,VT = pcd_centre.point.positions.svd()
        #     normal = VT[-1]
        #     A, B, C = normal
        #     D = (-normal.mul(centroid)).sum(dim=0)

        #     #A*x+B*y+C*z+D = 0
        #     y = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float32).cuda()
        #     z = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float32).cuda()
        #     x = - (B*y + C*z + D) / A
            
        #     plane_twist.vertex.positions = (x.append(y,axis = 0)).append(z,axis = 0).T()
        #     plane_twist.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()

        # centroid_m = icp_model.select_by_mask(mask_slice_model).get_center()
        # pcd_centre_m = icp_model.select_by_mask(mask_slice_model).clone()
        # pcd_centre_m.translate(-centroid_m)

        # plane_model = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
        # if pcd_centre_m.point.positions.shape[0] != 0:
        #     Um,Sm,VTm = pcd_centre_m.point.positions.svd()
        #     normal_m = VTm[-1]
        #     A_m, B_m, C_m = normal_m
        #     D_m = (-normal_m.mul(centroid_m)).sum(dim=0)

        #     #A*x+B*y+C*z+D = 0
        #     y_m = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float32).cuda()
        #     z_m = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float32).cuda()
        #     x_m = - (B_m*y_m + C_m*z_m + D_m) / A_m
            
        #     plane_model.vertex.positions = (x_m.append(y_m,axis = 0)).append(z_m,axis = 0).T()
        #     plane_model.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()
        
        # print("t2cam",normal)
        # print("model", normal_m)

        # n1 = normal.cpu().numpy()
        # n2 = normal_m.cpu().numpy()
        # dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
        # angle = np.arccos(dot)
        # print(angle)
        # if np.isclose(angle, 0.0):
        #     return np.eye(3)     # already aligned
        # if np.isclose(angle, np.pi):
        #     # 180-degree rotation: axis is ambiguous. pick any orth perpendicular to n1.
        #     # find a vector orthogonal to n1:
        #     orth = np.array([1.0, 0.0, 0.0])
        #     if np.allclose(np.abs(n1), orth):
        #         orth = np.array([0.0, 1.0, 0.0])
        #     axis = np.cross(n1, orth)
        #     axis = normalize(axis)
        #     return rotation_matrix_from_axis_angle(axis, np.pi)
        # axis = np.cross(n1, n2)
        # return rotation_matrix_from_axis_angle(axis, angle)
        # o3d.visualization.draw([icp_t2cam, icp_model, plane_twist.cpu(), plane_model.cpu()])
        #o3d.visualization.draw([icp_t2cam, icp_model])
    print(T)
    return T.cpu(), invT.cpu()
    
def draw_registration_result(source, target,frame):
    '''
    Function used for Visualising the Source and Target PointCloud (Called before and after registration for debugging)
    
    Args:
        source (o3d.geometry.PointCloud): open3d Point Cloud that is transformed to align with source
        target (o3d.geometry.PointCloud): open3d Point Cloud that is used as the reference to align to
        frame (o3d.geometry.TriangleMesh): open3d Coordinate Axis showing the midpoint w.r.t. all axis 
    '''
    source_temp = source
    target_temp = target
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw([source_temp, target_temp,frame])

def segment_pcd_using_bounding_box(t2cam_pcd_cuda, model_pcd_cuda):
    """
    Segment the model point cloud to contain that section of t2Cam point cloud
    calculate bounding box of t2cam_pcd and crop model according to it

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): A Point Cloud of T2Cam image
        model_pcd (o3d.geometry.PointCloud): A Point Cloud of inner tyre model

    Returns:
        cropped_model (o3dd.geometry.PointCloud): a Point Cloud consisting of t2cam bounds
    """
    #mask = (t2cam_pcd_cuda.point.positions > 0.01).all(dim=1)
    
    #o3d.visualization.draw([t2cam_pcd_cuda,t2cam_pcd_cuda.select_by_mask(mask)])
    t2cam_bounding_box = t2cam_pcd_cuda.get_axis_aligned_bounding_box()
    cropped_model_cuda = model_pcd_cuda.crop(t2cam_bounding_box)
    # o3d.visualization.draw([t2cam_pcd_cuda.cpu(),cropped_model_cuda.cpu()])
    #cropped_model = cropped_model_cuda.cpu()
    return cropped_model_cuda

def pca_normal(points):
    """
    Compute PCA normal of a 3D point cloud.
    points: (N,3) array
    Returns: normal vector (unit length)
    """
    pts = np.asarray(points)
    centroid = pts.mean(axis=0)
    X = pts - centroid

    # covariance
    C = np.cov(X.T)

    # eigen decomposition
    evals, evecs = np.linalg.eigh(C)  # eigh since C is symmetric
    idx = np.argsort(evals)           # ascending order

    # eigenvector with smallest eigenvalue = normal
    normal = evecs[:, idx[0]]

    # normalize
    normal /= np.linalg.norm(normal)

    if np.dot(normal, centroid) < 0:
        normal = -normal

    return normal

def ICP_register_T2Cam_with_model(p,q,model_pcd_cuda,t2cam_pcd_cuda,d_t2_points,cropped_model_cu, draw_reg):
    '''
    Performs Iterative Closest Point Registration to the Undeformed Cropped Inner Model and Deformed Inner T2Cam PCD

    Args:
        cropped_model (o3d.geometry.PointCloud): open3d Point Cloud Undeformed Cropped Inner Model
        t2cam_pcd (o3d.geometry.PointCloud): open3d Point Cloud 
        draw_reg (bool): A boolean to visualise the registration before and after result
    '''

    #find centre of pointcloud 
    dl_d_t2_points = d_t2_points.to_dlpack()
    cp_d_t2_points = cp.from_dlpack(dl_d_t2_points)
    cp_model_points = cp.from_dlpack(cropped_model_cu.point.positions.to_dlpack())
    cp_model_normals = cp.from_dlpack(cropped_model_cu.point.normals.to_dlpack())

    mid_xyz = (cp.ptp(cp_d_t2_points,axis=0)/2 + cp.min(cp_d_t2_points,axis=0))# + cp.array([0,0.04,0.04])
    mid_xyz_model = (cp.ptp(cp_model_points,axis=0)/2 + cp.min(cp_model_points,axis=0))
    dl_mid_xyz = mid_xyz.toDlpack()
    o3d_mid_xyz = o3d.core.Tensor.from_dlpack(dl_mid_xyz)
    
    # create mask to select the outer points in the pcd to avoid registration with the deformed parts
    mask_min = ((cp_d_t2_points[:,0]-mid_xyz[0])**2+(cp_d_t2_points[:,1]-mid_xyz[1])**2+((cp_d_t2_points[:,2]-mid_xyz[2])/1.2)**2) > 0.14**2
    mask_max = ((cp_d_t2_points[:,0]-mid_xyz[0])**2+(cp_d_t2_points[:,1]-mid_xyz[1])**2+((cp_d_t2_points[:,2]-mid_xyz[2])/1.2)**2) < 0.17**2
    mask_model_min = ((cp_model_points[:,0]-mid_xyz_model[0])**2+(cp_model_points[:,1]-mid_xyz_model[1])**2+((cp_model_points[:,2]-mid_xyz_model[2])/1.2)**2) > 0.14**2
    mask_model_max = ((cp_model_points[:,0]-mid_xyz_model[0])**2+(cp_model_points[:,1]-mid_xyz_model[1])**2+((cp_model_points[:,2]-mid_xyz_model[2])/1.2)**2) < 0.17**2
    points_removed_def = cp_d_t2_points[mask_min & mask_max]
    points_model_removed = cp_model_points[mask_model_min & mask_model_max] 
    normals_model_removed = cp_model_normals[mask_model_min & mask_model_max]

    dl_points_removed_def = points_removed_def.toDlpack()
    dl_point_model_removed = points_model_removed.toDlpack()
    dl_normal_model_removed = normals_model_removed.toDlpack()

    t2cam_removed_def = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
    t2cam_removed_def.point.positions = o3d.core.Tensor.from_dlpack(dl_points_removed_def) #o3d.core.Tensor(downsampled_points_removed_def, device = o3d.core.Device("CUDA:0")) 
    model_points_removed = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
    model_points_removed.point.positions = o3d.core.Tensor.from_dlpack(dl_point_model_removed)
    model_points_removed.point.normals = o3d.core.Tensor.from_dlpack(dl_normal_model_removed)
    #o3d.visualization.draw([t2cam_removed_def,model_points_removed])

    t_s_reg = time.time()
    reg_p2p = o3d.t.pipelines.registration.icp(source = t2cam_pcd_cuda.uniform_down_sample(every_k_points=10),
                                                target = cropped_model_cu.uniform_down_sample(every_k_points=10),
                                                max_correspondence_distance = 0.001,
                                                estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                                                criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-12, relative_rmse=1.000000e-12, max_iteration=500))
    t_e_reg = time.time()
    # print("REG:",t_e_reg-t_s_reg)
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # t2cam_pcd_cuda.transform(reg_p2p.transformation)
    icp_t2cam = t2cam_pcd_cuda.select_by_index(p)
    icp_model = model_pcd_cuda.select_by_index(q)
    
    # n = pca_normal(t2cam_pcd_cuda.uniform_down_sample(100).point.positions.cpu().numpy())
    # reg_p2p.transformation[:3, 3] += o3d.core.Tensor(-n*0.013)
    # t2cam_pcd_cuda.translate(o3d.core.Tensor(-n*0.013).cuda())

    #create lookup table with 21 to 31mm and loadand subtract max def and apply translation to it
    #take change in distance of center of contact patch from depth map and minus max def and translate that amount
    # then calbration factor everytime you put in the camera

    #do pose estimation on undeformed tyre and get the pose from icp that alignes , do it for many images and get the traj of camera
    #use that pose to get the radius from center of tyre and angle , and then find a way to get the exact location in where the camera is
    # or use the global reg to get the estimate , get the estimated angle and then use transfrom to 
    
    # o3d.visualization.draw([t2cam_pcd_cuda,cropped_model_cu,icp_t2cam, icp_model,t2cam_removed_def.cpu(), model_points_removed.cpu()])
    #t2cam_pcd_cuda.point.positions = t2cam_pcd_cuda.point.positions.add(o3d.core.Tensor([0,0.01,-0.01],dtype=o3d.core.float32).cuda())
    if draw_reg: 
        frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1,device=o3d.core.Device("CUDA:0"))
        frame.translate(o3d_mid_xyz)
        draw_registration_result(t2cam_pcd_cuda,cropped_model_cu,frame)
    
    #mesh.transform(reg_p2p.transformation)
    if draw_reg: draw_registration_result(t2cam_pcd_cuda,cropped_model_cu,frame)
    return reg_p2p.transformation

def mesh_indices_ref():
    triangles = []
    H = 480
    W = 848
    for i in range(H - 1):
        for j in range(W - 1):
            idx = i * W + j
            v0 = idx
            v1 = idx + 1
            v2 = idx + W
            v3 = idx + W + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)
    return triangles

def inner_deformed_to_outer_deformed(p,q,invT,rough_T,stream_o3d_cp,stream_ray,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,t2cam_pcd_cuda,count,model_ply,draw_reg):
    """
    Function to get the SIGNED corressponding distances between 
    the T2Cam Point Cloud and the Inner Tyre Model

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): A Point Cloud of T2Cam image
        model_pcd (o3d.geometry.PointCloud): A Point Cloud of inner tyre model
        draw_reg (bool): Boolean Flag to Visualise ICP Registration

    Returns:
        t2_d_pcd (o3d.geometry.PointCloud): Returns the T2Cam point Cloud with a colormap applied representing deformed distances
        max_d_dist (int): Max scalar distance 
        downsampled_cropped_model (o3d.geometry.PointCloud): Returns downsampled cropped model 
        d_dist (np.ndarray): A NumPy array containing scalar distances for each point in the t2_d_pcd
    """
    with stream_o3d_cp:
        event_segment_and_fine_align = cp.cuda.Event()
        start_cv2 = cv2.getTickCount()
        ## crop the undeformed model to the t2cam pcd size
        cropped_model_cuda = segment_pcd_using_bounding_box(t2cam_pcd_cuda,model_pcd_cuda)
        end_cv2 = cv2.getTickCount()
        time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
        #print("SEGMENT:", time_sec)
        t_s_begin = time.time()
        
        ## register using ICP for better alignment
        fine_T = 1 #ICP_register_T2Cam_with_model(p,q,model_pcd_cuda,t2cam_pcd_cuda,t2cam_pcd_cuda.point.positions,cropped_model_cuda,draw_reg)
        t_e_begin = time.time()
        event_segment_and_fine_align.record(stream_o3d_cp)
        #print("BEGIN:", t_e_begin-t_s_begin)

        full_T = rough_T#fine_T.matmul(rough_T)
        inv_full_T = invT #full_T.inv()

    stream_ray.wait_event(event_segment_and_fine_align)

    ## OptiX ray tracing engine
    start_cv21 = cv2.getTickCount()
    with stream_ray:
        origins = cp.from_dlpack(t2cam_pcd_cuda.point.positions.contiguous().to_dlpack())
        directions = cp.from_dlpack(t2cam_pcd_cuda.point.normals.contiguous().to_dlpack())
        rays = cp.concatenate([origins, directions], axis=1) 

        hit_point, tri_id, t_hit, hit_point_o, tri_id_o, t_hit_o, hit_point_t, tri_id_t, t_hit_t = raycaster.cast(rays)

        # stream_ray.synchronize()
        # hp = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        # print("hp")
        # hp.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack())
        # print("pos")
        # o3d.visualization.draw_geometries([hp.cpu().to_legacy()])
        # print("vis")

        #save hit_point, hit_point_o, origins
        # np.savez_compressed(f"saved_arrays/iteration_{count:03d}.npz",
        #     orig=cp.asnumpy(origins),
        #     hit_p=cp.asnumpy(hit_point),
        #     hit_p_o=cp.asnumpy(hit_point_o),
        #     inv_T = inv_full_T.numpy()
        # )
        
        #print(hit_point_o)
        end_cv21 = cv2.getTickCount()
        time_sec = (end_cv21-start_cv21)/cv2.getTickFrequency()
        
        #print("OptiX: ", time_sec)
    
        # stream_ray.synchronize()
        # cp.cuda.runtime.deviceSynchronize()
        # cp.cuda.Device().synchronize()

        # r_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        # r_pcd.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack())
        # o3d.visualization.draw([r_pcd.cpu()])

        # print("hit_point_t shape", hit_point_t.shape)
        # print("hit_point shape", hit_point.shape)
        # print("t_hit shape", t_hit_t.shape)

        ## vector distance
        t_s = time.time()
        # dist = d_model_points - d_t2_points[idx]
        dist = hit_point - origins
        # cp_dist = cp.asarray(dist)
        ## scalar distance
        d_dist = t_hit
        #print(d_dist.min(), d_dist.max())
        d_dist_o = t_hit_o
        #print(d_dist_o[d_dist_o>=0.0].min(), d_dist_o.max())
        d_dist_t = t_hit_t
        # plt.figure(figsize=(6, 4))
        # plt.hist(t_hit.get(), bins=500, color='steelblue', edgecolor='black')
        # plt.xlabel("Distance to fitted plane (m)")
        # plt.ylabel("Number of points")
        # plt.title("Histogram of distances to plane")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # plt.show(block=True)
        #print(d_dist_t[d_dist_t>=0.0].min(), d_dist_t.max())
        #d_dist = cp.linalg.norm(dist, axis=1)

        ## color correspodnece to signed distance according to 'plasma'
        # d_colors = plt.get_cmap('plasma')(((d_dist - d_dist.min()) / (d_dist.max() - d_dist.min())).get())
        # d_colors = d_colors[:, :3]

        # d_colors_o = plt.get_cmap('plasma')(((d_dist_o - d_dist_o[d_dist_o>=0.0].min()) / (d_dist_o.max() - d_dist_o[d_dist_o>=0.0].min())).get())
        # d_colors_o = d_colors_o[:, :3]

        #d_colors_t = plt.get_cmap('plasma')(((d_dist_t - d_dist_t[d_dist_t>=0.0].min()) / (d_dist_t.max() - d_dist_t[d_dist_o>=0.0].min())).get())
        # d_colors_t = plt.get_cmap('plasma')(((d_dist_t - (-0.05)) / (0.05 - (-0.05))).get())
        # d_colors_t = d_colors_t[:, :3]

        bins = np.linspace(-0.035, 0, 11)

        # Digitize into bin indices
        bin_indices = np.digitize(d_dist.get(), bins) - 1
        #bin_indices = np.clip(bin_indices, 0, len(bins)-2)

        # Create colormap
        cmap = plt.get_cmap('plasma', len(bins)-1)  # discrete version
        d_colors_t = cmap(bin_indices / (len(bins)-2))[:, :3] 

        # mask_mid = (np.abs(d_model_points[:,2]-0.09) < 0.001) | (np.abs(d_model_points[:,2]-0.13) < 0.001) | (np.abs(d_model_points[:,2]-0.17) < 0.001)
        # mask_mid_1 = (np.abs(d_t2_points[:,2]-0.09) < 0.001) | (np.abs(d_t2_points[:,2]-0.13) < 0.001) | (np.abs(d_t2_points[:,2]-0.17) < 0.001) 
        # mask_lat = (np.abs(d_model_points[:,0]-0.04) < 0.001) | (np.abs(d_model_points[:,0]-0.0) < 0.001) | (np.abs(d_model_points[:,0]+0.04) < 0.001)
        # mask_lat_1 = (np.abs(d_t2_points[:,0]-0.04) < 0.001) | (np.abs(d_t2_points[:,0]-0.0) < 0.001) | (np.abs(d_t2_points[:,0]+0.04) < 0.001)
        # inner_arr.append(d_t2_points[mask_mid_1])
        # inner_lat.append(d_t2_points[mask_lat_1])
        # inner_undef.append(d_model_points[mask_mid])
        # inner_undef_lat.append(d_model_points[mask_lat])
        # t2_d_pcd_def = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        # t2_d_pcd_def.point.positions = downsampled_t2cam_cu.point.positions[mask_lat]
        # t2_d_pcd_def.paint_uniform_color([1,1,0])

        # Create new pcd with color correspondence to deformation
        # t2_d_pcd_cu = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        # t2_d_pcd_cu.point.positions = downsampled_cropped_model_cu.point.positions[mask_mid | mask_lat]
        # t2_d_pcd_cu.point.colors = o3d.core.Tensor(d_colors[mask_mid | mask_lat]).cuda()
        outer_def = (hit_point_o - dist).astype(np.float32)
        tread_def = (hit_point_t - dist).astype(np.float32)
        
        t2_d_pcd_cu = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        #t2_d_pcd_cu.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack()) #downsampled_cropped_model_cu.point.positions
        t2_d_pcd_cu.point.positions = t2cam_pcd_cuda.point.positions
        t2_d_pcd_cu.point.normals = t2cam_pcd_cuda.point.normals
        #t2_d_pcd_cu.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()
        
        # t2_d_pcd_cu.estimate_normals()
        # t2_d_pcd_cu.orient_normals_consistent_tangent_plane(k=10)
        t2_und_in = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und_in.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack()) #downsampled_cropped_model_cu.point.positions
        #t2_und_in.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()
        #t2_und_in = t2_und_in.uniform_down_sample(every_k_points = 10)
        
        
        t2_d_pcd_cu_o = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_d_pcd_cu_o.point.positions = o3d.core.Tensor.from_dlpack(outer_def.toDlpack()) #downsampled_cropped_model_cu.point.positions
        #t2_d_pcd_cu_o.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()

        
        t2_d_pcd_cu_t = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_d_pcd_cu_t.point.positions = o3d.core.Tensor.from_dlpack(tread_def.toDlpack()) #downsampled_cropped_model_cu.point.positions
        t2_d_pcd_cu_t.point.colors = o3d.core.Tensor(d_colors_t, dtype = o3d.core.float32).cuda()

        t2_und = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und.point.positions = o3d.core.Tensor.from_dlpack(hit_point_o.toDlpack()) #downsampled_cropped_model_cu.point.positions
        # t2_und.point.colors = o3d.core.Tensor(d_colors_o, dtype = o3d.core.float32).cuda()
        #t2_und = t2_und.uniform_down_sample(every_k_points = 10)

        t2_und_t = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und_t.point.positions = o3d.core.Tensor.from_dlpack(hit_point_t.toDlpack()) #downsampled_cropped_model_cu.point.positions
        # t2_und_t.point.colors = o3d.core.Tensor(d_colors_t, dtype = o3d.core.float32).cuda()
        #t2_und_t = t2_und_t.uniform_down_sample(every_k_points = 10)

        # o3d.visualization.draw([t2_d_pcd_cu_o.cpu().to_legacy(),t2_d_pcd_cu.to_legacy()])



        # # Current Config (Deformed Configuration)
        # # linspace of origins to outer_def

        # volumetric_def_points = cp.linspace(origins,outer_def,10,True,False,cp.float32,axis=0)
        # print("volumetric_def_points shape", volumetric_def_points.shape)

        # # Reference Config (Undeformed Configuration)
        # #linspace of hit_point to hit_point_o
        # volumetric_undef_points = cp.linspace(hit_point,hit_point_o,10,True,False,cp.float32,axis=0)
        # print("volumetric_def_points shape", volumetric_undef_points.shape)

        # #Deformation Gradient F using Finite Differences Method
        # volumetric_def_points = volumetric_def_points.reshape(10,480,848,3)
        # volumetric_undef_points = volumetric_undef_points.reshape(10,480,848,3)

        # dx = cp.empty((10,480,848),cp.float32)
        # dy = cp.empty((10,480,848),cp.float32)
        # dz = cp.empty((10,480,848),cp.float32)
        # dX = cp.empty((10,480,848),cp.float32)
        # dY = cp.empty((10,480,848),cp.float32)
        # dZ = cp.empty((10,480,848),cp.float32)

        # ##Foward Difference
        # # dx[:,:,:-1] = volumetric_def_points[:,:,1:,0] - volumetric_def_points[:,:,0:-1,0]
        # # dy[:,:-1,:] = volumetric_def_points[:,1:,:,1] - volumetric_def_points[:,0:-1,:,1]
        # # dz[:-1,:,:] = volumetric_def_points[1:,:,:,2] - volumetric_def_points[0:-1,:,:,2]

        # # dX[:,:,:-1] = volumetric_undef_points[:,:,1:,0] - volumetric_undef_points[:,:,0:-1,0]
        # # dY[:,:-1,:] = volumetric_undef_points[:,1:,:,1] - volumetric_undef_points[:,0:-1,:,1]
        # # dZ[:-1,:,:] = volumetric_undef_points[1:,:,:,2] - volumetric_undef_points[0:-1,:,:,2]

        # ## Central Difference
        # dx[:,:,1:-1] = volumetric_def_points[:,:,2:,0] - volumetric_def_points[:,:,0:-2,0]
        # dy[:,1:-1,:] = volumetric_def_points[:,2:,:,1] - volumetric_def_points[:,0:-2,:,1]
        # dz[1:-1,:,:] = volumetric_def_points[2:,:,:,2] - volumetric_def_points[0:-2,:,:,2]

        # dX[:,:,1:-1] = volumetric_undef_points[:,:,2:,0] - volumetric_undef_points[:,:,0:-2,0]
        # dY[:,1:-1,:] = volumetric_undef_points[:,2:,:,1] - volumetric_undef_points[:,0:-2,:,1]
        # dZ[1:-1,:,:] = volumetric_undef_points[2:,:,:,2] - volumetric_undef_points[0:-2,:,:,2]

        # dx[:,:,0] = volumetric_def_points[:,:,1,0] - volumetric_def_points[:,:,0,0]
        # dy[:,0,:] = volumetric_def_points[:,1,:,1] - volumetric_def_points[:,0,:,1]
        # dz[0,:,:] = volumetric_def_points[1,:,:,2] - volumetric_def_points[0,:,:,2]

        # dX[:,:,0] = volumetric_undef_points[:,:,1,0] - volumetric_undef_points[:,:,0,0]
        # dY[:,0,:] = volumetric_undef_points[:,1,:,1] - volumetric_undef_points[:,0,:,1]
        # dZ[0,:,:] = volumetric_undef_points[1,:,:,2] - volumetric_undef_points[0,:,:,2]

        # dx[:,:,-1] = volumetric_def_points[:,:,-1,0] - volumetric_def_points[:,:,-2,0]
        # dy[:,-1,:] = volumetric_def_points[:,-1,:,1] - volumetric_def_points[:,-2,:,1]
        # dz[-1,:,:] = volumetric_def_points[-1,:,:,2] - volumetric_def_points[-2,:,:,2]

        # dX[:,:,-1] = volumetric_undef_points[:,:,-1,0] - volumetric_undef_points[:,:,-2,0]
        # dY[:,-1,:] = volumetric_undef_points[:,-1,:,1] - volumetric_undef_points[:,-2,:,1]
        # dZ[-1,:,:] = volumetric_undef_points[-1,:,:,2] - volumetric_undef_points[-2,:,:,2]

        # F = cp.stack([
        #     cp.stack([dx/dX, dx/dY, dx/dZ], axis=-1),
        #     cp.stack([dy/dX, dy/dY, dy/dZ], axis=-1),
        #     cp.stack([dz/dX, dz/dY, dz/dZ], axis=-1),
        #     ], axis=-2)
        # print(F.shape) #(10,480,848,3,3)

        # # del dx, dy, dz, dX, dY, dZ
        # # del volumetric_def_points, volumetric_undef_points

        # F_T = F.transpose(0, 1, 2, 4, 3)
        # C = cp.matmul(F_T, F)
        # #del F
        # I = cp.eye(3, dtype=cp.float32)[None, None, None, :, :]
        # E = 0.5*(C-I) 


        #o3d_normals = model_ply.triangle.normals[tri_id_t]
        # o3d_idx = o3d.core.Tensor.from_dlpack(tri_id_t.toDlpack())
        # new = model_ply.triangle.normals[o3d_idx]

    # indices = mesh_indices_ref()
    # indices = cp.asarray(indices)
    # mask_idx = cp.where(tread_def[:,2] > 0.05)[0]
    # mask = cp.isin(indices, mask_idx) 
    # mask = cp.all(mask, axis=1) 
    # indices = indices[mask]
    # mesh_t2_def = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
    # mesh_t2_def.vertex.positions = o3d.core.Tensor.from_dlpack(tread_def.toDlpack())
    # mesh_t2_def.triangle.indices = o3d.core.Tensor.from_dlpack(indices.toDlpack())
    
    
    # o3d.visualization.draw([mesh_t2_def])

    #cp_normals = cp.from_dlpack(new.contiguous().to_dlpack())
    # cp_norms = cp.linalg.norm(cp_normals, axis=1, keepdims=True)
    # cp_normals = cp_normals/ cp.where(cp_norms == 0, 1e-8, cp_norms) 
    #dist_tread = cp.sum((hit_point_t-tread_def)*cp_normals, axis = 1) #cp.linalg.norm(tread_def,axis=1)

    with stream_ray:
        #valid_mask_cp = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1)).get() #& (t_hit < -0.006)

        dl_d_t2_points = t2cam_pcd_cuda.point.positions.to_dlpack()
        cp_d_t2_points = cp.from_dlpack(dl_d_t2_points)
        mid_xyz = (cp.ptp(cp_d_t2_points,axis=0)/2 + cp.min(cp_d_t2_points,axis=0))
        
        #print(mid_xyz)
        # dl_mid_xyz = mid_xyz.toDlpack()
        # o3d_mid_xyz = o3d.core.Tensor.from_dlpack(dl_mid_xyz)
        # frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1,device=o3d.core.Device("CUDA:0"))
        # frame.translate(o3d_mid_xyz)
        # create mask to select the outer points in the pcd to avoid registration with the deformed parts
        radius_mask = ((cp_d_t2_points[:,0]-mid_xyz[0])**2+(cp_d_t2_points[:,1]-mid_xyz[1])**2+((cp_d_t2_points[:,2]-mid_xyz[2])/1.2)**2) < 0.15**2
        # large_def_xyz = cp_d_t2_points[cp.argmax(t_hit[radius_mask])] 
        # large_def_radius_mask = ((cp_d_t2_points[:,0]-large_def_xyz[0])**2+(cp_d_t2_points[:,1]-large_def_xyz[1])**2+((cp_d_t2_points[:,2]-large_def_xyz[2])/1.2)**2) < 0.09**2
        #valid_mask_cp_threshold = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & radius_mask).get()

        zero_def = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & (t_hit > -0.008) & (t_hit < -0.007))
        valid_mask_radius_cp = (tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & radius_mask

        # plt.figure(figsize=(6, 4))
        # plt.hist(t_hit.get()[valid_mask_radius_cp.get()], bins=500, color='steelblue', edgecolor='black')
        # plt.xlabel("Distance to fitted plane (m)")
        # plt.ylabel("Number of points")
        # plt.title("Histogram of distances to plane")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # plt.show(block=True)

        max_stable = cp.percentile(t_hit[valid_mask_radius_cp], 20)
        min_stable = cp.percentile(t_hit[valid_mask_radius_cp], 5)
        # print(min_stable, max_stable)
        mask_smallest = (t_hit < max_stable) & valid_mask_radius_cp & (t_hit > min_stable)
        max_def = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & (t_hit < -0.0243))
        # print(t_hit[max_def][t_hit[max_def] < -0.0243])
        #tread_def_dist and not t_hit
        #histogram of valid t_hits

        # plt.figure(figsize=(6, 4))
        # plt.hist((t_hit[max_def][t_hit[max_def] < -0.0243]).get(), bins=500, color='steelblue', edgecolor='black')
        # plt.xlabel("Distance to fitted plane (m)")
        # plt.ylabel("Number of points")
        # plt.title("Histogram of distances to plane")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # plt.show(block=True)

    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(t_hit_t - t_hit)[valid_mask_radius_cp.get()], bins=500, color='steelblue', edgecolor='black')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of distances to plane")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show(block=True)

    # print(t_hit_t.shape, t_hit.shape)
    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(dist_tread[valid_mask_cp]), bins=500, color='steelblue', edgecolor='black')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of distances to plane")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show(block=False)

    

    # cp.cuda.runtime.deviceSynchronize()
    # cp.cuda.Device().synchronize()
    with stream_ray:
        valid_mask = (o3d.core.Tensor.from_dlpack(tri_id_t.toDlpack()) != 0) & (o3d.core.Tensor.from_dlpack(hit_point_t.toDlpack()) != 0.0).all(dim=1) #& (o3d.core.Tensor.from_dlpack(t_hit.toDlpack()) != -1) # & (o3d.core.Tensor.from_dlpack(t_hit.toDlpack()) < -0.007) #& (o3d.core.Tensor.from_dlpack(dist_tread.toDlpack()) > -0.00025)    # & (o3d.core.Tensor.from_dlpack((t_hit_t).toDlpack()) > 0.05) #& (o3d.core.Tensor.from_dlpack((t_hit_t - t_hit).toDlpack()) < 0.08) 
        #o3d.visualization.draw_geometries([t2_und.cpu().to_legacy()])
        #o3d.visualization.draw_geometries([t2_und_t.cpu().to_legacy(), t2_d_pcd_cu_t.cpu().to_legacy()])
        # o3d.visualization.draw_geometries([t2_d_pcd_cu_t.cpu().to_legacy()])
        # o3d.visualization.draw_geometries([t2_d_pcd_cu_o.cpu().to_legacy()])
        # o3d.visualization.draw_geometries([t2_und_t.cpu().to_legacy()])
        valid_mask_threshold = (o3d.core.Tensor.from_dlpack(tri_id_t.toDlpack()) != 0) & (o3d.core.Tensor.from_dlpack(hit_point_t.toDlpack()) != 0.0).all(dim = 1) & (o3d.core.Tensor.from_dlpack((radius_mask.astype(cp.uint8)).toDlpack()) != 0)
        curr_valid_mask = valid_mask #& prev_valid_mask
        
        valid_zero_def = (o3d.core.Tensor.from_dlpack(tri_id_t.toDlpack()) != 0) & (o3d.core.Tensor.from_dlpack(hit_point_t.toDlpack()) != 0.0).all(dim = 1) & (o3d.core.Tensor.from_dlpack((zero_def.astype(cp.uint8)).toDlpack()) != 0)
        #print(t2_d_pcd_cu_t.point.positions.shape)
        masked_radius_pcd = t2_d_pcd_cu_t.select_by_mask(valid_mask_threshold)
        masked_pcd = t2_d_pcd_cu_t.select_by_mask(curr_valid_mask) #.clone() #.voxel_down_sample(voxel_size = 0.002) 
        #contact_patch = masked_pcd
        extreme_def = t2_d_pcd_cu_t.select_by_mask(o3d.core.Tensor.from_dlpack((mask_smallest.astype(cp.uint8)).toDlpack()) != 0)
        validation_def_pcd = t2_d_pcd_cu_t.select_by_mask(o3d.core.Tensor.from_dlpack((max_def.astype(cp.uint8)).toDlpack()) != 0)
        
        #print(masked_pcd.point.positions.shape)
    #o3d.visualization.draw_geometries([masked_pcd.cpu().to_legacy()])
        t1 = time.perf_counter()
        #masked_pcd.estimate_normals() #.orient_normals_consistent_tangent_plane(k=10)
        t2 = time.perf_counter()
        #masked_pcd.orient_normals_consistent_tangent_plane(k=3)
        t3 = time.perf_counter()
    #masked_pcd.cuda()
    #print("one", t2-t1)
    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(t_hit)[valid_mask_cp], bins=500, color='steelblue', edgecolor='black')
    # plt.hist(cp.asnumpy(t_hit)[valid_mask_cp_threshold], bins=500, color='orange', edgecolor='red')
    # plt.hist(cp.asnumpy(t_hit)[mask_smallest.get()], bins=500, color='blue', edgecolor='blue')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.xlim([-0.020,0.010])
    # plt.ylim([0,1400])
    # plt.title("Histogram of distances")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show(block = False)
    # # plt.savefig(f"hist/{count:04d}.png")
    # # plt.close()
    #print("two", t3-t2)
    
    #o3d.visualization.draw([masked_radius_pcd.cpu(), masked_pcd.cpu(), t2_und_t.cpu()])
    # Get normals (Nx3)
    # normals = masked_pcd.point.normals.cpu().numpy()

    # # Convert to spherical coordinates
    # xn, yn, zn = normals[:, 0], normals[:, 1], normals[:, 2]
    # theta = np.arctan2(yn, xn)            # azimuth [-pi, pi]
    # phi = np.arccos(np.clip(zn, -1, 1))  # elevation [0, pi]

    # # Create 2D histogram
    # hist, xedges, yedges = np.histogram2d(theta, phi, bins=72)

    # #Plot histogram (azimuth vs elevation)
    # # plt.imshow(hist.T, origin='lower', aspect='auto',
    # #         extent=[-np.pi, np.pi, 0, np.pi])
    # # plt.xlabel("Azimuth θ")
    # # plt.ylabel("Elevation φ")
    # # plt.title("Normal Orientation Histogram")
    # # plt.colorbar(label="Count")
    # # plt.show()

    # ix, iy = np.unravel_index(np.argmax(hist), hist.shape)
    # # Initialize the valid mask with False
    # valid_mask = np.zeros_like(hist, dtype=bool)

    # # Clip to image boundaries to avoid out-of-bounds errors
    # x_start, x_end = max(ix - 1, 0), min(ix + 2, hist.shape[0])
    # y_start, y_end = max(iy - 1, 0), min(iy + 2, hist.shape[1])

    # # Set the 3×3 neighborhood to True
    # valid_mask[x_start:x_end, y_start:y_end] = True

    # # Get bin indices for each point
    # xidx = np.searchsorted(xedges, theta, side='right') - 1
    # yidx = np.searchsorted(yedges, phi, side='right') - 1

    # # Clamp indices to valid range
    # xidx = np.clip(xidx, 0, 72 - 1)
    # yidx = np.clip(yidx, 0, 72 - 1)

    # # Use mask to filter points
    # mask = valid_mask[xidx, yidx]
    # filtered_points = normals[mask]

    # # Visualize or save filtered point cloud
    # filtered_pcd = masked_pcd.select_by_mask(o3d.core.Tensor(mask).cuda())
    # #o3d.visualization.draw_geometries([filtered_pcd.cpu().to_legacy()])

    # # curv = calculate_surface_curvature(masked_pcd)
    # # print(curv.shape)

    # # plt.figure(figsize=(6, 4))
    # # plt.hist(curv, bins=500, color='steelblue', edgecolor='black')
    # # plt.xlabel("Distance to fitted plane (m)")
    # # plt.ylabel("Number of points")
    # # plt.title("Histogram of distances to plane")
    # # plt.grid(True)
    # # plt.tight_layout()
    # # plt.show()

    # # curv_colors_t = plt.get_cmap('plasma')((curv - curv.min()) / (curv[curv < 0.005].max() - curv.min()))
    # # curv_colors_t = curv_colors_t[:, :3]

    # # masked_pcd.point.colors = o3d.core.Tensor(curv_colors_t).cuda()
    # # o3d.visualization.draw_geometries([masked_pcd.cpu().to_legacy()])

    centroid = extreme_def.get_center()
    pcd_centre = extreme_def.clone()
    pcd_centre.translate(-centroid)

    plane = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
    #print(pcd_centre.point.positions.shape[0]) 
    if pcd_centre.point.positions.shape[0] != 0:
        U,S,VT = pcd_centre.point.positions.svd()
        #print(VT)
        #print(VT.shape)
        normal = VT[-1]
        #print(normal)
        #print(centroid)
        A, B, C = normal
        #print("A:",A,"B:",B,"C:",C)
        D = (-normal.mul(centroid)).sum(dim=0)
        #print("D:",D)

        #A*x+B*y+C*z+D = 0
        x = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float32).cuda()
        z = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float32).cuda()
        y = - (A*x + C*z + D) / B
        #print((x.append(y,axis = 0)).append(z,axis = 0).T())
        
        plane.vertex.positions = (x.append(y,axis = 0)).append(z,axis = 0).T()
        plane.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()
        
        dist_to_plane = ((t2_d_pcd_cu_t.point.positions).matmul(normal) + D).flatten()
        contact_patch_mask = dist_to_plane.abs() < 0.002

        contact_patch = t2_d_pcd_cu_t.select_by_mask(contact_patch_mask)

    #     n = normal.cpu().numpy().astype(np.float32)
    #     z_axis = np.array([0,0,1], dtype=np.float32)

    #     if np.dot(n, z_axis) > 0:
    #         n = -n

    #     v = np.cross(n, z_axis)
    #     c = np.dot(n, z_axis)
    #     if np.linalg.norm(v) < 1e-8:  # already aligned
    #         R = np.eye(3)
    #     else:
    #         vx = np.array([[0, -v[2], v[1]],
    #                     [v[2], 0, -v[0]],
    #                     [-v[1], v[0], 0]], dtype=np.float32)
    #         R = np.eye(3) + vx + vx @ vx * (1/(1+c))
    #     #contact_patch.point.positions = contact_patch.point.positions - (dist_to_plane[mask].reshape((-1, 1))).mul(normal.reshape((1, 3)))
    #     contact_patch.rotate(o3d.core.Tensor(R, dtype=o3d.core.float32).cuda(),center=[0,0,0])
    #    # o3d.visualization.draw([contact_patch.cpu()])
    #     contact_patch.point.positions[:,2] -= contact_patch.point.positions[:,2]

    #     source_ds = contact_patch.voxel_down_sample(voxel_size=0.002)
    #     contact_patch.translate(-source_ds.point.positions.mean(dim=0))
    #     contact_patch.translate(o3d.core.Tensor([0,0,0.001]))

        
    #save hit_point, hit_point_o, origins
    # np.savez(f"D:/stored_arrays/Rolling_back_cleat_test_24/iteration_{count:03d}.npz",
    #     orig=cp.asnumpy(origins),
    #     hit_p=cp.asnumpy(hit_point),
    #     hit_p_o=cp.asnumpy(hit_point_o),
    #     inv_T = inv_full_T.numpy(),
    #     Ap = A.cpu().numpy(),
    #     Bp = B.cpu().numpy(),
    #     Cp = C.cpu().numpy(),
    #     Dp = D.cpu().numpy(),
    #     cent = centroid.cpu().numpy()
    # )

    #o3d.visualization.draw([plane,masked_pcd,t2_und_t])
    # plane_model, inliers = masked_pcd.segment_plane(distance_threshold=0.001,
    #                                  ransac_n=300,
    #                                  num_iterations=1000)
    # print(plane_model)
    # inlier_cloud = masked_pcd.select_by_index(inliers)
    # inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = masked_pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw([inlier_cloud, outlier_cloud,t2_und_t])
    
    # t2_d_pcd_cu_o = t2_d_pcd_cu_o.cpu()
    # t2_d_pcd_cu_o.estimate_normals()
    # t2_d_pcd_cu_o.orient_normals_consistent_tangent_plane(k=3)
    # o3d.visualization.draw_geometries([t2_d_pcd_cu_o.cpu().to_legacy(), t2_und.cpu().to_legacy()])
    
    t_e = time.time()
    #print("REST: ",t_e-t_s)
    ## DEBUG Visualisation
    # o3d.visualization.draw_geometries([t2_d_pcd])
    # o3d.visualization.draw_geometries([downsampled_cropped_model,downsampled_t2cam])

    ## If want orthographic 3 graphs to be saved
    # figInner = vV.makeOrthoDeformationPlot(label = 'Inner', points=np.asarray(t2_d_pcd.points), dist = d_dist,vmin=-0.003,vmax=0.003)
    # cpu_dist = d_dist.get()
    # mask = (cpu_dist > -0.02) & (cpu_dist < 0.02)
    ## Save Figure 
    t__0 = time.perf_counter()
    # figInner,sc = vV.createOuterDeformationPlot(label = 'Tread Deformation under 4000N Vertical Load', points=t2_d_pcd_cu_t.point.positions[valid_mask].cpu().numpy(), dist = d_dist.get()[valid_mask.cpu().numpy()],vmin=-0.03,vmax=0)
    # plt.savefig("./Tread_Deformation/{}.png".format(count))
    # plt.show()
    #save_quickly(figInner, "fast_image.png")
    #plt.close(figInner)
    #o3d.io.write_point_cloud("./Inner_Deformation/{}.pcd".format(count), t2_d_pcd)
    t__1 = time.perf_counter()
    #print("Figure saving time: ",t__1-t__0)

    

    # cp.cuda.runtime.deviceSynchronize()
    # cp.cuda.Device().synchronize()
    #stream_ray.synchronize()
    #print(masked_pcd.point.positions.shape)
    return fine_T, curr_valid_mask, t2_und_t, np.max(d_dist), cropped_model_cuda , d_dist, t2_d_pcd_cu_t, masked_pcd,plane, contact_patch_mask, t2_d_pcd_cu.clone(), t2_und_in.clone() , t2_d_pcd_cu_o.clone(), t2_und.clone(), t2_d_pcd_cu_t.clone(), t2_und_t.clone(), contact_patch.clone(), dist, d_dist, d_dist_o, d_dist_t#t2_d_pcd_cu_t #1#t2_d_pcd_def

def calculate_surface_curvature(pcd, radius=0.1, max_nn=20):
    pcd_n = copy.deepcopy(pcd.to_legacy())
    pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    covs = np.asarray(pcd_n.covariances)
    vals, vecs = np.linalg.eig(covs)
    curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
    return curvature

def load_outer_inner_corres(file_name,scale):
    '''
    Loads the Inner Undeformed Model to Outer Undeformed Model Correspondences
    from a generated .npz file created in 'inner_projection.py'

    Args:
        file_name (str): file name of .npz file

    Returns:
        rays_hit_start_io (numpy.ndarray): A NumPy (N,3) array of the start ray location (Inner Undeformed Model)
        rays_hit_end_io (numpy.ndarray):  A NumPy (N,3) array of the ray end/hit location (Outer Undeformed Model)
    '''
    with open(file_name, 'rb') as f:
        rays_hit_start_io = np.load(f)
        rays_hit_end_io = np.load(f)
        
    return rays_hit_start_io*scale, rays_hit_end_io*scale

def draw_lines_lineset(start_points, end_points, line_set):
    line_start = cp.ascontiguousarray(cp.from_dlpack(start_points.to_dlpack()))
    line_end = cp.ascontiguousarray(cp.from_dlpack(end_points.to_dlpack()))

    dist = cp.linalg.norm(line_end-line_start,axis=1)
    # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # line_points = np.vstack((start_points, end_points))
    # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    valid_dist = dist < 0.055
    mask = (line_start != 0.0).all(axis=1) & (line_end != 0.0).all(axis=1) & valid_dist
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 
    
    disp = line_valid_end[:,0] - line_valid_start[:,0]
    #disp = (((local.reshape(-1,3))[::20])[mask])[:,2]

    if disp.shape[0] != 0:
        #normalized =  (disp - disp.min()) / (disp.max() - disp.min())
        normalized =  (disp - (-0.055)) / (0.055 - (-0.055))
        #(vel[mask] - vel[mask].mean()) / vel[mask].std() #
        # Use a colormap (e.g., viridis, jet, plasma)
        colormap = plt.cm.get_cmap('jet')
        colors = colormap(normalized.get())[:, :3]  # Drop alpha channel
        n = line_valid_start.shape[0]
        #print(np.max(line_start[:,2]),np.max(line_end[:,2]))
        lines = cp.empty((n, 2), dtype=np.int32)
        lines[:, 0] = cp.arange(n, dtype=np.int32)
        lines[:, 1] = cp.arange(n, 2 * n, dtype=np.int32)

        #lines = np.column_stack((np.arange(n), np.arange(n, 2*n)))
        #line_points = np.concatenate((start_points, end_points), axis=0)

        line_points = cp.empty((2 * n, 3), dtype=line_start.dtype)
        line_points[:n] = line_valid_start
        line_points[n:] = line_valid_end

        lps = line_points.toDlpack()
        ls = lines.toDlpack() 

        line_set.point.positions = o3d.core.Tensor.from_dlpack(lps)
        line_set.line.indices = o3d.core.Tensor.from_dlpack(ls)
        line_set.line.colors = o3d.core.Tensor(colors, dtype=o3d.core.float32, device = o3d.core.Device("CUDA:0"))

def align_to_ground_plane():
    R_r = 0.38515
    r_c = 0.2615
    theta = 50 * (np.pi)/180
    beta = 150 * (np.pi)/180
    w_c= np.array([-0.01,r_c*np.cos(theta)+0.025,-r_c*np.sin(theta)+0.02]) #np.array([0,-R_r,0])+
    t = o3d.core.Tensor(w_c,dtype = o3d.core.float32).cuda()
    yota = -5 * (np.pi)/180 #[0 test8,9] [-5 test7]
    eta = 48* (np.pi)/180
    R_z = o3d.core.Tensor([[np.cos(yota),-np.sin(yota),0],[np.sin(yota),np.cos(yota),0],[0,0,1]],dtype = o3d.core.float32).cuda()
    R_x = o3d.core.Tensor([[1,0,0],[0,np.cos(eta),-np.sin(eta)],[0,np.sin(eta),np.cos(eta)]],dtype = o3d.core.float32).cuda()
    #R = R_x.mul(R_z) 
    return R_z,t,R_x

def vel_gradient(vel_points,pcd):
    #color according to gradient
    return None

def trimmed_mean(a, trim_frac=0.1):
    lo = int(np.floor(trim_frac * a.shape[0]))
    hi = a.shape[0] - lo
    sort_idx = np.argsort(a, axis=0)
    trimmed = np.take_along_axis(a, sort_idx[lo:hi, :], axis=0)
    return np.mean(trimmed, axis=0)

def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid')

def main():
    start_time = time.time()
    """
    Main function to excute the script
    
    PERFORM STEPS TO ULTIMATELY GET OUTER TREAD DEFROMATION USING OPEN3D POINTCLOUD PROCESSING AND OPENCV OPTICAL FLOW
    """
    #===========================================================================================================================================
    ## LOAD CONFIG
    with open("./scripts/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    Real_Time = config["Real_Time"]
    Run_on_Jetson = config["Run_on_Jetson"]
    debug_mode = config["debug_mode"]
    view_video = config["view_video"]
    draw_reg = config["draw_reg"]
    scale = config["scale"] #0.03912#0.03805#0.0378047581618546 #0.0376047581618546 #scale = 0.019390745853434508

    ## LOAD APRILTAG CONTROL POINTS AND FIND CORRECTED TAG IDS
    #file_path_control_points = './4_row_model_control_points.csv'
    file_path_control_points = config["file_path_control_points"]#'./full_outer.csv'
    markers, m_points, numeric_markers = load_rc_control_points(file_path_control_points,scale)
    normalTag, RCcorTag = convert_rc_apriltag_hex_ids()
    correctTags = convert_rc_control_points(normalTag,numeric_markers)

    ## LOAD INNER TO OUTER CORRESPONDENCE
    inner_to_outer_np = config["inner_to_outer"]
    rays_hit_start_io, rays_hit_end_io = load_outer_inner_corres(inner_to_outer_np,scale)

    ## CREATE INNER MODEL PCD from CORRES
    # model_pcd = o3d.t.geometry.PointCloud()
    # model_pcd.point.positions = o3d.core.Tensor(rays_hit_start_io) #[np.abs(rays_hit_start_io[:,2]-0.15) < 0.001]
    model_pcd = load_model_pcd(config["file_path_model"],scale)
    model_pcd.estimate_normals()
    centroid = np.array([0.42841208,-1.6929364,3.6547658 ])*scale
    R = np.array([[ -0.03135062 , 0.35852575 , 0.93299323],
    [-0.03143202 , 0.9326368 , -0.35944495],
    [ -0.9990141 , -0.04059484 , -0.01796954]]).T
    # centroid = np.array([0.02758715, -0.07112041, 0.14297444])
    # # model_pcd.translate(o3d.core.Tensor(-centroid))
    # R = np.array([[2.52815128e-02, 3.33760291e-02, 9.99123058e-01], 
    #     [-7.85843857e-01, 6.18424477e-01, -7.73910520e-04], 
    #     [-6.17907985e-01, -7.85135152e-01,  4.18630537e-02]])
    # model_pcd.rotate(o3d.core.Tensor(R), center = [0,0,0])
    asd = o3d.t.geometry.PointCloud()
    asd.point.positions = o3d.core.Tensor(m_points)
    # o3d.visualization.draw([model_pcd,asd])
    model_pcd_cuda = model_pcd.cuda()

    # model_pcd2 = o3d.t.geometry.PointCloud()
    # model_pcd2.point.positions = o3d.core.Tensor(rays_hit_start_io) #[np.abs(rays_hit_start_io[:,2]-0.15) < 0.001]
    # model_pcd2.estimate_normals()
    # centroid = np.array([0.02758715, -0.07112041, 0.14297444])
    # model_pcd2.translate(o3d.core.Tensor(-centroid))
    # R = np.array([[2.52815128e-02, 3.33760291e-02, 9.99123058e-01], 
    #     [-7.85843857e-01, 6.18424477e-01, -7.73910520e-04], 
    #     [-6.17907985e-01, -7.85135152e-01,  4.18630537e-02]])
    # model_pcd2.rotate(o3d.core.Tensor(R), center = [0,0,0])
    # model_pcd_cuda2 = model_pcd2.cuda()

   

    #CREATE STREAMS
    stream_upload = cp.cuda.Stream(non_blocking = True)
    stream_download = cp.cuda.Stream(non_blocking = True)
    stream_cp = cp.cuda.Stream(non_blocking=True)
    stream_o3d_cp = cp.cuda.Stream(null=True)
    stream_ray = cp.cuda.Stream(non_blocking = True) #SET to FALSE if not taking DATA
    cv2_stream = cv2.cuda.Stream()
    wrap_cv2_cp_stream = cp.cuda.ExternalStream(cv2_stream.cudaPtr(), device_id=-1)
   
    ## REALTIME = Realsense, FOLDER = imageStream
    if Real_Time:
        depth_profile=config["cam_depth_profile"]
        color_profile=config["cam_color_profile"]
        rsManager = RealSenseManager(depth_profile=depth_profile,color_profile=color_profile,exposure=config["exposure"],gain=config["gain"],enable_spatial=False,enable_temporal=False)
    else:
        depth_profile=config["image_depth_profile"]
        color_profile=config["image_color_profile"]
        imageStream = read_RGB_D_folder(config["RGB_D_folder"],starting_index=config["start_index"], ending_index=config["end_index"],depth_num=depth_profile,debug_mode=debug_mode)
        with open(os.path.join(config["RGB_D_folder"],"time.npy"), 'rb') as f:
            time_arr = np.load(f)

    ## RUN APP to VIS IN REALTIME
    if view_video: viewer3d = Viewer3D("Outer Deformation and Tracking")

    ## RUN_ON_JETSON = apriltag library, WINDOWS = robotpy_apriltag library
    if Run_on_Jetson:
        from contact_patch.april_detect_jetson import DetectAprilTagsJetson
        detector = DetectAprilTagsJetson(depth_profile=depth_profile,debug_mode=debug_mode)
    else:
        from contact_patch.april_detect_windows import DetectAprilTagsWindows
        detector = DetectAprilTagsWindows(depth_profile=depth_profile,debug_mode=debug_mode)

    ## FIND PCD INDICES OF APRILTAG LOCATIONS 
    tag_norm, model_correspondence = find_tag_point_ID_correspondence(model_pcd_cuda, m_points)
    
    optmethod = config["opt_method"]
    if optmethod == 'dense':
        dense = DenseOptFlow(depth_profile,debug_mode,config['dense_method'], cv2_stream, wrap_cv2_cp_stream)
    elif optmethod == 'sparse':
        sparse = SparseOptFlow(depth_profile,debug_mode,config['sparse_step'])

    inner_arr = []
    outer_arr = []
    inner_lat = []
    outer_lat = []
    inner_undef = []
    inner_undef_lat = []
    outer_undef = []
    outer_undef_lat = []

    outer_model_ply = o3d.t.io.read_triangle_mesh("./assets/full_outer_outer_part_only.ply")
    outer_model_ply.compute_vertex_normals()
    outer_model_ply.compute_triangle_normals()
    outer_model_ply.normalize_normals()
    outer_model_ply.translate(o3d.core.Tensor(-centroid))
    outer_model_ply.rotate(o3d.core.Tensor(R), center = [0,0,0])
    outer_model_ply = outer_model_ply.cuda()
    
    raycaster = OptiXRaycaster("./assets/full_outer_inner_smoothed_part_only.ply", "./assets/full_outer_outer_part_only.ply", "./assets/full_outer_treads_part_only.ply", "./src/contact_patch/raycast.cu", stream_ray)

    load_time = time.time() - start_time
    print("Loading Completed in",load_time)

    outer_disp = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
    outer_vel =  o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
    prev_outer_deformed = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
    prev_outer_select = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
    prev_valid_mask = o3d.core.Tensor.empty((480,848),o3d.core.Dtype.Bool,o3d.core.Device("CUDA:0"))
    curr_t = o3d.core.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).cuda()

    Rzg,tg,Rxg = align_to_ground_plane()

    data7 = sio.loadmat("./assets/contact_patch_points_test_7.mat")
    data8 = sio.loadmat("./assets/contact_patch_points_test_8.mat")
    data9 = sio.loadmat("./assets/contact_patch_points_test_9.mat")
    data21 = sio.loadmat("./assets/contact_patch_points_test_21.mat")

    #boundary = data["boundary_mm_all"]    
    interior7 = data7["interior_mm"].astype(np.float32) /1000
    interior7 = interior7 - interior7.mean(axis=0)
    interior7 = -interior7
    interior8 = data8["interior_mm"] /1000 
    interior8 = interior8 - interior8.mean(axis=0)
    interior9 = data9["interior_mm"] /1000  
    interior9 = interior9 - interior9.mean(axis=0)
    interior21 = data21["interior_mm"] /1000 
    interior21 = interior21 - interior21.mean(axis=0)
    interior21 = -interior21
    #boundary_3d = np.hstack([boundary, np.zeros((boundary.shape[0], 1))])
    interior_3d7 = np.hstack([interior7, np.zeros((interior7.shape[0], 1))])
    interior_3d8 = np.hstack([interior8, np.zeros((interior8.shape[0], 1))])
    interior_3d9 = np.hstack([interior9, np.zeros((interior9.shape[0], 1))])
    interior_3d21 = np.hstack([interior21, np.zeros((interior21.shape[0], 1))])

    # boundary_pcd = o3d.t.geometry.PointCloud()
    #boundary_pcd.point.positions = o3d.core.Tensor(boundary_3d)
    # interior_pcd7 = o3d.t.geometry.PointCloud()
    # interior_pcd7.point.positions = o3d.core.Tensor(interior_3d7)
    # target_ds = interior_pcd7.voxel_down_sample(voxel_size=0.002)
    # interior_pcd7.translate(-target_ds.point.positions.mean(dim=0))
    # interior_pcd7.rotate(Rzg.cpu(), center = [0,0,0])
    # target_ds.rotate(Rzg.cpu(), center = [0,0,0])
    # interior_pcd7.translate(-target_ds.point.positions.mean(dim=0)+o3d.core.Tensor([0.008,-0.005,0]))
    # interior_pcd8 = o3d.t.geometry.PointCloud()
    # interior_pcd8.point.positions = o3d.core.Tensor(interior_3d8)
    # target_ds = interior_pcd8.voxel_down_sample(voxel_size=0.002)
    # interior_pcd8.translate(-target_ds.point.positions.mean(dim=0))
    # #interior_pcd8.rotate(Rzg.cpu(), center = [0,0,0])
    # #target_ds.rotate(Rzg.cpu(), center = [0,0,0])
    # interior_pcd8.translate(-target_ds.point.positions.mean(dim=0)+o3d.core.Tensor([0.01,-0.007,0]))
    # interior_pcd9 = o3d.t.geometry.PointCloud()
    # interior_pcd9.point.positions = o3d.core.Tensor(interior_3d9)
    # target_ds = interior_pcd9.voxel_down_sample(voxel_size=0.002)
    # interior_pcd9.translate(-target_ds.point.positions.mean(dim=0))
    # interior_pcd9.rotate(Rzg.cpu(), center = [0,0,0])
    # target_ds.rotate(Rzg.cpu(), center = [0,0,0])
    # interior_pcd9.translate(-target_ds.point.positions.mean(dim=0)+o3d.core.Tensor([0,-0.015,0]))
    interior_pcd21 = o3d.t.geometry.PointCloud()
    interior_pcd21.point.positions = o3d.core.Tensor(interior_3d21)
    target_ds = interior_pcd21.voxel_down_sample(voxel_size=0.002)
    interior_pcd21.translate(-target_ds.point.positions.mean(dim=0))
    #interior_pcd9.rotate(Rzg.cpu(), center = [0,0,0])
    #target_ds.rotate(Rzg.cpu(), center = [0,0,0])
    interior_pcd21.translate(-target_ds.point.positions.mean(dim=0)+o3d.core.Tensor([0,-0.003,-0.003]))

    # o3d.visualization.draw([interior_pcd7, interior_pcd8, interior_pcd9, interior_pcd21])
    ## Initialise PREV frame
    #==========================================================================================================================================================================
    start_loop_time = time.time()
    event_frame_done = cp.cuda.Event()
    if Real_Time:
        time_ms, count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = rsManager.get_frames()
        print("IMAGE NUMBER:",count, time_ms)
    elif imageStream.has_next():
        count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = imageStream.get_next_frame()
        time_ms = time_arr[count]
        # time_ms = 33.33
        print("IMAGE NUMBER:",count, time_ms)
    # o3d.visualization.draw([model_pcd2, model_pcd,t2cam_pcd_cuda.cpu()])
    gpu_curr = cv2.cuda.GpuMat()
    gpu_curr_gray = cv2.cuda.GpuMat()
    
    with wrap_cv2_cp_stream:
        dl_color = color_image.as_tensor().to_dlpack()
        cp_color = cp.from_dlpack(dl_color)
        color_ptr = cp_color.data.ptr

    gpu_prev = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_ptr)
    gpu_prev_gray = cv2.cuda.GpuMat()
    gpu_prev_gray_uint8 = gpu_prev.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
    gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev_gray_uint8, cv2.COLOR_BGR2GRAY, stream=cv2_stream)

    dense.init_disp(vertex_map_gpu)
    # print(vertex_map_gpu.as_tensor()[480//2,848//2])
    
    event_load_cv2_frames = cv2.cuda.Event()
    event_load_cv2_frames.record(stream=cv2_stream)
    event_load_cv2_frames.waitForCompletion()

    event_cpu_task = cp.cuda.Event()

    ## APRILTAG DETECT
    t_0 = time.time()
    detector.input_frame(color_image.as_tensor().cpu().numpy())
    t_1 = time.time()   
    tag_IDs, tag_locations, pcd_IDs = detector.process_3D_locations(vertex_map_gpu.as_tensor().cpu().numpy())
    t_2 = time.time() 
    time_at_detection = time.time()
    detect_time = time_at_detection - start_loop_time
    # print("INPUT DATA:", t_0-start_loop_time)
    # print("AprilTags Detected in", t_2-t_0)
    # print("2D_process:", t_1-t_0)
    # print("3D_process:", t_2-t_1)

    # PREPROCESING FOR APRILTAG ALIGNMENT
    t2cam_correspondence_gpu = o3d.core.Tensor.from_numpy(np.array(pcd_IDs).astype(np.int64)).cuda()
    correspondence_vector,p,q = make_correspondence_vector(t2cam_correspondence_gpu,model_correspondence,tag_IDs,correctTags,debug_mode)
    time_at_corres_vec = time.time()
    corres_vec_time = time_at_corres_vec - detect_time#time_at_t2cam_corres
    #print("Corres Vector made in", corres_vec_time)
    
    ## ROUGH ALIGNMENT WITH T2CAM USING APRILTAGS
    with stream_o3d_cp:
        event_rough_align = cp.cuda.Event()
        rough_T, invT = register_t2cam_with_model(t2cam_pcd_cuda,t2cam_pcd_cuda,model_pcd_cuda,correspondence_vector,p,q,curr_t)
        time_at_registration = time.time()
        register_time = time_at_registration - time_at_corres_vec
        #print("Registration in", register_time)
        event_rough_align.record(stream_o3d_cp)

    stream_ray.wait_event(event_rough_align)
    stream_cp.wait_event(event_rough_align)

    with stream_ray:
        event_deformation = cp.cuda.Event()
        fine_T,init_mask, t2_d_pcd_inner_cu, max_inner_def, cropped_model_pcd_cu, d_inner_dist,t2_d_pcd_outer_cu, outer_select,plane,contact_patch_mask, t2_d_pcd_cu, t2_und_in , t2_d_pcd_cu_o, t2_und, t2_d_pcd_cu_t, t2_und_t, contact_patch, dist, d_dist, d_dist_o, d_dist_t = inner_deformed_to_outer_deformed(p,q,invT,rough_T,stream_o3d_cp,stream_ray,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,t2cam_pcd_cuda,count,outer_model_ply,draw_reg)
        time_at_Inner_c2c = time.time()
        c2c_dist_time = time_at_Inner_c2c - time_at_registration
        #print("Inner and Outer C2C raycasting distance calculated in", c2c_dist_time)
        event_deformation.record(stream_ray)

    stream_o3d_cp.wait_event(event_deformation)    

    full_T = rough_T #fine_T.matmul(rough_T)
    inv_full_T = invT #full_T.inv()

    t2_d_pcd_outer_cu.transform(inv_full_T)

    t2_d_pcd_outer_cu.translate(tg)
    t2_d_pcd_outer_cu.rotate(Rxg,center=tg)

    prev_outer_deformed.point.positions = t2_d_pcd_outer_cu.point.positions #.clone()
    #prev_outer_select.point.positions = outer_select.transform(inv_full_T).point.positions #.clone()
    
    prev_valid_mask = init_mask.clone()

    normal_map_gpu_prev = normal_map_gpu #o3d tensor gpu

    event_frame_done.record(stream_o3d_cp)

    mean_vel = []
    tracked_p_loc = []

    bins = np.linspace(-0.035, 0, 11)
    cmap = plt.get_cmap('plasma', len(bins)-1)

    # Create a dummy ScalarMappable with the same colormap and normalization
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(2, 5))  # vertical colorbar

    # Plot the colorbar
    cbar = fig.colorbar(sm, cax=ax, ticks=bins)
    cbar.set_label("Value bins")
    plt.show(block=False)

    arr_pcd_undeform_inner = []
    arr_pcd_deform_inner = []
    arr_pcd_undeform_outer = []
    arr_pcd_deform_outer = []
    arr_pcd_undeform_tread = []
    arr_pcd_deform_tread = []
    arr_pcd_contact_patch = [] 
    arr_d_dist_inner = []
    arr_d_dist_outer = []
    arr_d_dist_tread = []


    
    ## LOOP THROUGH EACH FRAME 1 onwards
    #==========================================================================================================================================================================
    try:
        while True:
            with nvtx.annotate("intake", color="red"):
                start_loop_time = time.time()
                wrap_cv2_cp_stream.wait_event(event_frame_done)
                start_cv2 = cv2.getTickCount()
                ## INPUT
                if Real_Time:
                    time_ms, count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = rsManager.get_frames()
                    print("IMAGE NUMBER:",count, time_ms)
                elif imageStream.has_next():
                    count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = imageStream.get_next_frame()
                    time_ms = time_arr[count]
                    # time_ms = 66.67
                    print("IMAGE NUMBER:",count, time_ms)
                else:
                    # if Real_Time: rsManager.stop()
                    if view_video: viewer3d.stop()
                    
                    # sys.exit(0)
                    break
                
                event_load_cv2_frames = cv2.cuda.Event()

                with wrap_cv2_cp_stream:
                    dl_color_loop = color_image.as_tensor().to_dlpack()
                    cp_color_loop = cp.from_dlpack(dl_color_loop)
                    color_loop_ptr = cp_color_loop.data.ptr

                gpu_curr = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_loop_ptr)
                gpu_curr_gray_uint8 = gpu_curr.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
                gpu_curr_gray = cv2.cuda.cvtColor(gpu_curr_gray_uint8, cv2.COLOR_BGR2GRAY,stream=cv2_stream)

                
                event_load_cv2_frames.record(stream=cv2_stream)
                event_load_cv2_frames.waitForCompletion()

             ## APRILTAG DETECT
            try:
                with nvtx.annotate("cpu april", color="blue"):
                    t_0 = time.time()
                    detector.input_frame(color_image.as_tensor().cpu().numpy())
                    t_1 = time.time()   
                    tag_IDs, tag_locations, pcd_IDs = detector.process_3D_locations(vertex_map_gpu.as_tensor().cpu().numpy())
                    t_2 = time.time() 
                    time_at_detection = time.time()
                    detect_time = time_at_detection - start_loop_time
                    #print("INPUT DATA:", t_0-start_loop_time)
                    #print("AprilTags Detected in", t_2-t_0)
                    #print("2D_process:", t_1-t_0)
                    #print("3D_process:", t_2-t_1)

                ## DENSE OPTICAL FLOW
                with nvtx.annotate("optic", color="green"):
                    with wrap_cv2_cp_stream:
                        event_tracked_pcd = cp.cuda.Event()
                        frame_gpu  = dense.detect2D(gpu_prev_gray, gpu_curr_gray)
                        gpu_bgr = dense.vis_hsv_2D()
                        map_x_gpu, map_y_gpu, curr_tracked_t2cam_pcd = dense.detect3D(count, vertex_map_gpu,normal_map_gpu_prev, normal_map_gpu)
                        curr_tracked_t2cam_pcd.normalize_normals()
                        event_tracked_pcd.record(wrap_cv2_cp_stream)
                        dense.local_geo_calc()
                        dense.track_3D_vel(time_ms)
                        #g1,g2,d1,d2,frame_3D,vel_arrow,curr_outer_pcd = dense.vis_3D()

            except Exception as e:
                import traceback
                traceback.print_exc()
                print("CUDA ERROR:", e)
                                
                # x_points = map_x_gpu.download()
                # y_points = map_y_gpu.download()
                # curr_mask = (x_points <= 847) & (y_points <= 479) & (x_points >= 0) & (y_points >= 0)
                
            with nvtx.annotate("deform", color="yellow"):
                try:
                    #with stream_ray:                
                    # PREPROCESING FOR APRILTAG ALIGNMENT
                    t2cam_correspondence_gpu = o3d.core.Tensor.from_numpy(np.array(pcd_IDs).astype(np.int64)).cuda()
                    correspondence_vector,p,q = make_correspondence_vector(t2cam_correspondence_gpu,model_correspondence,tag_IDs,correctTags,debug_mode)
                    time_at_corres_vec = time.time()
                    corres_vec_time = time_at_corres_vec - detect_time
                    #print("Corres Vector made in", corres_vec_time)
                    
                    stream_o3d_cp.wait_event(event_tracked_pcd)
                    with stream_o3d_cp:
                        event_rough_align = cp.cuda.Event()
                        ## ROUGH ALIGNMENT WITH T2CAM USING APRILTAGS
                        rough_T,invT = register_t2cam_with_model(t2cam_pcd_cuda,curr_tracked_t2cam_pcd ,model_pcd_cuda,correspondence_vector,p,q,curr_t)
                        time_at_registration = time.time()
                        register_time = time_at_registration - time_at_corres_vec
                        #print("Registration in", register_time)
                        event_rough_align.record(stream_o3d_cp)

                    stream_ray.wait_event(event_rough_align)
                    stream_cp.wait_event(event_rough_align)

                    with stream_ray:
                        event_deformation = cp.cuda.Event()
                        fine_T, curr_valid_mask,t2_d_pcd_inner_cu, max_inner_def, cropped_model_pcd_cu, d_inner_dist,t2_d_pcd_outer_cu, outer_select,plane, contact_patch_mask, t2_d_pcd_cu, t2_und_in , t2_d_pcd_cu_o, t2_und, t2_d_pcd_cu_t, t2_und_t, contact_patch, dist, d_dist, d_dist_o, d_dist_t = inner_deformed_to_outer_deformed(p,q,invT,rough_T,stream_o3d_cp,stream_ray,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,curr_tracked_t2cam_pcd ,count,outer_model_ply,draw_reg)
                        # tracked_p_loc.append(curr_tracked_t2cam_pcd.point.positions[240*848+424].cpu().numpy())
                        time_at_Inner_c2c = time.time()
                        c2c_dist_time = time_at_Inner_c2c - time_at_registration
                        #print("Inner and Outer C2C raycasting distance calculated in", c2c_dist_time)
                        event_deformation.record(stream_ray)

                    stream_o3d_cp.wait_event(event_deformation)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("CUDA ERROR:", e)
                        
            #stream_ray.synchronize()
            full_T = rough_T #fine_T.matmul(rough_T)
            inv_full_T = invT#full_T.inv()
            # print("##################################################")
            # print("TRANSFORM")
            # print("##################################################")
            # print(inv_full_T)
            # print(full_T)
            # print("##################################################")
            with nvtx.annotate("vis rest", color="black"):
                try:
                    with stream_o3d_cp:
                        # print("t2_d_pcd_outer_cu shape", t2_d_pcd_outer_cu.point.positions.shape)
                        # print("prev_outer_deformed shape", prev_outer_deformed.point.positions.shape)
                        t2_d_pcd_outer_cu.transform(inv_full_T)
                        # t2_d_pcd_outer_cu.translate(tg)
                        # t2_d_pcd_outer_cu.rotate(Rxg,center=tg)
                        t2_d_pcd_inner_cu.transform(inv_full_T)
                        # t2_d_pcd_inner_cu.translate(tg)
                        # t2_d_pcd_inner_cu.rotate(Rxg,center=tg)
                        
                        curr_points = t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask & prev_valid_mask ).point.positions #& contact_patch_mask
                        prev_points = prev_outer_deformed.select_by_mask(curr_valid_mask & prev_valid_mask).point.positions # & contact_patch_mask
                        
                        # curr_points = t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask & prev_valid_mask & contact_patch_mask).point.positions 
                        # prev_points = prev_outer_deformed.select_by_mask(curr_valid_mask & prev_valid_mask & contact_patch_mask).point.positions 
                        
                        disp_points = curr_points.sub(prev_points)
                        
                        vel_points = disp_points.div(time_ms/1000) 

                        mean_vel_points = vel_points.mean(dim=0).cpu().numpy()
                        
                        tmean = trimmed_mean(vel_points.cpu().numpy(), 0.48)  
                        print("Mean Contact Patch Vel", mean_vel_points, tmean) 
                        #mean_vel.append(mean_vel_points) 
                        mean_vel.append(tmean)           

                        curr_vel_points = prev_points.add(vel_points)
                        wrt_cam_outer_select = outer_select.transform(inv_full_T)

                        #wrt_cam_outer_select.point.positions = wrt_cam_outer_select.point.positions.to(o3d.core.float32)
                        #interior_pcd7.point.positions = interior_pcd7.point.positions.to(o3d.core.float32)
                        # Convert normals (if they exist)
                        #wrt_cam_outer_select.point.normals = wrt_cam_outer_select.point.normals.to(o3d.core.float32)

                        # Convert colors (if they exist)
                        # wrt_cam_outer_select.point.colors = wrt_cam_outer_select.point.colors.to(o3d.core.float32)

                        # reg = o3d.t.pipelines.registration.icp(source = wrt_cam_outer_select.uniform_down_sample(every_k_points=10),
                        #                         target = interior_pcd7.cuda().uniform_down_sample(every_k_points=1000),
                        #                         max_correspondence_distance = 0.02,
                        #                         estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                        #                         criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=100))
                        # wrt_cam_outer_select.transform(reg.transformation)

                        #wrt_cam_outer_select.rotate(Rzg,center=[0,0,0])
                        wrt_cam_outer_select.translate(tg)
                        wrt_cam_outer_select.rotate(Rxg,center=tg)

                        plane.transform(inv_full_T)
                        plane.translate(tg)
                        plane.rotate(Rxg,center=tg)

                        curr_tracked_t2cam_pcd.transform(inv_full_T)
                        curr_tracked_t2cam_pcd.translate(tg)
                        curr_tracked_t2cam_pcd.rotate(Rxg,center=tg)

                        t2_d_pcd_cu.transform(inv_full_T)
                        t2_und_in.transform(inv_full_T)
                        t2_d_pcd_cu_o.transform(inv_full_T)
                        t2_und.transform(inv_full_T)
                        t2_d_pcd_cu_t.transform(inv_full_T)
                        # t2_d_pcd_cu_t.translate(tg)
                        # t2_d_pcd_cu_t.rotate(Rxg,center=tg)
                        t2_und_t.transform(inv_full_T)
                        contact_patch.transform(inv_full_T)
                        # contact_patch.translate(tg)
                        # contact_patch.translate(o3d.core.Tensor([0,0,0.002]))
                        # contact_patch.rotate(Rxg,center=tg)

                        # np.savez_compressed(f"saved_arrays/iteration_{count:03d}.npz",
                        #     t = time_arr,
                        #     pcd_inner_deform = t2_d_pcd_cu.point.positions.cpu().numpy(),
                        #     pcd_inner_undeform = t2_und_in.point.positions.cpu().numpy(),
                        #     pcd_outer_deform = t2_d_pcd_cu_o.point.positions.cpu().numpy(),
                        #     pcd_outer_undeform = t2_und.point.positions.cpu().numpy(),
                        #     pcd_tread_deform = t2_d_pcd_cu_t.point.positions.cpu().numpy(),
                        #     pcd_tread_undeform = t2_und_t.point.positions.cpu().numpy(),
                        #     pcd_contact_patch = contact_patch.point.positions.cpu().numpy(),
                        #     d_inner_deform= d_dist.get(),
                        #     d_inner_to_outer_undeform = d_dist_o.get(),
                        #     d_inner_to_tread_undeform = d_dist_t.get()
                        # )
                        # np.savez(f"saved_arrays/iteration_unc_{count:03d}.npz",
                        #     t = time_arr,
                        #     pcd_inner_deform = t2_d_pcd_cu.point.positions.cpu().numpy(),
                        #     pcd_inner_undeform = t2_und_in.point.positions.cpu().numpy(),
                        #     pcd_outer_deform = t2_d_pcd_cu_o.point.positions.cpu().numpy(),
                        #     pcd_outer_undeform = t2_und.point.positions.cpu().numpy(),
                        #     pcd_tread_deform = t2_d_pcd_cu_t.point.positions.cpu().numpy(),
                        #     pcd_tread_undeform = t2_und_t.point.positions.cpu().numpy(),
                        #     pcd_contact_patch = contact_patch.point.positions.cpu().numpy(),
                        #     d_inner_deform= d_dist.get(),
                        #     d_inner_to_outer_undeform = d_dist_o.get(),
                        #     d_inner_to_tread_undeform = d_dist_t.get()
                        # )
                        # savemat(f"D:/stored_arrays/Steering_test_8/iteration_{count:03d}.mat",{
                        #     "t" : time_arr,
                        #     "pcd_inner_deform" : t2_d_pcd_cu.point.positions.cpu().numpy(),
                        #     "pcd_inner_undeform" : t2_und_in.point.positions.cpu().numpy(),
                        #     "pcd_outer_deform" : t2_d_pcd_cu_o.point.positions.cpu().numpy(),
                        #     "pcd_outer_undeform" : t2_und.point.positions.cpu().numpy(),
                        #     "pcd_tread_deform" : t2_d_pcd_cu_t.point.positions.cpu().numpy(),
                        #     "pcd_tread_undeform" : t2_und_t.point.positions.cpu().numpy(),
                        #     "pcd_contact_patch" : contact_patch.point.positions.cpu().numpy(),
                        #     "dist_3d": dist.get(),
                        #     "d_inner_deform" : d_dist.get(),
                        #     "d_inner_to_outer_undeform" : d_dist_o.get(),
                        #     "d_inner_to_tread_undeform" : d_dist_t.get(),
                        #     "curr_valid_mask" : curr_valid_mask.cpu().numpy(),
                        #     "prev_valid_mask" : prev_valid_mask.cpu().numpy(),
                        #     "contact_patch_mask" : contact_patch_mask.cpu().numpy()
                        # })

                        if view_video:
                            #draw_lines_lineset(prev_outer_deformed.select_by_mask(curr_valid_mask & prev_valid_mask).transform(inv_full_T).point.positions,t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask & prev_valid_mask).transform(inv_full_T).point.positions,outer_disp)
                            #draw_lines_lineset(prev_outer_deformed.select_by_mask(curr_valid_mask | prev_valid_mask).transform(inv_full_T).point.positions,t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask | prev_valid_mask).transform(inv_full_T).point.positions,outer_disp)
                            draw_lines_lineset(prev_points,curr_points,outer_disp)
                            # draw_lines_lineset(prev_points,curr_vel_points,outer_vel)
                            # viewer3d.update_cloud(geometries = t2_d_pcd_inner_cu.cpu(),lines = t2_d_pcd_outer_cu.cpu())
                            #viewer3d.update_cloud(undeformed_outer= t2_d_pcd_inner_cu.cpu(), deformed_outer = outer_select.cpu(), prev_def = prev_outer_select.cpu(), outer_disp = outer_disp)
                            #viewer3d.update_cloud(deformed_outer = wrt_cam_outer_select.cpu(), outer_disp = outer_disp.cpu(), outer_vel = outer_vel.cpu())
                            #viewer3d.update_cloud(deformed_outer_shape = wrt_cam_outer_select.cpu(), outer_disp = outer_disp.cpu(), outer_vel = outer_vel.cpu())
                            # viewer3d.update_cloud(deformed_outer_shape = wrt_cam_outer_select.cpu(),undeformed_outer= t2_d_pcd_inner_cu.cpu(), inner=curr_tracked_t2cam_pcd.cpu(),outer = t2_d_pcd_cu_o.cpu())
                            # viewer3d.update_cloud(outer = t2_d_pcd_cu_o.cpu(), tread_undef = t2_und_t.cpu(),tread_def = t2_d_pcd_cu_t.cpu(), inner =  t2_d_pcd_cu.cpu())
                            viewer3d.update_cloud(contact = contact_patch.cpu(),plane = plane.cpu())
                            
                            #undeformed_outer= t2_d_pcd_inner_cu.transform(inv_full_T).cpu(),, prev_def = prev_outer_select.cpu(), plane = plane.transform(inv_full_T).cpu()
                            #viewer3d.update_cloud(outer_disp = outer_disp.cpu())# .paint_uniform_color(o3d.core.Tensor([1,0,0])) .paint_uniform_color(o3d.core.Tensor([0,1,0]))
                            #viewer3d.update_cloud(deformed_outer = outer_select.transform(inv_full_T).paint_uniform_color(o3d.core.Tensor([1,0,0])).cpu(), outer_disp = outer_disp)                    
                            viewer3d.tick()
                            # 0.003858396836115 
                        time_at_app_view = time.time()
                        app_view_time = time_at_app_view - time_at_Inner_c2c
                        print("Time to update viewer", app_view_time)

                        prev_outer_deformed.point.positions = t2_d_pcd_outer_cu.point.positions #.clone()
                        prev_outer_undeformed = t2_d_pcd_inner_cu #.clone()
                        prev_valid_mask = curr_valid_mask.clone()
                        prev_outer_select.point.positions = outer_select.point.positions #.clone()

                        gpu_prev = gpu_curr
                        gpu_prev_gray = gpu_curr_gray

                        normal_map_gpu_prev = normal_map_gpu

                        end_loop_time = time.time()
                        end_cv2 = cv2.getTickCount()
                        time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
                        print("FPS:", 1/time_sec)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("CUDA ERROR:", e)


            frame = frame_gpu.download()
            #valid_mask_cpu = valid_mask.download()
            # valid_image = np.zeros((480, 848), dtype=np.uint8)
            # valid_image[curr_mask] = 255
            # prev_mask = curr_mask
            # visualization for dense
            cv2.imshow("original", frame)
            cv2.imshow("result", gpu_bgr.download())
            # cv2.imshow("valid points", valid_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if view_video: viewer3d.stop()
                break
            if key == ord('p'):
                print("Paused. Press any key to continue...")
                key2 = cv2.waitKey(0)
            #key2 = cv2.waitKey(0)
            print("Time for one frame:",time_sec)
            event_frame_done.record(stream_o3d_cp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CUDA ERROR:", e)
    finally:
        if Real_Time: rsManager.stop()
        stream_cp.synchronize()
        stream_o3d_cp.synchronize()
        stream_ray.synchronize() #SET to FALSE if not taking DATA
        wrap_cv2_cp_stream.synchronize()
        cp.cuda.Device().synchronize()

        cv2.destroyAllWindows()
        cv2.waitKey(1)  

        # # --- Make axes equal ---
        # def set_axes_equal(ax):
        #     '''Make 3D plot axes have equal scale so that spheres look like spheres'''
        #     x_limits = ax.get_xlim3d()
        #     y_limits = ax.get_ylim3d()
        #     z_limits = ax.get_zlim3d()

        #     x_range = abs(x_limits[1] - x_limits[0])
        #     x_middle = np.mean(x_limits)
        #     y_range = abs(y_limits[1] - y_limits[0])
        #     y_middle = np.mean(y_limits)
        #     z_range = abs(z_limits[1] - z_limits[0])
        #     z_middle = np.mean(z_limits)

        #     plot_radius = 0.5 * max([x_range, y_range, z_range])

        #     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        #     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        #     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        # tracked_p_loc = np.array(tracked_p_loc)

       
        # # plt.plot(tracked_p_loc[:,0],tracked_p_loc[:,2])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot the points
        # ax.plot(tracked_p_loc[:,0],tracked_p_loc[:,1],tracked_p_loc[:,2], label='3D line')
        # end_pcd = curr_tracked_t2cam_pcd.point.positions.cpu().numpy()
        # model_pcd_np = model_pcd.point.positions.cpu().numpy()
        # ax.scatter(model_pcd_np[::1000,0],model_pcd_np[::1000,1],model_pcd_np[::1000,2],c='r', marker='o')
        # ax.scatter(end_pcd[::100,0],end_pcd[::100,1],end_pcd[::100,2],c='g', marker='o')
        # set_axes_equal(ax)
        # # Labels
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.set_title('3D Scatter Plot')
        # print(np.linalg.norm(tracked_p_loc[0]-tracked_p_loc[-1]))

        # plt.show()
        # print(1)
        # mean_vel = np.array(mean_vel)
        # print(mean_vel.shape)
        # t_m = np.cumsum(time_arr[config['start_index']:config['end_index']-1]/1000)
        # print(t_m.shape)
        # print(2)
        # plt.figure()
        # plt.plot(t_m,mean_vel[:,2])
        # plt.plot(t_m,mean_vel[:,1])
        # plt.plot(t_m,mean_vel[:,0])
        # plt.legend(['x','y','z'])
        # plt.title("Mean Contact Patch Velocity [m/s]")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Velocity [m/s]")
        # plt.grid()
        # plt.show()

        # # savemat("cp_vel_roll_over_cleat_backwards.mat", {"time": t_m,"data_vel": mean_vel})

        # slip_angle = np.arctan2(moving_average(mean_vel[:,0],10),moving_average(-mean_vel[:,2],10)) * (180/np.pi)

        # plt.figure()
        # plt.plot(t_m[:-9], slip_angle)
        # plt.title("Slip Angle [Deg]")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Angle [Deg]")
        # plt.grid()
        # plt.show()

        #if view_video: viewer3d.stop()
        # max_len = 4000
        # uniform_inner_arr = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in inner_arr
        # ])

        # uniform_outer_arr = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in outer_arr
        # ])

        # max_len = 4000
        # uniform_inner_lat = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in inner_lat
        # ])

        # uniform_outer_lat = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in outer_lat
        # ])

        # max_len = 4000
        # uniform_inner_arr_un = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in inner_undef
        # ])

        # uniform_outer_arr_un = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in outer_undef
        # ])

        # max_len = 4000
        # uniform_inner_lat_un = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in inner_undef_lat
        # ])

        # uniform_outer_lat_un = np.array([
        #     np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
        #     for a in outer_undef_lat
        # ])

        # np.save('inner_arr.npy', uniform_inner_arr)
        # np.save('outer_arr.npy', uniform_outer_arr)
        # np.save('inner_lat.npy', uniform_inner_lat)
        # np.save('outer_lat.npy', uniform_outer_lat)
        # np.save('inner_arr_un.npy', uniform_inner_arr_un)
        # np.save('outer_arr_un.npy', uniform_outer_arr_un)
        # np.save('inner_lat_un.npy', uniform_inner_lat_un)
        # np.save('outer_lat_un.npy', uniform_outer_lat_un)
        print("done")
        
        sys.exit(0)



if __name__ == "__main__":
    main()