import cv2
import open3d as o3d
import numpy as np
import csv
import matplotlib
#matplotlib.use("Agg")
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from capture_realsense_tensor import RealSenseManager
from replay_realsense_tensor import read_RGB_D_folder
from Dense_Opt_Flow import DenseOptFlow
from Sparse_Opt_Flow import SparseOptFlow
from app_vis import Viewer3D
import vidVisualiser as vV
import time
import yaml
from scipy.spatial import cKDTree
from numba import njit
import os
import cupy as cp
import sys
import copy
from optix_castrays import OptiXRaycaster
import nvtx
import threading

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
                    scale_factor*float(row[1]),
                    scale_factor*float(row[2]),
                    scale_factor*float(row[3])
                ])  
                numeric_markers.append(row[5])

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return markers, m_points, numeric_markers

def convert_rc_apriltag_hex_ids(file_path="./RCtoTag.csv"):
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
    return pcd

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

def register_t2cam_with_model(t2cam_pcd_cuda,model_pcd_cuda,corres_vector,p,q, curr_t):
    """
    Function that transforms the T2Cam point cloud to register with the inner tyre model point cloud

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): open3d Point Cloud of T2Cam 
        model_pcd (o3d.geometry.PointCloud): open3d Point Cloud of inner tyre model
        corres_vector (o3d.utility.Vector2iVector): Correspondence array for registration 
    """
    estimator = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    icp_t2cam = t2cam_pcd_cuda.select_by_index(p)
    #o3d.visualization.draw([t2cam_pcd_cuda,icp_t2cam])
    icp_model = model_pcd_cuda.select_by_index(q)
    T = estimator.compute_transformation(icp_t2cam,icp_model,corres_vector, current_transform = curr_t)
    t2cam_pcd_cuda.transform(T)
    #mesh.transform(T)
    #o3d.visualization.draw([icp_t2cam, icp_model,t2cam_pcd_cuda, model_pcd_cuda])
    return T
    
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

def segment_pcd_using_bounding_box(t2cam_pcd_cuda,model_pcd, model_pcd_cuda):
    """
    Segment the model point cloud to contain that section of t2Cam point cloud
    calculate bounding box of t2cam_pcd and crop model according to it

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): A Point Cloud of T2Cam image
        model_pcd (o3d.geometry.PointCloud): A Point Cloud of inner tyre model

    Returns:
        cropped_model (o3dd.geometry.PointCloud): a Point Cloud consisting of t2cam bounds
    """
    t2cam_bounding_box = t2cam_pcd_cuda.get_axis_aligned_bounding_box()
    cropped_model_cuda = model_pcd_cuda.crop(t2cam_bounding_box)
    cropped_model = cropped_model_cuda.cpu()
    return cropped_model, cropped_model_cuda

def ICP_register_T2Cam_with_model(cropped_model_cuda,t2cam_pcd_cuda,d_t2_points,downsampled_cropped_model_cu, draw_reg):
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
    mid_xyz = (cp.ptp(cp_d_t2_points,axis=0)/2 + cp.min(cp_d_t2_points,axis=0))

    dl_mid_xyz = mid_xyz.toDlpack()
    o3d_mid_xyz = o3d.core.Tensor.from_dlpack(dl_mid_xyz)
    frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1,device=o3d.core.Device("CUDA:0"))
    frame.translate(o3d_mid_xyz)
    # create mask to select the outer points in the pcd to avoid registration with the deformed parts
    mask = ((cp_d_t2_points[:,0]-mid_xyz[0])**2+(cp_d_t2_points[:,1]-mid_xyz[1])**2+((cp_d_t2_points[:,2]-mid_xyz[2])/1.2)**2) > 0.12**2
    

    downsampled_points_removed_def = cp_d_t2_points[mask]

    dl_downsampledd_points_removed_def = downsampled_points_removed_def.toDlpack()
    downsampled_t2cam_removed_def = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
    downsampled_t2cam_removed_def.point.positions = o3d.core.Tensor.from_dlpack(dl_downsampledd_points_removed_def) #o3d.core.Tensor(downsampled_points_removed_def, device = o3d.core.Device("CUDA:0")) 

    t_s_reg = time.time()
    reg_p2p = o3d.t.pipelines.registration.icp(source = downsampled_t2cam_removed_def.uniform_down_sample(every_k_points=10),
                                                target = downsampled_cropped_model_cu.uniform_down_sample(every_k_points=10),
                                                max_correspondence_distance = 0.02,
                                                estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                                                criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=100))
    t_e_reg = time.time()
    print("REG:",t_e_reg-t_s_reg)
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    
    if draw_reg: draw_registration_result(t2cam_pcd_cuda,cropped_model_cuda,frame)
    t2cam_pcd_cuda.transform(reg_p2p.transformation)
    #mesh.transform(reg_p2p.transformation)
    if draw_reg: draw_registration_result(t2cam_pcd_cuda,cropped_model_cuda,frame)
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

def inner_deformed_to_inner_undeformed(prev_valid_mask,shared_stream,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,t2cam_pcd_cuda,count,model_pcd,model_ply,draw_reg):
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
    start_cv2 = cv2.getTickCount()
    ## crop the undeformed model to the t2cam pcd size
    cropped_model_pcd, cropped_model_cuda = segment_pcd_using_bounding_box(t2cam_pcd_cuda,model_pcd,model_pcd_cuda)
    end_cv2 = cv2.getTickCount()
    time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
    print("SEGMENT:", time_sec)
    t_s_begin = time.time()
    
    ## register using ICP for better alignment
    fine_T = ICP_register_T2Cam_with_model(cropped_model_cuda,t2cam_pcd_cuda,t2cam_pcd_cuda.point.positions,cropped_model_cuda,draw_reg)
    t_e_begin = time.time()
    
    print("BEGIN:", t_e_begin-t_s_begin)

    ## OptiX ray tracing engine
    start_cv21 = cv2.getTickCount()
    with shared_stream:
        origins = cp.from_dlpack(t2cam_pcd_cuda.point.positions.contiguous().to_dlpack())
        directions = cp.from_dlpack(t2cam_pcd_cuda.point.normals.contiguous().to_dlpack())
        rays = cp.concatenate([origins, directions], axis=1) 

        hit_point, tri_id, t_hit, hit_point_o, tri_id_o, t_hit_o, hit_point_t, tri_id_t, t_hit_t = raycaster.cast(rays)

        #save hit_point, hit_point_o, origins
        np.savez_compressed(f"saved_arrays/iteration_{count:03d}.npz",
            orig=cp.asnumpy(origins),
            hit_p=cp.asnumpy(hit_point),
            hit_p_o=cp.asnumpy(hit_point_o)
        )
        
        print(hit_point_o)
        end_cv21 = cv2.getTickCount()
        time_sec = (end_cv21-start_cv21)/cv2.getTickFrequency()
        print("OptiX: ", time_sec)
    
        # shared_stream.synchronize()
        # cp.cuda.runtime.deviceSynchronize()
        # cp.cuda.Device().synchronize()

        # r_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        # r_pcd.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack())
        # o3d.visualization.draw([r_pcd.cpu()])

        print("hit_point_t shape", hit_point_t.shape)
        print("hit_point shape", hit_point.shape)
        print("t_hit shape", t_hit_t.shape)

        ## vector distance
        t_s = time.time()
        # dist = d_model_points - d_t2_points[idx]
        dist = hit_point - origins
        # cp_dist = cp.asarray(dist)
        ## scalar distance
        d_dist = t_hit
        print(d_dist.min(), d_dist.max())
        d_dist_o = t_hit_o
        print(d_dist_o[d_dist_o>=0.0].min(), d_dist_o.max())
        d_dist_t = t_hit_t
        print(d_dist_t[d_dist_t>=0.0].min(), d_dist_t.max())
        #d_dist = cp.linalg.norm(dist, axis=1)

        ## color correspodnece to signed distance according to 'plasma'
        d_colors = plt.get_cmap('plasma')(((d_dist - d_dist.min()) / (d_dist.max() - d_dist.min())).get())
        d_colors = d_colors[:, :3]

        d_colors_o = plt.get_cmap('plasma')(((d_dist_o - d_dist_o[d_dist_o>=0.0].min()) / (d_dist_o.max() - d_dist_o[d_dist_o>=0.0].min())).get())
        d_colors_o = d_colors_o[:, :3]

        d_colors_t = plt.get_cmap('plasma')(((d_dist_t - d_dist_t[d_dist_t>=0.0].min()) / (d_dist_t.max() - d_dist_t[d_dist_o>=0.0].min())).get())
        d_colors_t = d_colors_t[:, :3]

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
        outer_def = hit_point_o - dist
        tread_def = hit_point_t - dist
        
        t2_d_pcd_cu = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        #t2_d_pcd_cu.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack()) #downsampled_cropped_model_cu.point.positions
        t2_d_pcd_cu.point.positions = t2cam_pcd_cuda.point.positions
        t2_d_pcd_cu.point.normals = t2cam_pcd_cuda.point.normals
        t2_d_pcd_cu.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()
        
        # t2_d_pcd_cu.estimate_normals()
        # t2_d_pcd_cu.orient_normals_consistent_tangent_plane(k=10)
        t2_und_in = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und_in.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack()) #downsampled_cropped_model_cu.point.positions
        #t2_und_in.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()
        #t2_und_in = t2_und_in.uniform_down_sample(every_k_points = 10)
        
        
        t2_d_pcd_cu_o = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_d_pcd_cu_o.point.positions = o3d.core.Tensor.from_dlpack(outer_def.toDlpack()) #downsampled_cropped_model_cu.point.positions
        t2_d_pcd_cu_o.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()

        
        t2_d_pcd_cu_t = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_d_pcd_cu_t.point.positions = o3d.core.Tensor.from_dlpack(tread_def.toDlpack()) #downsampled_cropped_model_cu.point.positions
        t2_d_pcd_cu_t.point.colors = o3d.core.Tensor(d_colors, dtype = o3d.core.float32).cuda()

        t2_und = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und.point.positions = o3d.core.Tensor.from_dlpack(hit_point_o.toDlpack()) #downsampled_cropped_model_cu.point.positions
        #t2_und.point.colors = o3d.core.Tensor(d_colors_o, dtype = o3d.core.float32).cuda()
        t2_und = t2_und.uniform_down_sample(every_k_points = 10)

        t2_und_t = o3d.t.geometry.PointCloud(device = o3d.core.Device("CUDA:0"))
        t2_und_t.point.positions = o3d.core.Tensor.from_dlpack(hit_point_t.toDlpack()) #downsampled_cropped_model_cu.point.positions
        #t2_und_t.point.colors = o3d.core.Tensor(d_colors_t, dtype = o3d.core.float32).cuda()
        #t2_und_t = t2_und_t.uniform_down_sample(every_k_points = 10)

        #o3d.visualization.draw_geometries([t2_d_pcd_cu.cpu().to_legacy(),t2_und_in.to_legacy()])



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
        o3d_idx = o3d.core.Tensor.from_dlpack(tri_id_t.toDlpack())
        new = model_ply.triangle.normals[o3d_idx]

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
    with shared_stream:
        valid_mask_cp = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1)).get() #& (t_hit < -0.006)

        dl_d_t2_points = t2cam_pcd_cuda.point.positions.to_dlpack()
        cp_d_t2_points = cp.from_dlpack(dl_d_t2_points)
        mid_xyz = (cp.ptp(cp_d_t2_points,axis=0)/2 + cp.min(cp_d_t2_points,axis=0))
        
        print(mid_xyz)
        # dl_mid_xyz = mid_xyz.toDlpack()
        # o3d_mid_xyz = o3d.core.Tensor.from_dlpack(dl_mid_xyz)
        # frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1,device=o3d.core.Device("CUDA:0"))
        # frame.translate(o3d_mid_xyz)
        # create mask to select the outer points in the pcd to avoid registration with the deformed parts
        radius_mask = ((cp_d_t2_points[:,0]-mid_xyz[0])**2+(cp_d_t2_points[:,1]-mid_xyz[1])**2+((cp_d_t2_points[:,2]-mid_xyz[2])/1.2)**2) < 0.09**2
        # large_def_xyz = cp_d_t2_points[cp.argmax(t_hit[radius_mask])] 
        # large_def_radius_mask = ((cp_d_t2_points[:,0]-large_def_xyz[0])**2+(cp_d_t2_points[:,1]-large_def_xyz[1])**2+((cp_d_t2_points[:,2]-large_def_xyz[2])/1.2)**2) < 0.09**2
        valid_mask_cp_threshold = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & radius_mask).get()

        zero_def = ((tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & (t_hit > -0.008) & (t_hit < -0.007))
        valid_mask_radius_cp = (tri_id_t != 0) & (hit_point_t != 0.0).all(axis = 1) & radius_mask
        mask_smallest =  (t_hit - 0.01 < cp.min(t_hit[valid_mask_radius_cp])) & valid_mask_radius_cp

    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(t_hit_t)[valid_mask_cp], bins=500, color='steelblue', edgecolor='black')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of distances to plane")
    # plt.grid(True)
    # plt.tight_layout()
    # #plt.show(block=False)

    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(t_hit_t - t_hit)[valid_mask_cp], bins=500, color='steelblue', edgecolor='black')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of distances to plane")
    # plt.grid(True)
    # plt.tight_layout()
    # #plt.show(block=False)

    # print(t_hit_t.shape, t_hit.shape)
    # plt.figure(figsize=(6, 4))
    # plt.hist(cp.asnumpy(dist_tread[valid_mask_cp]), bins=500, color='steelblue', edgecolor='black')
    # plt.xlabel("Distance to fitted plane (m)")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of distances to plane")
    # plt.grid(True)
    # plt.tight_layout()
    #plt.show(block=False)

    

    # cp.cuda.runtime.deviceSynchronize()
    # cp.cuda.Device().synchronize()
    with shared_stream:
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
        extreme_def = t2_d_pcd_cu_t.select_by_mask(o3d.core.Tensor.from_dlpack((mask_smallest.astype(cp.uint8)).toDlpack()) != 0)
        #print(masked_pcd.point.positions.shape)
    #o3d.visualization.draw_geometries([masked_pcd.cpu().to_legacy()])
    t1 = time.perf_counter()
    #masked_pcd.estimate_normals() #.orient_normals_consistent_tangent_plane(k=10)
    t2 = time.perf_counter()
    #masked_pcd.orient_normals_consistent_tangent_plane(k=3)
    t3 = time.perf_counter()
    #masked_pcd.cuda()
    print("one", t2-t1)
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
    print("two", t3-t2)
    
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
        
        dist_to_plane = ((masked_pcd.point.positions).matmul(normal) + D).flatten()

        contact_patch = masked_pcd.select_by_mask(dist_to_plane.abs() < 0.001)
    
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
    print("REST: ",t_e-t_s)
    ## DEBUG Visualisation
    # o3d.visualization.draw_geometries([t2_d_pcd])
    # o3d.visualization.draw_geometries([downsampled_cropped_model,downsampled_t2cam])

    ## If want orthographic 3 graphs to be saved
    # figInner = vV.makeOrthoDeformationPlot(label = 'Inner', points=np.asarray(t2_d_pcd.points), dist = d_dist,vmin=-0.003,vmax=0.003)
    # cpu_dist = d_dist.get()
    # mask = (cpu_dist > -0.02) & (cpu_dist < 0.02)
    ## Save Figure 
    t__0 = time.perf_counter()
    #figInner,sc = vV.createOuterDeformationPlot(label = 'Inner', points=t2_d_pcd_cu.point.positions.cpu().numpy()[mask], dist = cpu_dist[mask],vmin=-0.01,vmax=0.01)
    #plt.savefig("./Inner_Deformation/{}.png".format(count))
    #plt.show()
    #save_quickly(figInner, "fast_image.png")
    #plt.close(figInner)
    #o3d.io.write_point_cloud("./Inner_Deformation/{}.pcd".format(count), t2_d_pcd)
    t__1 = time.perf_counter()
    print("Figure saving time: ",t__1-t__0)

    

    # cp.cuda.runtime.deviceSynchronize()
    # cp.cuda.Device().synchronize()
    #shared_stream.synchronize()
    print(masked_pcd.point.positions.shape)
    return fine_T, curr_valid_mask, t2_und_t, np.max(d_dist), cropped_model_cuda , d_dist, t2_d_pcd_cu_t, contact_patch,plane#t2_d_pcd_cu_t #1#t2_d_pcd_def

def calculate_surface_curvature(pcd, radius=0.1, max_nn=20):
    pcd_n = copy.deepcopy(pcd.to_legacy())
    pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    covs = np.asarray(pcd_n.covariances)
    vals, vecs = np.linalg.eig(covs)
    curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
    return curvature


def load_outer_inner_corres(file_name):
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
    return rays_hit_start_io, rays_hit_end_io

def draw_lines_lineset(start_points, end_points, line_set):
    line_start = cp.ascontiguousarray(cp.from_dlpack(start_points.to_dlpack()))
    line_end = cp.ascontiguousarray(cp.from_dlpack(end_points.to_dlpack()))

    dist = cp.linalg.norm(line_end-line_start,axis=1)
    # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # line_points = np.vstack((start_points, end_points))
    # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    valid_dist = dist < 0.1
    mask = (line_start != 0.0).all(axis=1) & (line_end != 0.0).all(axis=1) & valid_dist
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 
    
    disp = line_valid_end[:,2] - line_valid_start[:,2]
    #disp = (((local.reshape(-1,3))[::20])[mask])[:,2]

    if disp.shape[0] != 0:
        normalized =  (disp - disp.min()) / (disp.max() - disp.min())
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

def main():
    start_time = time.time()
    """
    Main function to excute the script
    
    PERFORM STEPS TO ULTIMATELY GET OUTER TREAD DEFROMATION USING OPEN3D POINTCLOUD PROCESSING AND OPENCV OPTICAL FLOW
    """
    #===========================================================================================================================================
    ## LOAD CONFIG
    with open("config.yaml", "r") as file:
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
    rays_hit_start_io, rays_hit_end_io = load_outer_inner_corres(inner_to_outer_np)
    rays_pcd = o3d.geometry.PointCloud()
    rays_pcd.points = o3d.utility.Vector3dVector(rays_hit_start_io)
    #kdtree = o3d.geometry.KDTreeFlann(rays_pcd)
    kdtree = cKDTree(rays_hit_start_io)

    ## LOAD INNER TO TREAD ONLY CORRESPONDENCE
    inner_to_treads_np = config["inner_to_treads"]
    rays_hit_start_it, rays_hit_end_it = load_outer_inner_corres(inner_to_treads_np)
    rays_tread_pcd = o3d.geometry.PointCloud()
    rays_tread_pcd.points = o3d.utility.Vector3dVector(rays_hit_start_it)
    kdtree_it = cKDTree(rays_hit_start_it)

    ## LOAD INNER MODELS
    #file_path_model = '4_row_model_HighPoly_Smoothed.ply'
    file_path_model = config["file_path_model"]#'full_outer_inner_part_only.ply'
    file_path_tread = config["file_path_tread"]

    ## CREATE INNER MODEL PCD from CORRES
    model_pcd = o3d.t.geometry.PointCloud()
    model_pcd.point.positions = o3d.core.Tensor(rays_hit_start_io) #[np.abs(rays_hit_start_io[:,2]-0.15) < 0.001]
    model_pcd.estimate_normals()
    model_pcd_cuda = model_pcd.cuda()

    ## LOAD PLY AND PCD OF INNER TREAD AND INNER FULL MODEL
    model_ply = load_model_ply(file_path_model,scale)
    model_tread_pcd = load_model_pcd(file_path_tread,scale)
    normals_model_pcd = model_pcd.point.normals.cuda()
    normals_tread_pcd = np.asarray(model_tread_pcd.normals)

    model_ply_outer = load_model_pcd("full_outer_outer_part_only.ply",scale)
   
    ## REALTIME = Realsense, FOLDER = imageStream
    if Real_Time:
        depth_profile=config["cam_depth_profile"]
        color_profile=config["cam_color_profile"]
        rsManager = RealSenseManager(depth_profile=depth_profile,color_profile=color_profile,exposure=config["exposure"],gain=config["gain"],enable_spatial=False,enable_temporal=False)
    else:
        depth_profile=config["image_depth_profile"]
        color_profile=config["image_color_profile"]
        imageStream = read_RGB_D_folder(config["RGB_D_folder"],starting_index=config["start_index"],depth_num=depth_profile,debug_mode=debug_mode)
        # with open(os.path.join(config["RGB_D_folder"],"time.npy"), 'rb') as f:
        #     time_arr = np.load(f)

    ## RUN APP to VIS IN REALTIME
    if view_video: viewer3d = Viewer3D("Outer Deformation and Tracking")

    ## RUN_ON_JETSON = apriltag library, WINDOWS = robotpy_apriltag library
    if Run_on_Jetson:
        from april_detect_jetson import DetectAprilTagsJetson
        detector = DetectAprilTagsJetson(depth_profile=depth_profile,debug_mode=debug_mode)
    else:
        from april_detect_windows import DetectAprilTagsWindows
        detector = DetectAprilTagsWindows(depth_profile=depth_profile,debug_mode=debug_mode)

    ## FIND PCD INDICES OF APRILTAG LOCATIONS 
    tag_norm, model_correspondence = find_tag_point_ID_correspondence(model_pcd_cuda, m_points)
    
    optmethod = config["opt_method"]
    cv2_stream = cv2.cuda.Stream()
    if optmethod == 'dense':
        dense = DenseOptFlow(depth_profile,debug_mode,config['dense_method'], cv2_stream)
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

    outer_model_ply = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
    outer_model_ply.compute_vertex_normals()
    outer_model_ply.compute_triangle_normals()
    outer_model_ply.normalize_normals()
    outer_model_ply = outer_model_ply.cuda()

    shared_stream = cp.cuda.Stream(non_blocking = True) #SET to FALSE if not taking DATA
    raycaster = OptiXRaycaster("full_outer_inner_smoothed_part_only.ply", "full_outer_outer_part_only.ply", "full_outer_treads_part_only.ply", "raycast.cu", shared_stream)

    load_time = time.time() - start_time
    print("Loading Completed in",load_time)

    outer_disp = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
    outer_vel =  o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
    prev_outer_deformed = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
    prev_outer_select = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
    prev_valid_mask = o3d.core.Tensor.empty((480,848),o3d.core.Dtype.Bool,o3d.core.Device("CUDA:0"))
    curr_t = o3d.core.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).cuda()
    ## Initialise PREV frame
    #==========================================================================================================================================================================
    start_loop_time = time.time()
    if Real_Time:
        time_ms, count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = rsManager.get_frames()
        print("IMAGE NUMBER:",count, time_ms)
    elif imageStream.has_next():
        count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = imageStream.get_next_frame()
        #time_ms = time_arr[count-150]
        print("IMAGE NUMBER:",count)#, time_ms)
    
    prev_t2cam_pcd_cuda = t2cam_pcd_cuda

    gpu_curr = cv2.cuda.GpuMat()
    gpu_curr_gray = cv2.cuda.GpuMat()
    
    dl_color = color_image.as_tensor().to_dlpack()
    cp_color = cp.from_dlpack(dl_color)
    color_ptr = cp_color.data.ptr

    gpu_prev = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_ptr)
    gpu_prev_gray = cv2.cuda.GpuMat()
    gpu_prev_gray_uint8 = gpu_prev.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
    gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev_gray_uint8, cv2.COLOR_BGR2GRAY)

    dense.init_disp(vertex_map_gpu)

    ## APRILTAG DETECT
    t_0 = time.time()
    detector.input_frame(color_image.as_tensor().cpu().numpy())
    t_1 = time.time()   
    tag_IDs, tag_locations, pcd_IDs = detector.process_3D_locations(vertex_map_gpu.as_tensor().cpu().numpy())
    t_2 = time.time() 
    time_at_detection = time.time()
    detect_time = time_at_detection - start_loop_time
    print("INPUT DATA:", t_0-start_loop_time)
    print("AprilTags Detected in", t_2-t_0)
    print("2D_process:", t_1-t_0)
    print("3D_process:", t_2-t_1)

    # ## APRILTAG CORRESPONDENCE
    # t2cam_correspondence_gpu = find_t2cam_correspondence(t2cam_pcd_cuda,np.array(tag_locations),debug_mode)
    # time_at_t2cam_corres = time.time()
    # t2cam_corres_time = time_at_t2cam_corres - time_at_detection
    # print("T2Cam Correspondence Point IDs calculated in:", t2cam_corres_time)

    # PREPROCESING FOR APRILTAG ALIGNMENT
    t2cam_correspondence_gpu = o3d.core.Tensor.from_numpy(np.array(pcd_IDs).astype(np.int64)).cuda()
    correspondence_vector,p,q = make_correspondence_vector(t2cam_correspondence_gpu,model_correspondence,tag_IDs,correctTags,debug_mode)
    time_at_corres_vec = time.time()
    corres_vec_time = time_at_corres_vec - detect_time#time_at_t2cam_corres
    print("Corres Vector made in", corres_vec_time)

    ## ROUGH ALIGNMENT WITH T2CAM USING APRILTAGS
    rough_T = register_t2cam_with_model(t2cam_pcd_cuda,model_pcd_cuda,correspondence_vector,p,q,curr_t)
    time_at_registration = time.time()
    register_time = time_at_registration - time_at_corres_vec
    print("Registration in", register_time)

    # with shared_stream:
    fine_T,init_mask, t2_d_pcd_inner_cu, max_inner_def, cropped_model_pcd_cu, d_inner_dist,t2_d_pcd_outer_cu, outer_select,plane = inner_deformed_to_inner_undeformed(prev_valid_mask,shared_stream,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,t2cam_pcd_cuda,count,model_pcd,outer_model_ply,draw_reg)
    time_at_Inner_c2c = time.time()
    c2c_dist_time = time_at_Inner_c2c - time_at_registration
    print("Inner and Outer C2C raycasting distance calculated in", c2c_dist_time)


    full_T = fine_T.matmul(rough_T)
    inv_full_T = full_T.inv()
    prev_outer_deformed.point.positions = t2_d_pcd_outer_cu.transform(inv_full_T).point.positions #.clone()
    #t2_d_pcd_outer_cu.point.positions = t2_d_pcd_outer_cu.point.positions - prev_outer_deformed.point.positions
    prev_outer_undeformed = t2_d_pcd_inner_cu
    prev_outer_select.point.positions = outer_select.transform(inv_full_T).point.positions #.clone()
    
    prev_valid_mask = init_mask.clone()
    #vis 2D flow INIT
    # gpu_hsv = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC3)
    # gpu_hsv_8u = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_8UC3)
    # gpu_h = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    # gpu_s = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    # gpu_v = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    # gpu_s.upload(np.ones_like(np.ones((gpu_prev.size()[1], gpu_prev.size()[0]), dtype=np.float32))) # set saturation to 1
    
    #SPARSE OPTICAL FLOW INIT
    # p0 = init_grid((480,848),step=20)
    # #p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY), mask=None, **dict(maxCorners=40000, qualityLevel=0.3, minDistance=7, blockSize=7))
    # gpu_prev_p = cv2.cuda.GpuMat()
    # gpu_prev_p.upload(p0.reshape(1,-1,2))
    # gpu_next_p = cv2.cuda.GpuMat(gpu_prev_p.size(), cv2.CV_32FC2)
    # gpu_status = cv2.cuda.GpuMat()
    # gpu_err = cv2.cuda.GpuMat()

    vertex_map_gpu_prev = vertex_map_gpu #o3d tensor gpu
    normal_map_gpu_prev = normal_map_gpu #o3d tensor gpu


    prev_mask = np.ones((480,848), dtype=bool)
    
    ## LOOP THROUGH EACH FRAME 1 onwards
    #==========================================================================================================================================================================
    try:
        while True:
            with nvtx.annotate("intake", color="red"):
                start_loop_time = time.time()
                start_cv2 = cv2.getTickCount()
                ## INPUT
                if Real_Time:
                    time_ms, count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = rsManager.get_frames()
                    print("IMAGE NUMBER:",count, time_ms)
                elif imageStream.has_next():
                    count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = imageStream.get_next_frame()
                    #time_ms = time_arr[count-150]
                    time_ms = 66.67
                    print("IMAGE NUMBER:",count)#, time_ms)
                else:
                    break

                dl_color_loop = color_image.as_tensor().to_dlpack()
                cp_color_loop = cp.from_dlpack(dl_color_loop)
                color_loop_ptr = cp_color_loop.data.ptr

                gpu_curr = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_loop_ptr)
                gpu_curr_gray_uint8 = gpu_curr.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
                gpu_curr_gray = cv2.cuda.cvtColor(gpu_curr_gray_uint8, cv2.COLOR_BGR2GRAY)

             ## APRILTAG DETECT
            cv2_stream.waitForCompletion()
            try:
                with nvtx.annotate("cpu april", color="blue"):
                    t_0 = time.time()
                    detector.input_frame(color_image.as_tensor().cpu().numpy())
                    t_1 = time.time()   
                    tag_IDs, tag_locations, pcd_IDs = detector.process_3D_locations(vertex_map_gpu.as_tensor().cpu().numpy())
                    t_2 = time.time() 
                    time_at_detection = time.time()
                    detect_time = time_at_detection - start_loop_time
                    print("INPUT DATA:", t_0-start_loop_time)
                    print("AprilTags Detected in", t_2-t_0)
                    print("2D_process:", t_1-t_0)
                    print("3D_process:", t_2-t_1)

                ## DENSE OPTICAL FLOW
                with nvtx.annotate("optic", color="green"):
                    frame_gpu, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = dense.detect2D(gpu_prev_gray, gpu_curr_gray)
                    
                    gpu_bgr = dense.vis_hsv_2D()
                    
                    map_x_gpu, map_y_gpu, traj_motion_2D_x, traj_motion_2D_y,traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5 = dense.detect3D(count, vertex_map_gpu,normal_map_gpu_prev, normal_map_gpu)
                    
                    dense.track_3D_vel(time_ms)
                    
                    g1,g2,d1,d2,frame,vel_arrow,curr_outer_pcd = dense.vis_3D()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("CUDA ERROR:", e)
                                
            ## SPARSE OPTICAL FLOW
            # gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y, gpu_next_p, gpu_status = sparse.detect2D(gpu_prev_gray, gpu_curr_gray)
            # bgr,frame = sparse.vis_grid_2D(gpu_curr_gray)
            # traj_motion_2D_x, traj_motion_2D_y,traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5 = sparse.detect3D(count,vertex_map_np,normal_map_prev)
            #bgr,frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y, gpu_next_p, gpu_status = Sparse_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray,gpu_prev_p,gpu_next_p,gpu_status,gpu_err)
            
            ## FEATURE DETECTION ONLY
            #FAST_feat(gpu_curr_gray)

            ## FEATURE DETECTION AND MATCHING
            #ORB_feat(gpu_prev_gray,gpu_curr_gray)
            #SURF_feat(gpu_prev_gray,gpu_curr_gray)

            ## VIS
            #bgr = vis_2D_flow_hsv(gpu_magnitude, gpu_angle,gpu_hsv_8u,gpu_hsv,gpu_h,gpu_s,gpu_v)
            #bgr = vis_2D_flow_vec(color_image,gpu_status,gpu_prev_p,gpu_next_p)
            
            #traj_motion_2D_x, traj_motion_2D_y,traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5= to_3D_disp(count, vertex_map_gpu_prev, vertex_map_prev, vertex_map_gpu, vertex_map_np, normal_map_prev, normal_map_np, gpu_flow_x, gpu_flow_y,traj_motion_2D_x, traj_motion_2D_y, traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5)
            
                # x_points = map_x_gpu.download()
                # y_points = map_y_gpu.download()

                # mask = (x_points <= 847) & (y_points <= 479) & (x_points >= 0) & (y_points >= 0)

                # #now create a border to indicate lost tracks that will clump into regions

                # # how to reseed tracks correctly
                # #now get the indices of those points lost and reinitialise with the INTERPOLATED VALUE of those neighbouring points of the index lost?
                # #                 
                # x_points = x_points[mask]
                # y_points = y_points[mask]

                # curr_mask = mask #& prev_mask

            # xy_pixels = np.argwhere(curr_mask)
             

            # Create 2D histogram
            # hist, xedges, yedges = np.histogram2d(x_points.ravel(), -y_points.ravel(), bins=(848,480))
            # hist[hist < 1] = -10
            
            
            # #Plot histogram (azimuth vs elevation)
            # plt.figure(figsize=(16, 8))
            # plt.imshow(hist.T, origin='lower', aspect='auto', vmin=-10, vmax=10) #,                         
            #         #extent=[-np.pi, np.pi, 0, np.pi])
            # plt.xlabel("x_track")
            # plt.ylabel("y_track")
            # plt.title("density track Histogram")
            # plt.colorbar(label="Count")
            # plt.savefig(f"track/{count:04d}.png")
            # plt.close()

            # print(x_points.ravel().shape)

                

            with nvtx.annotate("deform", color="yellow"):
                cv2_stream.waitForCompletion()
                try:
                    #with shared_stream:
                    ## APRILTAG CORRESPONDENCE
                    # t2cam_correspondence_gpu = find_t2cam_correspondence(curr_outer_pcd,np.array(tag_locations),debug_mode)
                    # time_at_t2cam_corres = time.time()
                    # t2cam_corres_time = time_at_t2cam_corres - time_at_detection
                    # print("T2Cam Correspondence Point IDs calculated in:", t2cam_corres_time)

                    # PREPROCESING FOR APRILTAG ALIGNMENT
                    t2cam_correspondence_gpu = o3d.core.Tensor.from_numpy(np.array(pcd_IDs).astype(np.int64)).cuda()
                    correspondence_vector,p,q = make_correspondence_vector(t2cam_correspondence_gpu,model_correspondence,tag_IDs,correctTags,debug_mode)
                    time_at_corres_vec = time.time()
                    corres_vec_time = time_at_corres_vec - detect_time#time_at_t2cam_corres
                    print("Corres Vector made in", corres_vec_time)

                    ## ROUGH ALIGNMENT WITH T2CAM USING APRILTAGS
                    rough_T = register_t2cam_with_model(curr_outer_pcd,model_pcd_cuda,correspondence_vector,p,q,curr_t)
                    time_at_registration = time.time()
                    register_time = time_at_registration - time_at_corres_vec
                    print("Registration in", register_time)

                    fine_T, curr_valid_mask,t2_d_pcd_inner_cu, max_inner_def, cropped_model_pcd_cu, d_inner_dist,t2_d_pcd_outer_cu, outer_select,plane = inner_deformed_to_inner_undeformed(prev_valid_mask,shared_stream,raycaster,inner_undef_lat,inner_undef,inner_lat,inner_arr,model_pcd_cuda,curr_outer_pcd,count,model_pcd,outer_model_ply,draw_reg)
                    time_at_Inner_c2c = time.time()
                    c2c_dist_time = time_at_Inner_c2c - time_at_registration
                    print("Inner and Outer C2C raycasting distance calculated in", c2c_dist_time)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("CUDA ERROR:", e)
                        
            #shared_stream.synchronize()
            full_T = fine_T.matmul(rough_T)
            inv_full_T = full_T.inv()
            with nvtx.annotate("vis rest", color="black"):
                cv2_stream.waitForCompletion()
                try:
                    # with shared_stream:
                        # print("t2_d_pcd_outer_cu shape", t2_d_pcd_outer_cu.point.positions.shape)
                        # print("prev_outer_deformed shape", prev_outer_deformed.point.positions.shape)
                        
                    curr_points = t2_d_pcd_outer_cu.transform(inv_full_T).select_by_mask(curr_valid_mask & prev_valid_mask).point.positions
                    prev_points = prev_outer_deformed.select_by_mask(curr_valid_mask & prev_valid_mask).point.positions

                    disp_points = curr_points.sub(prev_points)
                    vel_points = disp_points.div(time_ms/1000)               

                    curr_vel_points = prev_points.add(vel_points)

                    wrt_cam_outer_select = outer_select.transform(inv_full_T)

                    if view_video:
                        #draw_lines_lineset(prev_outer_deformed.select_by_mask(curr_valid_mask & prev_valid_mask).transform(inv_full_T).point.positions,t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask & prev_valid_mask).transform(inv_full_T).point.positions,outer_disp)
                        #draw_lines_lineset(prev_outer_deformed.select_by_mask(curr_valid_mask | prev_valid_mask).transform(inv_full_T).point.positions,t2_d_pcd_outer_cu.select_by_mask(curr_valid_mask | prev_valid_mask).transform(inv_full_T).point.positions,outer_disp)
                        #draw_lines_lineset(prev_points,curr_points,outer_disp)
                        #draw_lines_lineset(prev_points,curr_vel_points,outer_vel)
                        # viewer3d.update_cloud(geometries = t2_d_pcd_inner_cu.cpu(),lines = t2_d_pcd_outer_cu.cpu())
                        #viewer3d.update_cloud(undeformed_outer= t2_d_pcd_inner_cu.cpu(), deformed_outer = outer_select.cpu(), prev_def = prev_outer_select.cpu(), outer_disp = outer_disp)
                        #viewer3d.update_cloud(deformed_outer = wrt_cam_outer_select.cpu(), outer_disp = outer_disp.cpu(), outer_vel = outer_vel.cpu())
                        viewer3d.update_cloud(deformed_outer_shape = wrt_cam_outer_select.cpu(), outer_disp = outer_disp.cpu(), outer_vel = outer_vel.cpu())
                        #undeformed_outer= t2_d_pcd_inner_cu.transform(inv_full_T).cpu(),, prev_def = prev_outer_select.cpu(), plane = plane.transform(inv_full_T).cpu()
                        #viewer3d.update_cloud(outer_disp = outer_disp.cpu())# .paint_uniform_color(o3d.core.Tensor([1,0,0])) .paint_uniform_color(o3d.core.Tensor([0,1,0]))
                        #viewer3d.update_cloud(deformed_outer = outer_select.transform(inv_full_T).paint_uniform_color(o3d.core.Tensor([1,0,0])).cpu(), outer_disp = outer_disp)                    
                        viewer3d.tick()
                    time_at_app_view = time.time()
                    app_view_time = time_at_app_view - time_at_Inner_c2c
                    print("Time to update viewer", app_view_time)

                    # print("outer select shape", outer_select.point.positions.shape)
                    # print("prev outer select shape", prev_outer_select.point.positions.shape)

                    # o3d_disp = t2_d_pcd_outer_cu.point.positions - prev_outer_deformed.point.positions
                    # prev_outer_deformed.point.positions = t2_d_pcd_outer_cu.point.postions + o3d_disp #.clone()
                    prev_outer_deformed.point.positions = t2_d_pcd_outer_cu.point.positions #.clone()
                    prev_outer_undeformed = t2_d_pcd_inner_cu #.clone()
                    prev_valid_mask = curr_valid_mask.clone()
                    prev_outer_select.point.positions = outer_select.point.positions #.clone()
                
                    # t0 = time.time()
                    # if view_video:
                    #     #viewer3d.update_cloud(d1=d1.cpu(),d2=d2.cpu(),d3=d3.cpu(),d4=d4.cpu(),g5=g5.cpu())
                    #     viewer3d.update_cloud(g1=g1.cpu(),g2=g2.cpu(),d1=d1.cpu(),d2=d2.cpu(), frame = frame.cpu(), vel_arrow = vel_arrow.cpu())   
                    #     viewer3d.tick()
                    #     #o3d.visualization.draw([g1.cpu(),g2.cpu(),d1.cpu(),d2.cpu()])
                    # t1 = time.time()
                    # print("APP:", t1-t0)
                    
                    # gpu_prev_p = gpu_next_p

                    gpu_prev = gpu_curr
                    gpu_prev_gray = gpu_curr_gray

                    #prev_img = (color_image*255).astype(np.uint8)
                    # prev_t2cam_pcd = t2cam_pcd
                    prev_t2cam_pcd_cuda = t2cam_pcd_cuda

                    # vertex_map_prev = vertex_map_np
                    # normal_map_prev = normal_map_np
                    vertex_map_gpu_prev = vertex_map_gpu
                    normal_map_gpu_prev = normal_map_gpu
                

                    #prev_tag_locations = tag_locations
                    #correspondence vector (N,2,3) where idx = pointIDs or (H,W,2,3) vertex_map combination
                    #initial = prev_frame vertex map
                    # use time difference to calculate velocity 
                    # take vy and vx to get the slip in the contact patch region


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
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CUDA ERROR:", e)
    finally:
        if Real_Time: rsManager.stop()
        #if view_video: viewer3d.stop()
        max_len = 4000
        uniform_inner_arr = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in inner_arr
        ])

        uniform_outer_arr = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in outer_arr
        ])

        max_len = 4000
        uniform_inner_lat = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in inner_lat
        ])

        uniform_outer_lat = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in outer_lat
        ])

        max_len = 4000
        uniform_inner_arr_un = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in inner_undef
        ])

        uniform_outer_arr_un = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in outer_undef
        ])

        max_len = 4000
        uniform_inner_lat_un = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in inner_undef_lat
        ])

        uniform_outer_lat_un = np.array([
            np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) if a.shape[0] < max_len else a[:max_len]
            for a in outer_undef_lat
        ])

        # np.save('inner_arr.npy', uniform_inner_arr)
        # np.save('outer_arr.npy', uniform_outer_arr)
        # np.save('inner_lat.npy', uniform_inner_lat)
        # np.save('outer_lat.npy', uniform_outer_lat)
        # np.save('inner_arr_un.npy', uniform_inner_arr_un)
        # np.save('outer_arr_un.npy', uniform_outer_arr_un)
        # np.save('inner_lat_un.npy', uniform_inner_lat_un)
        # np.save('outer_lat_un.npy', uniform_outer_lat_un)
        sys.exit()



if __name__ == "__main__":
    main()