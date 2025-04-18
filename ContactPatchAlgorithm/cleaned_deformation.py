import cv2
import open3d as o3d
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from capture_realsense import RealSenseManager
from replay_realsense import read_RGB_D_folder
from SIFT_Detect import siftDetect
from Farneback_Detect import farnebackDetect
from LK_Dense import lkDense
from SURF_Detect import surfDetect
from app_vis import Viewer3D
import vidVisualiser as vV
import time
import yaml
from scipy.spatial import cKDTree
import cProfile
import faiss
import os

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
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    tag_norm = []
    model_correspondence = []
    for tag in range(0,len(m_points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(m_points[tag],1)
        model_correspondence.append(idx[0])
        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        m_norm = np.asarray(pcd.normals)[idx[1:], :]
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

def find_t2cam_correspondence(t2cam_pcd,tag_locations,debug_mode=True):
    """
    Function to find t2cam point IDs from tag locations 
    using np.where

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): The PointCloud captured by t2cam
        tag_locations (list): list of tag [x,y,z] locations

    Returns:
        indices (numpy.ndarray) A NumPy array of correspondence Point IDs
    """
    t2cam_points = np.asarray(t2cam_pcd.points)
    tag_locs = np.array(tag_locations)
    indices = []
    for xyz in tag_locs:
        idx = np.where((t2cam_points == xyz).all(axis=1))[0]
        if debug_mode: print(t2cam_points[idx],xyz)
        indices.append(idx)
    return np.array(indices).flatten()

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
    q = np.array([model_correspondence[correctTags.index(tag)] for tag in str_tags])
    pq = np.asarray([[p[i], q[i]] for i in range(len(p))]) 
    if debug_mode: print(pq) 
    corres = o3d.utility.Vector2iVector(pq)
    return corres

def register_t2cam_with_model(t2cam_pcd,model_pcd,corres_vector):
    """
    Function that transforms the T2Cam point cloud to register with the inner tyre model point cloud

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): open3d Point Cloud of T2Cam 
        model_pcd (o3d.geometry.PointCloud): open3d Point Cloud of inner tyre model
        corres_vector (o3d.utility.Vector2iVector): Correspondence array for registration 
    """
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T = estimator.compute_transformation(t2cam_pcd,model_pcd,corres_vector)
    t2cam_pcd.transform(T)
    
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
    o3d.visualization.draw_geometries([source_temp, target_temp,frame])

def segment_pcd_using_bounding_box(t2cam_pcd,model_pcd):
    """
    Segment the model point cloud to contain that section of t2Cam point cloud
    calculate bounding box of t2cam_pcd and crop model according to it

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): A Point Cloud of T2Cam image
        model_pcd (o3d.geometry.PointCloud): A Point Cloud of inner tyre model

    Returns:
        cropped_model (o3dd.geometry.PointCloud): a Point Cloud consisting of t2cam bounds
    """
    t2cam_bounding_box = t2cam_pcd.get_oriented_bounding_box() #.get_axis_aligned_bounding_box() 
    #bounding_box_center = t2cam_bounding_box.get_center()
    #t2cam_bounding_box.scale(scale = 1.5,center = bounding_box_center)
    cropped_model = model_pcd.crop(t2cam_bounding_box)
    return cropped_model

def ICP_register_T2Cam_with_model(cropped_model,t2cam_pcd, d_t2_points,downsampled_cropped_model, draw_reg):
    '''
    Performs Iterative Closest Point Registration to the Undeformed Cropped Inner Model and Deformed Inner T2Cam PCD

    Args:
        cropped_model (o3d.geometry.PointCloud): open3d Point Cloud Undeformed Cropped Inner Model
        t2cam_pcd (o3d.geometry.PointCloud): open3d Point Cloud 
        draw_reg (bool): A boolean to visualise the registration before and after result
    '''

    #find centre of pointcloud 
    mid_xyz = (np.ptp(d_t2_points,axis=0)/2 + np.min(d_t2_points,axis=0))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin = mid_xyz,size = 0.1)

    # create mask to select the outer points in the pcd to avoid registration with the deformed parts
    mask = ((d_t2_points[:,0]-mid_xyz[0])**2+(d_t2_points[:,1]-mid_xyz[1])**2+((d_t2_points[:,2]-mid_xyz[2])/1.2)**2) > 0.12**2
    
    # create a new pcd with the masked downsampled pointcloud
    downsampled_points_removed_def = d_t2_points[mask]
    downsampled_t2cam_removed_def = o3d.geometry.PointCloud()
    downsampled_t2cam_removed_def.points = o3d.utility.Vector3dVector(downsampled_points_removed_def) 

    print("Apply point-to-plane ICP")
    t_s_reg = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(downsampled_t2cam_removed_def, downsampled_cropped_model, max_correspondence_distance = 0.02,
                                                          estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                          criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8,relative_rmse=1.000000e-08,max_iteration=100))
    t_e_reg = time.time()
    print("REG:",t_e_reg-t_s_reg)
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    if draw_reg: draw_registration_result(t2cam_pcd,cropped_model,frame)
    t2cam_pcd.transform(reg_p2p.transformation)
    #if draw_reg: draw_registration_result(downsampled_t2cam_removed_def, downsampled_t2cam, reg_p2p.transformation,frame)
    if draw_reg: draw_registration_result(t2cam_pcd,cropped_model,frame)


def inner_deformed_to_inner_undeformed(count,t2cam_pcd,model_pcd,model_ply,draw_reg):
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
    t_seg_begin = time.time()
    ## crop the undeformed model to the t2cam pcd size
    cropped_model_pcd = segment_pcd_using_bounding_box(t2cam_pcd,model_pcd)
    t_seg_end = time.time()
    print("SEGMENT:", t_seg_end-t_seg_begin)
    t_s_begin = time.time()
    ## downsample t2cam and cropped model and store its respective points
    downsampled_t2cam = t2cam_pcd.uniform_down_sample(every_k_points=10)
    downsampled_cropped_model = cropped_model_pcd.uniform_down_sample(every_k_points=10)
    #downsampled_t2cam = t2cam_pcd
    #downsampled_cropped_model = cropped_model_pcd
    d_t2_points = np.asarray(downsampled_t2cam.points)
    d_model_points = np.asarray(downsampled_cropped_model.points)
    t_e_begin = time.time()
    ## register using ICP for better alignment
    ICP_register_T2Cam_with_model(cropped_model_pcd,t2cam_pcd,d_t2_points,downsampled_cropped_model,draw_reg)
    
    ## downsample t2cam and cropped model and store its respective points
    downsampled_t2cam = t2cam_pcd.uniform_down_sample(every_k_points=10)
    d_t2_points = np.asarray(downsampled_t2cam.points)
    
    print("BEGIN:", t_e_begin-t_s_begin)
    ## CANT USE .compute_point_cloud_distance as it does not give vector distance
    #d_dist = np.asarray(t2cam_pcd.compute_point_cloud_distance(downsampled_cropped_model))
    #d_dist = np.asarray(downsampled_cropped_model.compute_point_cloud_distance(downsampled_t2cam))

    ## OPEN3D KDtree implementation needs for loop
    # t_s_o3d = time.time()
    # downsampled_t2cam_pcd_tree = o3d.geometry.KDTreeFlann(downsampled_t2cam)
    # idx = [(downsampled_t2cam_pcd_tree.search_knn_vector_3d(xyz, 1))[1][0] for xyz in d_model_points]
    # t_e_o3d = time.time()
    # print("Open3D: ",t_e_o3d-t_s_o3d)

    ## SciPy cKDTree implementation 
    t_s_scipy = time.time()
    tree = cKDTree(d_t2_points)
    _,idx = tree.query(d_model_points,1)
    t_e_scipy = time.time()
    print("SCIPY: ",t_e_scipy-t_s_scipy)

    ## FAISS 
    # t_s_faiss = time.time()
    # cpu_index = faiss.IndexFlatL2(3)

    # # Move the index to GPU
    # gpu_res = faiss.StandardGpuResources()  # Initialize GPU resources
    # gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)  # 0 = GPU ID

    # # Add points to GPU index
    # gpu_index.add(d_t2_points)

    # # Search on GPU
    # _, idx = gpu_index.search(d_model_points, 1)
    # t_e_faiss = time.time()
    # print("FAISS: ",t_e_faiss-t_s_faiss)

    ## vector distance
    t_s = time.time()
    dist = d_model_points - d_t2_points[idx]

    ## scalar distance
    d_dist = np.linalg.norm(dist, axis=1)
    
    ## normalised normals of model 
    normals_undeformed = np.asarray(downsampled_cropped_model.normals)
    normals_undeformed /= np.linalg.norm(normals_undeformed, axis=1, keepdims=True) + 1e-8

    ## Calculated Signed Distance (dot of the vector distance and normal: if < 90 = 1 if > 90 = -1)
    dot_product = (-1)*np.sign(np.sum(dist * normals_undeformed, axis=1))
    d_dist = dot_product*d_dist

    ## color correspodnece to signed distance according to 'plasma'
    d_colors = plt.get_cmap('plasma')((d_dist - d_dist.min()) / (d_dist.max() - d_dist.min()))
    d_colors = d_colors[:, :3]

    # Create new pcd with color correspondence to deformation
    t2_d_pcd = o3d.geometry.PointCloud()
    t2_d_pcd.points = downsampled_cropped_model.points
    t2_d_pcd.colors = o3d.utility.Vector3dVector(d_colors)
    t_e = time.time()
    print("REST: ",t_e-t_s)
    ## DEBUG Visualisation
    #o3d.visualization.draw_geometries([t2_d_pcd])
    #o3d.visualization.draw_geometries([downsampled_cropped_model,downsampled_t2cam])

    ## If want orthographic 3 graphs to be saved
    #figInner = vV.makeOrthoDeformationPlot(label = 'Inner', points=np.asarray(t2_d_pcd.points), dist = d_dist,vmin=-0.003,vmax=0.003)
    
    ## Save Figure 
    t__0 = time.perf_counter()
    figInner,sc = vV.createOuterDeformationPlot(label = 'Inner', points=np.asarray(t2_d_pcd.points), dist = d_dist,vmin=-0.01,vmax=0.01)
    plt.savefig("./Inner_Deformation/{}.png".format(count))
    #plt.show()
    #save_quickly(figInner, "fast_image.png")
    plt.close(figInner)
    #o3d.io.write_point_cloud("./Inner_Deformation/{}.pcd".format(count), t2_d_pcd)
    t__1 = time.perf_counter()
    print("Figure saving time: ",t__1-t__0)
    return t2_d_pcd, np.max(d_dist), downsampled_cropped_model, d_dist

def run_vis_app(t2cam_pcd,cropped_model_pcd):
    '''
    Visualisation Application for two pcds

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): Open3D PointCloud 1
        cropped_model_pcd (o3d.geometry.PointCloud): Open3D PointCloud 2
    '''
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    o3dvis = o3d.visualization.O3DVisualizer()
    o3dvis.show_settings = True

    o3dvis.add_geometry("T2Cam",t2cam_pcd)
    o3dvis.add_geometry("my points",cropped_model_pcd)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:        
        '''visualize'''
        o3dvis.reset_camera_to_default()
        app.add_window(o3dvis)
        app.run()

def vis_apriltag_motion(prev_tag_locs,curr_tag_locs):
    '''
    Visualise Apriltag Movement between two consecutive frames

    Args:
        prev_tag_locs (numpy.ndarray): Previous Frame AprilTag Locations
        curr_tag_locs (numpy.ndarray): Current Frame AprilTag Locations
    '''
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    prev_pcd = o3d.t.geometry.PointCloud(device)
    prev_pcd.point.positions = o3d.core.Tensor(prev_tag_locs, dtype, device)
    curr_pcd = o3d.t.geometry.PointCloud(device)
    curr_pcd.point.positions = o3d.core.Tensor(curr_tag_locs, dtype, device)
    o3d.visualization.draw_geometries([prev_pcd.to_legacy(),curr_pcd.to_legacy()])

def find_points(prev_3D_points, curr_3D_points, prev_t2cam_pcd, curr_t2cam_pcd):
    '''
    Finds the Point Indices of the tag locations of two consective frames 

    Args:
        prev_3D_points (numpy.ndarray): A NumPy (N,3) array of the tag location xyz coordinates of previous frame
        curr_3D_points (numpy.ndarray): A NumPy (N,3) array of the tag location xyz coordinates of current frame
        prev_t2cam_pcd (o3d.geometry.PointCloud): A Open3D PointCloud of previous frame
        curr_t2cam_pcd (o3d.geometry.PointCloud): A Open3D PointCloud of current frame

    Returns:
        Tuple:
           prev_indices (np.ndarray): A NumPy 1D array containing indices of the Previous PointCloud Tag Locations
           curr_indices (np.ndarray): A NumPy 1D array containing indices of the Current PointCloud Tag Locations
    '''
    t2 = time.time()
    prev_kdtree = o3d.geometry.KDTreeFlann(prev_t2cam_pcd)
    curr_kdtree = o3d.geometry.KDTreeFlann(curr_t2cam_pcd)

    prev_indices = np.array([prev_kdtree.search_knn_vector_3d(xyz, 1)[1][0] for xyz in prev_3D_points], dtype=np.int32)
    curr_indices = np.array([curr_kdtree.search_knn_vector_3d(xyz, 1)[1][0] for xyz in curr_3D_points], dtype=np.int32)    

    t3 = time.time()
    print("find SIFT point IDS",t3-t2)
    return prev_indices,curr_indices

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

def projectOuter(count,kdtree,rays_hit_start_io,rays_hit_end_io,cropped_model_pcd,d_dist,normals,t2_d_pcd):
    '''
    Function to use inner_to_outer.npz correspondence and get the SIGNED corresponding distances between 
    the Outer Undeformed Model and the Projected Outer T2Cam  

    Args:
        count (int): A counter to indicate frame number
        kdtree (o3d.geometry.KDTreeFlann): A kdtree of the inner model from rays_hit_start
        rays_hit_start_io (numpy.ndarray): A NumPy (N,3) array of the start ray location (Inner Undeformed Model)
        rays_hit_end_io (numpy.ndarray): A NumPy (N,3) array of the ray end/hit location (Outer Undeformed Model)
        cropped_model_pcd (o3d.geometry.PointCloud): Open3D PointCloud of cropped inner model
        d_dist (numpy.ndarray): Scalar Signed distance of inner deformed to inner undeformed 
        normals (np.ndarray): A NumPy (N,3) array of the model_pcd normals
        t2_d_pcd (o3d.geometry.PointCloud): Open3D PointCloud of the Color Mapped t2Cam pcd

    Returns:
        t2_d_pcd_outer (o3d.geometry.PointCloud): Returns the Outer Deformed pcd with a colormap applied representing deformed distances
        max_d_outer_dist (np.float64): Max scalar distance
        cropped_model_pcd (o3d.geometry.PointCloud): Open3D model of cropped inner undeformed
        d_outer_dist (np.ndarray): A NumPy 1D array of signed distances from outer undeformed to outer deformed
        model_outer_deformed (o3d.geometry.PointCloud): Open3D PointCloud of Outer Deformed
        model_outer_undeformed (o3d.geometry.PointCloud): Open3D PointCloud of Outer Undeformed Model
        d_colors_outer (np.ndarray): A NumPy (N,3) RGB array of color mapped distances
    '''
    t0 = time.time()
    cropped_points = np.asarray(cropped_model_pcd.points)

    ## Finding index using np.where
    #idx = [np.where((rays_hit_start_io == xyz).all(axis=1))[0] for xyz in cropped_points]

    ## Finding index using Open3D KDTree 
    #idx = [(kdtree.search_knn_vector_3d(xyz, 1))[1][0] for xyz in cropped_points]
    #print(idx)

    # Finding index using SciPy cKDTree 
    t_0 = time.perf_counter()
    _,idx = kdtree.query(cropped_points,1)
    t_1 = time.perf_counter()
    print(t_1-t_0)

    # Create Outer Undeformed and Deformed PointClouds and use the normals and signed distances to get the deformed pcd
    model_outer_deformed = o3d.geometry.PointCloud()
    model_outer_deformed.points = o3d.utility.Vector3dVector(rays_hit_end_io[idx] + d_dist[:, None] * normals[idx])
    model_outer_undeformed = o3d.geometry.PointCloud()
    model_outer_undeformed.points = o3d.utility.Vector3dVector(rays_hit_end_io[idx])
    
    # Assume rubber is incompressible: Therefore, outer tread moves the same as the inner carcass
    d_outer_dist = d_dist

    # Color map according to distance
    d_colors_outer = plt.get_cmap('plasma')((d_outer_dist - d_outer_dist.min()) / (d_outer_dist.max() - d_outer_dist.min()))
    d_colors_outer = d_colors_outer[:, :3]

    # Create colormapped PointCloud
    t2_d_pcd_outer = o3d.geometry.PointCloud()
    t2_d_pcd_outer.points = model_outer_deformed.points
    t2_d_pcd_outer.colors = o3d.utility.Vector3dVector(d_colors_outer)

    t1 = time.time()
    print(t1-t0)

    # DEBUG VIS OF OUTER DEFORMED AND UNDEFORMED
    #o3d.visualization.draw_geometries([model_outer_deformed,model_outer_undeformed])

    # DEBUG VIS OF DEFORMED INNER AND OUTER
    #o3d.visualization.draw_geometries([t2_d_pcd_outer, t2_d_pcd])

    # CREATE ORTHOGRAPHIC Fig for 3 views of Outer Deformation
    #figOuter = vV.makeOrthoDeformationPlot(label = 'Outer', points=np.asarray(t2_d_pcd_outer.points), dist = d_outer_dist,vmin=-0.003,vmax=0.003)
    
    #Save Fig
    t__0 = time.perf_counter()
    figOuter,sc = vV.createOuterDeformationPlot(label = 'Outer', points=np.asarray(t2_d_pcd_outer.points), dist = d_outer_dist,vmin=-0.01,vmax=0.01)
    #vV.updateScatterPlot(sc, np.asarray(t2_d_pcd_outer.points), d_outer_dist)
    plt.savefig("./Outer_Deformation/{}.png".format(count))
    #plt.show()
    plt.close(figOuter)
    t__1 = time.perf_counter()
    print("Figure saving time: ",t__1-t__0)

    return t2_d_pcd_outer, np.max(d_outer_dist), cropped_model_pcd, d_outer_dist, model_outer_deformed,model_outer_undeformed, d_colors_outer

def extract_inner_contact_patch_edge(count,t2_d_pcd_inner,d_inner_dist):
    eps = 0.001
    mask = np.abs(d_inner_dist) < eps
    points = np.asarray(t2_d_pcd_inner.points)
    mid_xyz = (np.ptp(points,axis=0)/2 + np.min(points,axis=0))
    contact_patch_inner = o3d.geometry.PointCloud()
    contact_patch_inner.points = o3d.utility.Vector3dVector(points[mask])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05,origin=mid_xyz)
    #o3d.visualization.draw_geometries([contact_patch_inner,mesh_frame])
    #figInnerContact = vV.visContactEdge("Inner",points[mask],-0.01,0.01,d_inner_dist[mask])
    #plt.savefig("./Inner_Contact_Edge/{}.png".format(count))
    #plt.show()
    #plt.close(figInnerContact)

# def extract_outer_contact_patch_edge(count,t2_d_pcd_outer,d_outer_dist,model_outer_undeformed,d_colors_outer,rays_hit_start_it,rays_hit_end_it,kdtree_it,cropped_model_pcd,d_dist,normals_tread_pcd):
#     t0 = time.time()
#     cropped_points = np.asarray(cropped_model_pcd.points)
#     _,idx = kdtree_it.query(cropped_points,1)
#     model_outer_deformed = o3d.geometry.PointCloud()
#     model_outer_deformed.points = o3d.utility.Vector3dVector(rays_hit_end_it[idx] + d_dist[:,None] * normals_tread_pcd[idx])
#     model_outer_undeformed = o3d.geometry.PointCloud()
#     model_outer_undeformed.points = o3d.utility.Vector3dVector(rays_hit_end_it[idx])
#     d_outer_dist = d_dist

#     d_colors_outer = plt.get_cmap('plasma')((d_outer_dist - d_outer_dist.min()) / (d_outer_dist.max() - d_outer_dist.min()))
#     d_colors_outer = d_colors_outer[:, :3]

#     t2_d_pcd_outer = o3d.geometry.PointCloud()
#     t2_d_pcd_outer.points = model_outer_deformed.points
#     t2_d_pcd_outer.colors = o3d.utility.Vector3dVector(d_colors_outer)

#     t1 = time.time()
#     print(t1-t0)
#     o3d.visualization.draw_geometries([model_outer_deformed,model_outer_undeformed])

#     t2_d_pcd_outer.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
#     eps = 0
#     points = np.asarray(t2_d_pcd_outer.points)
#     colors = np.asarray(t2_d_pcd_outer.colors)
#     mid_xyz = (np.ptp(points,axis=0)/2 + np.min(points,axis=0))
#     mask = (d_outer_dist > eps) & (points[:,1] > mid_xyz[1]) & (d_outer_dist > 0.001) & (points[:,0] < mid_xyz[0]+0.10)
#     contact_patch_outer = o3d.geometry.PointCloud()
#     contact_patch_outer.points = o3d.utility.Vector3dVector(points[mask])
#     contact_patch_outer.colors = o3d.utility.Vector3dVector(colors[mask])
#     undef = o3d.geometry.PointCloud()
#     #undef.points = o3d.utility.Vector3dVector(np.asarray(model_outer_undeformed))
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05,origin=mid_xyz)
#     o3d.visualization.draw_geometries([contact_patch_outer,mesh_frame])

#     figOuterContact = vV.visContactEdge("Outer",points[mask],-0.01,0.01,d_outer_dist[mask])
#     plt.savefig("./Outer_Contact_Edge/{}.png".format(count))
#     #plt.show()
#     plt.close(figOuterContact)


def main():
    start_time = time.time()
    """
    Main function to excute the script
    
    PERFORM STEPS TO ULTIMATELY GET OUTER TREAD DEFROMATION USING OPEN3D POINTCLOUD PROCESSING AND OPENCV OPTICAL FLOW
    """

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
    #model_pcd = load_model_pcd(file_path_model,scale)
    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(rays_hit_start_io)
    model_pcd.estimate_normals()

    ## LOAD PLY AND PCD OF INNER TREAD AND INNER FULL MODEL
    model_ply = load_model_ply(file_path_model,scale)
    model_tread_pcd = load_model_pcd(file_path_tread,scale)
    normals_model_pcd = np.asarray(model_pcd.normals)
    normals_tread_pcd = np.asarray(model_tread_pcd.normals)

    ## FIND PCD INDICES OF APRILTAG LOCATIONS 
    tag_norm, model_correspondence = find_tag_point_ID_correspondence(model_pcd, m_points)
    tag_normals_line_set = draw_lines(m_points, tag_norm)

    ## DEBUG
    if debug_mode: vis_window(model_pcd, tag_normals_line_set)
    
    ## REALTIME = Realsense, FOLDER = imageStream
    if Real_Time:
        depth_profile=config["cam_depth_profile"]
        color_profile=config["cam_color_profile"]
        rsManager = RealSenseManager(depth_profile=depth_profile,color_profile=color_profile,exposure=config["exposure"],gain=config["gain"],enable_spatial=False,enable_temporal=False)
    else:
        depth_profile=config["image_depth_profile"]
        color_profile=config["image_color_profile"]
        imageStream = read_RGB_D_folder(config["RGB_D_folder"],starting_index=config["start_index"],depth_num=depth_profile,debug_mode=debug_mode)
        with open(os.path.join(config["RGB_D_folder"],"time.npy"), 'rb') as f:
            time_arr = np.load(f)

    ## RUN_ON_JETSON = apriltag library, WINDOWS = robotpy_apriltag library
    if Run_on_Jetson:
        from april_detect_jetson import DetectAprilTagsJetson
        detector = DetectAprilTagsJetson(depth_profile=depth_profile,debug_mode=debug_mode)
    else:
        from april_detect_windows import DetectAprilTagsWindows
        detector = DetectAprilTagsWindows(depth_profile=depth_profile,debug_mode=debug_mode)

    ## RUN APP to VIS IN REALTIME
    if view_video: viewer3d = Viewer3D("Distance Deformation")

    ## OPTICAL FLOW INITIALISATION
    sift = siftDetect(depth_profile,debug_mode)
    farne = farnebackDetect(depth_profile,debug_mode)
    lkdense = lkDense(depth_profile,debug_mode)
    surf = surfDetect(depth_profile,debug_mode)

    load_time = time.time() - start_time
    print("Loading Completed in",load_time)
    
    ## OPTICAL FLOW FOR CONSECUTIVE COMPARISON
    prev_img = None
    prev_rgbd_img = None
    prev_tag_locations = None
    prev_flag = False
    idno = 0

    ## LOOP THROUGH EACH FRAME
    try:
        while True:
            start_loop_time = time.time()

            ## INPUT
            if Real_Time:
                time_ms, count, depth_image, color_image, rgbd_image, t2cam_pcd = rsManager.get_frames()
                print("IMAGE NUMBER:",count)
            elif imageStream.has_next():
                count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
                time_ms = time_arr[count]
                print("IMAGE NUMBER:",count)
            else:
                break
            
            ## APRILTAG DETECT
            detector.input_frame(color_image)
            tag_IDs, tag_locations = detector.process_3D_locations(rgbd_image)
            time_at_detection = time.time()
            detect_time = time_at_detection - start_loop_time
            print("AprilTags Detected in", detect_time)
        
            ## APRILTAG CORRESPONDENCE
            t2cam_correspondence = find_t2cam_correspondence(t2cam_pcd,tag_locations,debug_mode)
            time_at_t2cam_corres = time.time()
            t2cam_corres_time = time_at_t2cam_corres - time_at_detection
            print("T2Cam Correspondence Point IDs calculated in:", t2cam_corres_time)

            # PREPROCESING FOR APRILTAG ALIGNMENT
            correspondence_vector = make_correspondence_vector(t2cam_correspondence,model_correspondence,tag_IDs,correctTags,debug_mode)
            time_at_corres_vec = time.time()
            corres_vec_time = time_at_corres_vec - time_at_t2cam_corres
            print("Corres Vector made in", corres_vec_time)

            ## ROUGH ALIGNMENT WITH T2CAM USING APRILTAGS
            register_t2cam_with_model(t2cam_pcd,model_pcd,correspondence_vector)
            time_at_registration = time.time()
            register_time = time_at_registration - time_at_corres_vec
            print("Registration in", register_time)

            ## DEBUG ROUGH ALIGNEMENT
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # vis.add_geometry(t2cam_pcd)
            # pcdt2points = np.asarray(t2cam_pcd.points)
            # pcdmodelpoints = np.asarray(model_pcd.points)
            # for xyz1,xyz2 in np.array(correspondence_vector):
            #     frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.005, origin = pcdt2points[xyz1])
            #     frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.005, origin = pcdmodelpoints[xyz2])
            #     vis.add_geometry(frame1)
            #     #vis.add_geometry(frame2)
            #     vis.update_renderer()
            #     vis.poll_events()
            # vis.run()
            # vis.destroy_window()

            ## INNER SIGNED DISTANCE DEFORMATION CALCULATION WITH REFINED ICP REGISTRATION
            t2_d_pcd_inner, max_inner_def, cropped_model_pcd, d_inner_dist = inner_deformed_to_inner_undeformed(count,t2cam_pcd,model_pcd,model_ply,draw_reg)
            time_at_Inner_c2c = time.time()
            c2c_dist_time = time_at_Inner_c2c - time_at_registration
            print("Inner C2C distance calculated in", c2c_dist_time)

            ## OUTER TREAD DEFORMATION
            t2_d_pcd_outer, max_outer_def, cropped_model_pcd, d_outer_dist, model_outer_deformed,model_outer_undeformed,d_outer_colors = projectOuter(count,kdtree,rays_hit_start_io,rays_hit_end_io,cropped_model_pcd,d_inner_dist,normals_model_pcd,t2_d_pcd_inner)
            time_at_Outer_c2c = time.time()
            c2c_dist_time = time_at_Outer_c2c - time_at_Inner_c2c
            print("Outer C2C distance calculated in", c2c_dist_time)
            #o3d.visualization.draw_geometries([t2cam_dist_pcd])

            ## UPDATE VIEWER
            if view_video:
                viewer3d.update_cloud(t2_d_pcd_inner,t2_d_pcd_outer)
                viewer3d.tick()
            time_at_app_view = time.time()
            app_view_time = time_at_app_view - time_at_Outer_c2c
            print("Time to update viewer", app_view_time)

            ## GET THE EDGE INNER AND TREAD ONLY OUTER            
            extract_inner_contact_patch_edge(count,t2_d_pcd_inner,d_inner_dist)
            #extract_outer_contact_patch_edge(count,t2_d_pcd_outer,d_outer_dist,model_outer_undeformed,d_outer_colors,rays_hit_start_it,rays_hit_end_it,kdtree_it,cropped_model_pcd,d_inner_dist,normals_tread_pcd)
            #prev_3D_points, curr_3D_points = surf.detect(color_image,prev_rgbd_img,rgbd_image, max_inner_def, t2cam_pcd)
            # OPTICAL FLOW 
            #if prev_flag == True: 
                #o3d.visualization.draw_geometries([prev_t2cam_pcd,t2cam_pcd])
                #prev_3D_points, curr_3D_points = sift.detect(prev_img,color_image,prev_rgbd_img,rgbd_image, max_inner_def, t2cam_pcd)
            
            prev_3D_points, curr_3D_points = farne.detect(color_image, prev_rgbd_img, rgbd_image, 0.03)
                #prev_3D_points, curr_3D_points = lkdense.detect(prev_img, color_image, prev_rgbd_img, rgbd_image, max_inner_def)
                #vis_apriltag_motion(prev_tag_locations,tag_locations)

                #prev_idx,curr_idx = find_points(prev_3D_points,curr_3D_points,prev_t2cam_pcd,t2cam_pcd)

            prev_img = color_image
            prev_rgbd_img = rgbd_image
            prev_t2cam_pcd = t2cam_pcd
            #prev_tag_locations = tag_locations
            prev_flag = True
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            if debug_mode:
                key = cv2.waitKey(1)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.09), cv2.COLORMAP_JET)
                images = np.hstack((detector.color_image,depth_colormap))
                cv2.namedWindow('AprilTag Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('AprilTag Detection', 1000, 400)
                cv2.imshow('AprilTag Detection', images)

                print("============APRILTAG=DETECTIONS============")
                print("Number of Detections:",detector.number_of_detections)
                print("Number of Failures:", detector.fail_count_per_frame)
                print("===========================================")

                print("==============CORRESPONDENCES==============")
                print("T2Cam point IDs",t2cam_correspondence)
                print("Model point IDS", model_correspondence)
                print("===========================================")
                if key == 27:
                    cv2.destroyAllWindows()
                    if view_video: viewer3d.stop()
                    break
            print("Time for one frame:",time.time() - start_loop_time)

    finally:
        if Real_Time: rsManager.stop()
        #if view_video: viewer3d.stop()


if __name__ == "__main__":
    main()