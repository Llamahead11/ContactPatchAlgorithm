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

def Brox_optflow(gpu_prev_gray, gpu_curr_gray):
    brox = cv2.cuda.BroxOpticalFlow.create(alpha=0.197,
                                               gamma=5.0,
                                               scale_factor=0.8,
                                               inner_iterations=5,
                                               outer_iterations=150,
                                               solver_iterations=10
    )
    gpu_flow = cv2.cuda.GpuMat(gpu_prev_gray.size(),cv2.CV_32FC2)
    gpu_prev_gray_f32 = cv2.cuda_GpuMat(gpu_prev_gray.size(), cv2.CV_32FC1)
    gpu_curr_gray_f32 = cv2.cuda_GpuMat(gpu_curr_gray.size(), cv2.CV_32FC1)

    gpu_prev_gray.convertTo(dst=gpu_prev_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)
    gpu_curr_gray.convertTo(dst=gpu_curr_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)

    gpu_flow_x = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32F)
    gpu_flow_y = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32F)
    brox.calc(I0=gpu_prev_gray_f32,I1=gpu_curr_gray_f32,flow=gpu_flow)


    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )
    frame = gpu_curr_gray.download()
    return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

def Dense_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray):
    gpu_flow = cv2.cuda.DensePyrLKOpticalFlow.create(winSize=(15,15),
                                                     maxLevel=5,
                                                     iters=300,
                                                     useInitialFlow=False
    )
    # calculate optical flow
    gpu_flow = cv2.cuda.DensePyrLKOpticalFlow.calc(
        gpu_flow, gpu_prev_gray, gpu_curr_gray, None,
    )

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )
    frame = gpu_curr_gray.download()
    return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

def Farne_opt_flow(gpu_prev_gray,gpu_curr_gray):
    gpu_flow = cv2.cuda.FarnebackOpticalFlow.create(numLevels=15,
                                                    pyrScale=0.5,
                                                    fastPyramids=True,
                                                    winSize=5,
                                                    numIters=10,
                                                    polyN=7,
                                                    polySigma=1.5,
                                                    flags=0
       
    )# 15, 0.5, False, 5, 6, 7, 1.5, 0
    # calculate optical flow
    gpu_flow = cv2.cuda.FarnebackOpticalFlow.calc(
        gpu_flow, gpu_prev_gray, gpu_curr_gray, None,
    )

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )
    frame = gpu_curr_gray.download()
    return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y
    
def Dual_tvl1_opt_flow(gpu_prev_gray, gpu_curr_gray):
    gpu_flow = cv2.cuda.OpticalFlowDual_TVL1.create(tau=0.25,
                                                    lambda_=0.15,
                                                    theta=0.3,
                                                    nscales=5,
                                                    warps=5,
                                                    epsilon=0.01,
                                                    iterations=300,
                                                    scaleStep=0.8,
                                                    gamma=0.0,
                                                    useInitialFlow=False
                                                    

    )
    # calculate optical flow
    gpu_flow = cv2.cuda.OpticalFlowDual_TVL1.calc(
        gpu_flow, gpu_prev_gray, gpu_curr_gray, None,
    )

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )
    frame = gpu_curr_gray.download()
    return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

def init_grid(image_shape, step = 2):
    h,w = image_shape
    y,x = np.mgrid[step//2:h:step, step//2:w:step]
    grid = np.stack((x, y), axis=-1).astype(np.float32)
    return grid.reshape(1, -1, 2)


def Sparse_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray, gpu_prev_p, gpu_next_p,gpu_status,gpu_err):
    sparse = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize=(21,21),
                                                     maxLevel=50,
                                                     iters=100,
                                                     useInitialFlow=False
    )
    # calculate optical flow
    gpu_next_p = cv2.cuda.GpuMat(gpu_prev_p.size(), cv2.CV_32FC2)
    gpu_status = cv2.cuda.GpuMat()
    gpu_err = cv2.cuda.GpuMat()
    sparse.calc(
       prevImg = gpu_prev_gray, nextImg = gpu_curr_gray, prevPts = gpu_prev_p, nextPts = gpu_next_p, status = gpu_status, err = gpu_err 
    )

    gpu_flow_x = cv2.cuda_GpuMat(gpu_next_p.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_next_p.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_next_p, [gpu_flow_x, gpu_flow_y])
    
    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )
    frame = gpu_curr_gray.download()
    bgr = frame.copy()
    prev_p = gpu_prev_p.download()
    next_p = gpu_next_p.download()
    status = gpu_status.download()

    print(status)

    p0 = prev_p[0]
    p1 = next_p[0]
    print("Number of good points:", np.sum(status == 1))

    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        bgr = cv2.line(bgr, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        bgr = cv2.circle(bgr, (int(a), int(b)), 5, (0, 0, 255), -1)

    return bgr, frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y, gpu_next_p, gpu_status

# only a feature detector
# def FAST_feat(gpu_curr_gray):
#     fast = cv2.cuda.FastFeatureDetector.create(threshold=10,
#                                                nonmaxSuppression=True,
#                                                type=cv2.FastFeatureDetector_TYPE_9_16,
#                                                max_npoints=50000
#     )
#     gpu_kp = cv2.cuda.GpuMat()
#     gpu_des = cv2.cuda.GpuMat()
#     fast.detectAndComputeAsync(image = gpu_curr_gray, mask = None,keypoints=gpu_kp,descriptors=gpu_des)

def ORB_feat(gpu_prev_gray,gpu_curr_gray):
    orb = cv2.cuda.ORB.create(nfeatures=200,
                              scaleFactor=1.2,
                              nlevels=8,
                              edgeThreshold=31,
                              firstLevel=0,
                              WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE,
                              patchSize=31,
                              fastThreshold=20,
                              blurForDescriptor=False
    )
    gpu_kp1 = cv2.cuda.GpuMat()
    gpu_des1 = cv2.cuda.GpuMat()
    gpu_kp2 = cv2.cuda.GpuMat()
    gpu_des2 = cv2.cuda.GpuMat()

    gpu_kp1, gpu_des1 = orb.detectAndComputeAsync(image=gpu_prev_gray,mask=None,useProvidedKeypoints=False)
    gpu_kp2, gpu_des2 = orb.detectAndComputeAsync(image=gpu_curr_gray,mask=None,useProvidedKeypoints=False)

    matcher = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
    img1 = gpu_prev_gray.download()
    img2 = gpu_curr_gray.download()
    kp1 = orb.convert(gpu_kp1)
    kp2 = orb.convert(gpu_kp2)

    
    matches = matcher.matchAsync(queryDescriptors=gpu_des1,trainDescriptors=gpu_des2,mask=None)
    good_matches = matcher.matchConvert(gpu_matches=matches)

    # gpu_matches = matcher.knnMatchAsync(queryDescriptors=gpu_des1,trainDescriptors=gpu_des2,k=2,mask=None)
    # matches = matcher.knnMatchConvert(gpu_matches=gpu_matches,compactResult=False)
    # good_matches = []

    # for m in matches:
    #     if len(m) >= 2:  # make sure we have two matches
    #         best, second_best = m[0], m[1]
    #         if best.distance < 0.75 * second_best.distance:
    #             good_matches.append(best)
    # #print(good_matches)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Good Matches", img_matches)

def SURF_feat(gpu_prev_gray,gpu_curr_gray):
    surf= cv2.cuda.SURF_CUDA.create(_hessianThreshold=400,
                                    _nOctaves=4,
                                    _nOctaveLayers=2,
                                    _extended=False,
                                    _keypointsRatio=0.01,
                                    _upright=False
    )
    gpu_kp1 = cv2.cuda.GpuMat()
    gpu_des1 = cv2.cuda.GpuMat()
    gpu_kp2 = cv2.cuda.GpuMat()
    gpu_des2 = cv2.cuda.GpuMat()

    gpu_kp1, gpu_des1 = surf.detectWithDescriptors(img=gpu_prev_gray,mask=None,useProvidedKeypoints=False)
    gpu_kp2, gpu_des2 = surf.detectWithDescriptors(img=gpu_curr_gray,mask=None,useProvidedKeypoints=False)

    matcher = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_L2)
    img1 = gpu_prev_gray.download()
    img2 = gpu_curr_gray.download()
    kp1 = surf.downloadKeypoints(gpu_kp1)
    kp2 = surf.downloadKeypoints(gpu_kp2)

    matches = matcher.matchAsync(queryDescriptors=gpu_des1,trainDescriptors=gpu_des2,mask=None)
    good_matches = matcher.matchConvert(gpu_matches=matches)

    # gpu_matches = matcher.knnMatchAsync(queryDescriptors=gpu_des1,trainDescriptors=gpu_des2,k=2,mask=None)
    # matches = matcher.knnMatchConvert(gpu_matches=gpu_matches,compactResult=False)
    # good_matches = []

    # for m in matches:
    #     if len(m) >= 2:  # make sure we have two matches
    #         best, second_best = m[0], m[1]
    #         if best.distance < 0.75 * second_best.distance:
    #             good_matches.append(best)
    #print(good_matches)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Good Matches", img_matches)

def vis_2D_flow_hsv(gpu_magnitude, gpu_angle,gpu_hsv_8u,gpu_hsv,gpu_h,gpu_s,gpu_v):
    gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)
    
    angle = gpu_angle.download()
    angle *= (1 / 360.0) * (180 / 255.0)

    gpu_mean_std = cv2.cuda.GpuMat((2,1),cv2.CV_64FC1)
    cv2.cuda.meanStdDev(mtx=gpu_angle,dst=gpu_mean_std)

    mean,std = np.array(gpu_mean_std.download()).flatten()
    print("ANGLE:",mean,std)

    gpu_h.upload(angle)
    cv2.cuda.merge([gpu_h, gpu_s, gpu_v], gpu_hsv)
    gpu_hsv.convertTo(rtype = cv2.CV_8U, dst = gpu_hsv_8u, alpha=255.0)
    
    gpu_bgr = cv2.cuda.cvtColor(gpu_hsv_8u, cv2.COLOR_HSV2BGR)
    bgr = gpu_bgr.download()

    return bgr

# def vis_2D_flow_vec(color_img, gpu_status, gpu_prev_p, gpu_next_p):
#     bgr = color_img.copy()
#     prev_p = gpu_prev_p.download()
#     next_p = gpu_next_p.download()
#     status = gpu_status.download()

#     p0 = prev_p[status==1]
#     p1 = next_p[status==1]
#     print("Number of good points:", np.sum(status ==1))

#     for i, (new, old) in enumerate(zip(p1, p0)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         bgr = cv2.line(bgr, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#         bgr = cv2.circle(bgr, (int(a), int(b)), 5, (0, 0, 255), -1)
    
#     return bgr

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

    ## RUN APP to VIS IN REALTIME
    if view_video: viewer3d = Viewer3D("Distance Deformation")

    load_time = time.time() - start_time
    print("Loading Completed in",load_time)
    
    ## OPTICAL FLOW FOR CONSECUTIVE COMPARISON
    prev_img = None
    prev_rgbd_img = None
    prev_tag_locations = None

    ## Initialise PREV frame
    if Real_Time:
        time_ms, count, depth_image, color_image, rgbd_image, t2cam_pcd = rsManager.get_frames()
        print("IMAGE NUMBER:",count, time_ms)
    elif imageStream.has_next():
        count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
        time_ms = time_arr[count-150]
        print("IMAGE NUMBER:",count, time_ms)

    prev_img = color_image
    prev_rgbd_img = rgbd_image
    prev_t2cam_pcd = t2cam_pcd

    gpu_curr = cv2.cuda.GpuMat()
    gpu_curr_gray = cv2.cuda.GpuMat()

    gpu_prev = cv2.cuda.GpuMat()
    gpu_prev.upload(prev_img)
    gpu_prev_gray = cv2.cuda.GpuMat()
    gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev, cv2.COLOR_BGR2GRAY)
    
    gpu_hsv = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC3)
    gpu_hsv_8u = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_8UC3)
    gpu_h = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    gpu_s = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    gpu_v = cv2.cuda_GpuMat(gpu_prev.size(), cv2.CV_32FC1)
    gpu_s.upload(np.ones_like(np.ones((gpu_prev.size()[1], gpu_prev.size()[0]), dtype=np.float32))) # set saturation to 1
    
    p0 = init_grid((480,848),step=10)
    #p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY), mask=None, **dict(maxCorners=40000, qualityLevel=0.3, minDistance=7, blockSize=7))
    gpu_prev_p = cv2.cuda.GpuMat()
    gpu_prev_p.upload(p0.reshape(1,-1,2))
    gpu_next_p = cv2.cuda.GpuMat(gpu_prev_p.size(), cv2.CV_32FC2)
    gpu_status = cv2.cuda.GpuMat()
    gpu_err = cv2.cuda.GpuMat()

    if p0 is None:
        print("No initial points found!")
    else:
        print(f"Found {np.size(p0)} initial points")
    
    ## LOOP THROUGH EACH FRAME 1 onwards
    try:
        while True:
            start_loop_time = time.time()
            start_cv2 = cv2.getTickCount()
            ## INPUT
            if Real_Time:
                time_ms, count, depth_image, color_image, rgbd_image, t2cam_pcd = rsManager.get_frames()
                print("IMAGE NUMBER:",count, time_ms)
            elif imageStream.has_next():
                count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
                time_ms = time_arr[count-150]
                print("IMAGE NUMBER:",count, time_ms)
            else:
                break

            gpu_curr.upload(color_image)
            gpu_curr_gray = cv2.cuda.cvtColor(gpu_curr, cv2.COLOR_BGR2GRAY)
            #frame_diff = cv2.cuda.absdiff(gpu_prev_gray, gpu_curr_gray)
            #print("Frame difference sum:", np.sum(frame_diff.download()))
            #print("prev_gray type:", gpu_prev_gray.type(), "size:", gpu_prev_gray.size())
            #print("curr_gray type:", gpu_curr_gray.type(), "size:", gpu_curr_gray.size())

            ## DENSE OPTICAL FLOW
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Farne_opt_flow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Brox_optflow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Dense_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Dual_tvl1_opt_flow(gpu_prev_gray, gpu_curr_gray)
            
            ## SPARSE OPTICAL FLOW
            #bgr,frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y, gpu_next_p, gpu_status = Sparse_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray,gpu_prev_p,gpu_next_p,gpu_status,gpu_err)
            
            ## FEATURE DETECTION ONLY
            #FAST_feat(gpu_curr_gray

            ## FEATURE DETECTION AND MATCHING
            #ORB_feat(gpu_prev_gray,gpu_curr_gray)
            SURF_feat(gpu_prev_gray,gpu_curr_gray)

            ## VIS
            #bgr = vis_2D_flow_hsv(gpu_magnitude, gpu_angle,gpu_hsv_8u,gpu_hsv,gpu_h,gpu_s,gpu_v)
            #bgr = vis_2D_flow_vec(color_image,gpu_status,gpu_prev_p,gpu_next_p)
            gpu_prev_p = gpu_next_p

            gpu_prev = gpu_curr
            gpu_prev_gray = gpu_curr_gray

            prev_img = color_image
            prev_rgbd_img = rgbd_image
            prev_t2cam_pcd = t2cam_pcd
            #prev_tag_locations = tag_locat1    ions

            end_loop_time = time.time()
            end_cv2 = cv2.getTickCount()
            time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
            print("FPS:", 1/time_sec)
            
            # visualization
            # cv2.imshow("original", frame)
            # cv2.imshow("result", bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            print("Time for one frame:",time_sec)

    finally:
        if Real_Time: rsManager.stop()
        #if view_video: viewer3d.stop()


if __name__ == "__main__":
    main()