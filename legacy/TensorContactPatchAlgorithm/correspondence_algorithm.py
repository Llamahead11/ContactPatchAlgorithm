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
import os
import cupy as cp
import sys

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

# def Brox_optflow(gpu_prev_gray, gpu_curr_gray):
#     brox = cv2.cuda.BroxOpticalFlow.create(alpha=0.197,
#                                                gamma=5.0,
#                                                scale_factor=0.8,
#                                                inner_iterations=5,
#                                                outer_iterations=150,
#                                                solver_iterations=10
#     )
#     gpu_flow = cv2.cuda.GpuMat(gpu_prev_gray.size(),cv2.CV_32FC2)
#     gpu_prev_gray_f32 = cv2.cuda_GpuMat(gpu_prev_gray.size(), cv2.CV_32FC1)
#     gpu_curr_gray_f32 = cv2.cuda_GpuMat(gpu_curr_gray.size(), cv2.CV_32FC1)

#     gpu_prev_gray.convertTo(dst=gpu_prev_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)
#     gpu_curr_gray.convertTo(dst=gpu_curr_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)

#     gpu_flow_x = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32F)
#     gpu_flow_y = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32F)
#     brox.calc(I0=gpu_prev_gray_f32,I1=gpu_curr_gray_f32,flow=gpu_flow)


#     cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
#     gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
#         gpu_flow_x, gpu_flow_y, angleInDegrees=True,
#     )
#     frame = gpu_curr_gray.download()
#     return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

# def Dense_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray):
#     gpu_flow = cv2.cuda.DensePyrLKOpticalFlow.create(winSize=(15,15),
#                                                      maxLevel=5,
#                                                      iters=300,
#                                                      useInitialFlow=False
#     )
#     # calculate optical flow
#     gpu_flow = cv2.cuda.DensePyrLKOpticalFlow.calc(
#         gpu_flow, gpu_prev_gray, gpu_curr_gray, None,
#     )

#     gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
#     gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
#     cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
#     # convert from cartesian to polar coordinates to get magnitude and angle
#     gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
#         gpu_flow_x, gpu_flow_y, angleInDegrees=True,
#     )
#     frame = gpu_curr_gray.download()
#     return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

def Farne_opt_flow(gpu_prev_gray,gpu_curr_gray):
    gpu_flow = cv2.cuda.FarnebackOpticalFlow.create(numLevels=15,
                                                    pyrScale=0.5,
                                                    fastPyramids=True,
                                                    winSize=5,
                                                    numIters=6,
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
    
# def Dual_tvl1_opt_flow(gpu_prev_gray, gpu_curr_gray):
#     gpu_flow = cv2.cuda.OpticalFlowDual_TVL1.create(tau=0.25,
#                                                     lambda_=0.15,
#                                                     theta=0.3,
#                                                     nscales=5,
#                                                     warps=5,
#                                                     epsilon=0.01,
#                                                     iterations=300,
#                                                     scaleStep=0.8,
#                                                     gamma=0.0,
#                                                     useInitialFlow=False
                                                    

#     )
#     # calculate optical flow
#     gpu_flow = cv2.cuda.OpticalFlowDual_TVL1.calc(
#         gpu_flow, gpu_prev_gray, gpu_curr_gray, None,
#     )

#     gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
#     gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
#     cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
    
#     # convert from cartesian to polar coordinates to get magnitude and angle
#     gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
#         gpu_flow_x, gpu_flow_y, angleInDegrees=True,
#     )
#     frame = gpu_curr_gray.download()
#     return frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y

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

# def to_3D_disp(count,vertex_map_gpu_prev, vertex_map_prev, vertex_map_gpu_curr, vertex_map_curr, normal_map_prev, normal_map_curr, gpu_flow_x, gpu_flow_y, traj_motion_2D_x, traj_motion_2D_y,traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5):
    
#     t0 = time.time()
#     #find the flow at the subpixels of the previous 2D flow trajectory vector
#     interp_flow_x = cv2.cuda.remap(gpu_flow_x, traj_motion_2D_x, traj_motion_2D_y, interpolation=cv2.INTER_LINEAR)
#     interp_flow_y = cv2.cuda.remap(gpu_flow_y, traj_motion_2D_x, traj_motion_2D_y, interpolation=cv2.INTER_LINEAR)

#     # calc the next 2D trajectory 
#     map_x_gpu = cv2.cuda.add(traj_motion_2D_x,interp_flow_x)
#     map_y_gpu = cv2.cuda.add(traj_motion_2D_y,interp_flow_y)

#     cv2_vertex_map_gpu_curr = cv2.cuda.GpuMat()
#     cv2_vertex_map_gpu_curr.upload(vertex_map_curr)

#     cv2_vertex_map_gpu_prev = cv2.cuda.GpuMat()
#     cv2_vertex_map_gpu_prev.upload(vertex_map_prev)
#     #correspondence_vertex_map_prev = vertex_map_prev[curr_image_pixels[:,:,0],curr_image_pixels[:,:,1]]

#     # calc x,y,z 3D flow from 2D trajectory
#     subpixel_interp_vertex_map_gpu = cv2.cuda.remap(cv2_vertex_map_gpu_curr, map_x_gpu, map_y_gpu, interpolation=cv2.INTER_LINEAR)
    
#     #assign next 2D traj
#     traj_motion_2D_x = map_x_gpu  
#     traj_motion_2D_y = map_y_gpu 

#     buffer_idx = count % 5
#     if buffer_idx == 0:
#         traj_motion_3D_1 = subpixel_interp_vertex_map_gpu 
#         print(0)
#     elif buffer_idx == 1:
#         traj_motion_3D_2 = subpixel_interp_vertex_map_gpu 
#         print(1)
#     elif buffer_idx == 2:
#         traj_motion_3D_3 = subpixel_interp_vertex_map_gpu 
#         print(2)
#     elif buffer_idx == 3:
#         traj_motion_3D_4 = subpixel_interp_vertex_map_gpu 
#         print(3)
#     else:
#         traj_motion_3D_5 = subpixel_interp_vertex_map_gpu  
#         print(4)  

#     #point_disp_wrt_cam_gpu = cv2.cuda.subtract(subpixel_interp_vertex_map_gpu,cv2_vertex_map_gpu_prev) 
#     #subpixel_interp_vertex_map = subpixel_interp_vertex_map_gpu.download()
#     #point_disp_wrt_cam = point_disp_wrt_cam_gpu.download()

 

#     #create a (H,W) and add (gpu_flow_x and gpu_flow_y) array and mask it on the prev_frame vertex_map 
#     #  pixel (1,1) prev moves (1,1) so pixel (2,2) curr
#     #  so (1,1) -> (2,2) = [x1,y1,z1] -> [x2,y2,z2]
#     #  [u1,v1,w1] = [x2,y2,z2] - [x1,y1,z1]  

#     # possibly filter out subpixel flow < 1 maybe or do bilinear interpolation
#     # need to compensate for camera movement through pose estimation 

#     # therefore, vertex_map_prev(H+gpu_flow_y,W+gpu_flow_x) = vertex_map_curr(H,W) 

#     # need normal_map_prev and normal_map_curr for coordinate system
#     # use normal_maps (surface normal) -> z axis
#     # calculate x and y axis for each point

#     # transform [u1,v1,w1] vector in correct coordinates system

#     #flow_vertex_map = vertex_map[image_pixe+(dx,dy)]
    
#     # normal_map_prev_gpu = cp.asarray(normal_map_prev)
#     # normal_map_prev_gpu /= cp.linalg.norm(normal_map_prev_gpu, axis=2, keepdims=True)

#     # # Calculate norms of the normal map
#     # norms_gpu = cp.linalg.norm(normal_map_prev_gpu, axis=2)

#     # # Create a reference vector based on the norm check
#     # reference_vector_gpu = cp.where(norms_gpu[..., None] < 0.9, cp.array([0, 1, 0], dtype=cp.float32)[None, None, :], 
#     #                                 cp.array([1, 0, 0], dtype=cp.float32)[None, None, :])

#     # # Cross product to get the x axis of the local coordinate system
#     # x_axis_gpu = cp.cross(normal_map_prev_gpu, reference_vector_gpu)
#     # x_axis_gpu /= cp.linalg.norm(x_axis_gpu, axis=2, keepdims=True)

#     # # Cross product again to get the y axis of the local coordinate system
#     # y_axis_gpu = cp.cross(normal_map_prev_gpu, x_axis_gpu)

#     # # Rotation matrix is constructed by stacking the axes
#     # rotation_matrix_gpu = cp.stack((x_axis_gpu, y_axis_gpu, normal_map_prev_gpu), axis=-1)

#     #cp_point_disp_wrt_cam = cp.asarray(point_disp_wrt_cam)
#     # Use `einsum` to apply the rotation to point displacements on GPU
#     #local_point_displacements_gpu = cp.einsum('...ji,...j->...i', rotation_matrix_gpu, cp_point_disp_wrt_cam)

#     # Download the result if necessary
#     #local_point_displacements = local_point_displacements_gpu.get()

#     t1 = time.time()
#     #x_axis = x_axis_gpu.get()
#     #y_axis = y_axis_gpu.get()

#     print("EINSUM:",t1-t0)

#     # x_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+x_axis.reshape(-1,3)*0.0001)
#     # y_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+y_axis.reshape(-1,3)*0.0001)
#     # z_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+normal_map_prev.reshape(-1,3)*0.0001)
    
#     #disp_lines = draw_lines(vertex_map_prev.reshape(-1,3), subpixel_warped_vertex_map.reshape(-1,3))

#     t2 = time.time()
#     print("DRAW_LINES",t2-t1)
#     return traj_motion_2D_x, traj_motion_2D_y, traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5

def to_3D_vel(dt,local_point_displacements):
    velocity = local_point_displacements/dt
    return velocity
    
def draw_lines(start_points, end_points):
    line_start = start_points
    line_end = end_points
    dist = np.linalg.norm(line_end-line_start,axis=1)
    # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # line_points = np.vstack((start_points, end_points))
    valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    valid_dist = dist < 0.2

    mask = valid_start & valid_end & valid_dist
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 

    n = line_valid_start.shape[0]
    #print(np.max(line_start[:,2]),np.max(line_end[:,2]))
    lines = np.empty((n, 2), dtype=np.int32)
    lines[:, 0] = np.arange(n, dtype=np.int32)
    lines[:, 1] = np.arange(n, 2 * n, dtype=np.int32)
    #lines = np.column_stack((np.arange(n), np.arange(n, 2*n)))
    #line_points = np.concatenate((start_points, end_points), axis=0)

    line_points = np.empty((2 * n, 3), dtype=line_start.dtype)
    line_points[:n] = line_valid_start
    line_points[n:] = line_valid_end

    line_set = o3d.t.geometry.LineSet()
    line_set.point.positions = o3d.core.Tensor(line_points, dtype = o3d.core.float32)
    line_set.line.indices = o3d.core.Tensor(lines,dtype = o3d.core.int32)
    #line_set.line.colors = o3d.core.Tensor([[1, 0, 0]] * len(lines))
    #line_set.paint_uniform_color(o3d.core.Tensor([1,0,0]))
    return line_set

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
        # with open(os.path.join(config["RGB_D_folder"],"time.npy"), 'rb') as f:
        #     time_arr = np.load(f)

    ## RUN APP to VIS IN REALTIME
    if view_video: viewer3d = Viewer3D("Scene Flow")

    optmethod = config["opt_method"]
    if optmethod == 'dense':
        dense = DenseOptFlow(depth_profile,debug_mode,config['dense_method'])
    elif optmethod == 'sparse':
        sparse = SparseOptFlow(depth_profile,debug_mode,config['sparse_step'])


    load_time = time.time() - start_time
    print("Loading Completed in",load_time)


    ## Initialise PREV frame
    #==========================================================================================================================================================================
    if Real_Time:
        time_ms, count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = rsManager.get_frames()
        print("IMAGE NUMBER:",count, time_ms)
    elif imageStream.has_next():
        count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu = imageStream.get_next_frame()
        #time_ms = time_arr[count-150]
        print("IMAGE NUMBER:",count)#, time_ms)
    
    #INITIALISE DENSE/SPARSE Opt Flow with gpu_frame


    #INITIALISE FRAME and GRAY
    #prev_img = (color_image*255).astype(np.uint8)
    # prev_t2cam_pcd = t2cam_pcd
    prev_t2cam_pcd_cuda = t2cam_pcd_cuda

    gpu_curr = cv2.cuda.GpuMat()
    gpu_curr_gray = cv2.cuda.GpuMat()
    
    # gpu_prev = cv2.cuda.GpuMat()
    # gpu_prev.upload(prev_img)
    dl_color = color_image.as_tensor().to_dlpack()
    cp_color = cp.from_dlpack(dl_color)
    color_ptr = cp_color.data.ptr

    gpu_prev = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_ptr)
    gpu_prev_gray = cv2.cuda.GpuMat()
    gpu_prev_gray_uint8 = gpu_prev.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
    gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev_gray_uint8, cv2.COLOR_BGR2GRAY)

    dense.init_disp(vertex_map_gpu)
    
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

    # pixel to 3D
    # vertex_map_prev = vertex_map_np
    # normal_map_prev = normal_map_np
    vertex_map_gpu_prev = vertex_map_gpu #o3d tensor gpu
    normal_map_gpu_prev = normal_map_gpu #o3d tensor gpu

    #DENSE OPTICAL FLOW (GRID IMAGE TRACKING)
    # grid_x_row = np.arange(848//20, dtype=np.float32)[None, :]  # (1, W)
    # grid_y_col = np.arange(480//20, dtype=np.float32)[:, None]  # (H, 1)
    # grid_x_row = np.arange(848, dtype=np.float32)[None, :]  # (1, W)
    # grid_y_col = np.arange(480, dtype=np.float32)[:, None]  # (H, 1)
    
    # grid_x_gpu = cv2.cuda.GpuMat()
    # grid_y_gpu = cv2.cuda_GpuMat()
    # grid_x_gpu.upload(np.tile(grid_x_row, (480, 1)))
    # grid_y_gpu.upload(np.tile(grid_y_col, (1, 848)))

    # INIT TRAJECTORY 2D with GRID IMAGE 
    # traj_motion_2D_x = cv2.cuda.GpuMat()
    # traj_motion_2D_y = cv2.cuda.GpuMat()
    # # traj_motion_2D_x.upload(np.tile(grid_x_row, (480//20, 1)))
    # # traj_motion_2D_y.upload(np.tile(grid_y_col, (1, 848//20)))
    # traj_motion_2D_x.upload(np.tile(grid_x_row, (480, 1)))
    # traj_motion_2D_y.upload(np.tile(grid_y_col, (1, 848)))

    #INITIALISE 5 frame buffer of vertex maps
    # traj_motion_3D_1 = cv2.cuda.GpuMat(rows = 480//20, cols = 848//20,type= cv2.CV_32FC3)
    # traj_motion_3D_2 = cv2.cuda.GpuMat(rows = 480//20, cols = 848//20,type= cv2.CV_32FC3)
    # traj_motion_3D_3 = cv2.cuda.GpuMat(rows = 480//20, cols = 848//20,type= cv2.CV_32FC3)
    # traj_motion_3D_4 = cv2.cuda.GpuMat(rows = 480//20, cols = 848//20,type= cv2.CV_32FC3)
    # traj_motion_3D_5 = cv2.cuda.GpuMat(rows = 480//20, cols = 848//20,type= cv2.CV_32FC3)
    # traj_motion_3D_1 = cv2.cuda.GpuMat(rows = 480, cols = 848,type= cv2.CV_32FC3)
    # traj_motion_3D_2 = cv2.cuda.GpuMat(rows = 480, cols = 848,type= cv2.CV_32FC3)
    # traj_motion_3D_3 = cv2.cuda.GpuMat(rows = 480, cols = 848,type= cv2.CV_32FC3)
    # traj_motion_3D_4 = cv2.cuda.GpuMat(rows = 480, cols = 848,type= cv2.CV_32FC3)
    # traj_motion_3D_5 = cv2.cuda.GpuMat(rows = 480, cols = 848,type= cv2.CV_32FC3)

    # traj_motion_3D_1.upload(vertex_map_np)
    # traj_motion_3D_2.upload(vertex_map_np)
    # traj_motion_3D_3.upload(vertex_map_np)
    # traj_motion_3D_4.upload(vertex_map_np)
    # traj_motion_3D_5.upload(vertex_map_np)

    prev_mask = np.ones((480,848), dtype=bool)
    
    ## LOOP THROUGH EACH FRAME 1 onwards
    #==========================================================================================================================================================================
    try:
        while True:
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

            #gpu_curr.upload((color_image*255).astype(np.uint8))
            dl_color_loop = color_image.as_tensor().to_dlpack()
            cp_color_loop = cp.from_dlpack(dl_color_loop)
            color_loop_ptr = cp_color_loop.data.ptr

            gpu_curr = cv2.cuda.createGpuMatFromCudaMemory(rows = 480, cols = 848, type=cv2.CV_32FC3, cudaMemoryAddress=color_loop_ptr)
            gpu_curr_gray_uint8 = gpu_curr.convertTo(rtype=cv2.CV_8UC3, alpha=255.0)
            gpu_curr_gray = cv2.cuda.cvtColor(gpu_curr_gray_uint8, cv2.COLOR_BGR2GRAY)

            #frame_diff = cv2.cuda.absdiff(gpu_prev_gray, gpu_curr_gray)
            #print("Frame difference sum:", np.sum(frame_diff.download()))
            #print("prev_gray type:", gpu_prev_gray.type(), "size:", gpu_prev_gray.size())
            #print("curr_gray type:", gpu_curr_gray.type(), "size:", gpu_curr_gray.size())

            ## DENSE OPTICAL FLOW

            frame_gpu, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = dense.detect2D(gpu_prev_gray, gpu_curr_gray)
            
            bgr = dense.vis_hsv_2D()
            
            map_x_gpu, map_y_gpu, traj_motion_2D_x, traj_motion_2D_y,traj_motion_3D_1, traj_motion_3D_2,traj_motion_3D_3,traj_motion_3D_4,traj_motion_3D_5 = dense.detect3D(count, vertex_map_gpu,normal_map_gpu_prev, normal_map_gpu)
            
            dense.track_3D_vel(time_ms)
            
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Farne_opt_flow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Brox_optflow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Dense_pyrlk_opt_flow(gpu_prev_gray, gpu_curr_gray)
            #frame, gpu_magnitude, gpu_angle, gpu_flow_x, gpu_flow_y = Dual_tvl1_opt_flow(gpu_prev_gray, gpu_curr_gray)
            
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
            
            # g1_p = traj_motion_3D_1.download()
            # g1 = o3d.t.geometry.PointCloud()
            # g1.point.positions = o3d.core.Tensor(g1_p.reshape(-1,3),dtype=o3d.core.float32)
            # g1d = g1.uniform_down_sample(every_k_points = 10)
            # g2_p = traj_motion_3D_2.download()
            # g2 = o3d.t.geometry.PointCloud()
            # g2.point.positions = o3d.core.Tensor(g2_p.reshape(-1,3),dtype=o3d.core.float32)
            # g2d = g2.uniform_down_sample(every_k_points = 10)
            # g3_p = traj_motion_3D_3.download()
            # g3 = o3d.t.geometry.PointCloud()
            # g3.point.positions = o3d.core.Tensor(g3_p.reshape(-1,3),dtype=o3d.core.float32)
            # g3d = g3.uniform_down_sample(every_k_points = 10)
            # g4_p = traj_motion_3D_4.download()
            # g4 = o3d.t.geometry.PointCloud()
            # g4.point.positions = o3d.core.Tensor(g4_p.reshape(-1,3),dtype=o3d.core.float32)
            # g4d = g4.uniform_down_sample(every_k_points = 10)
            # g5_p = traj_motion_3D_5.download()
            # g5 = o3d.t.geometry.PointCloud()
            # g5.point.positions = o3d.core.Tensor(g5_p.reshape(-1,3),dtype=o3d.core.float32)
            # g5.point.colors = o3d.core.Tensor(color_image,dtype=o3d.core.float32)
            # g5d = g5.uniform_down_sample(every_k_points = 10)

            # d1 = draw_lines(g1.point.positions.numpy(), g2.point.positions.numpy())
            # d1.paint_uniform_color(o3d.core.Tensor([0,1,0]))
            # d2 = draw_lines(g2.point.positions.numpy(), g3.point.positions.numpy())
            # d3 = draw_lines(g3.point.positions.numpy(), g4.point.positions.numpy())
            # d4 = draw_lines(g4.point.positions.numpy(), g5.point.positions.numpy())
            # d4.paint_uniform_color(o3d.core.Tensor([1,0,0]))
            x_points = map_x_gpu.download()
            y_points = map_y_gpu.download()

            mask = (x_points <= 848 - 1) & (y_points <= 480-1) & (x_points >= 0+1) & (y_points >= 0+1)

            #now create a border to indicate lost tracks that will clump into regions

            # how to reseed tracks correctly
            #now get the indices of those points lost and reinitialise with the INTERPOLATED VALUE of those neighbouring points of the index lost?
            #                 
            x_points = x_points[mask]
            y_points = y_points[mask]

            curr_mask = mask & prev_mask

            xy_pixels = np.argwhere(curr_mask)
             

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

            # g1,g2,g3,g4,g5,d1,d2,d3,d4 = dense.vis_3D()
            g1,g2,d1,d2,frame,vel_arrow, curr_outer_pcd = dense.vis_3D()

            #g1,g2,g3,g4,g5,d1,d2,d3,d4 = sparse.vis_3D()

            t0 = time.time()
            if view_video:
                #viewer3d.update_cloud(d1=d1.cpu(),d2=d2.cpu(),d3=d3.cpu(),d4=d4.cpu(),g5=g5.cpu())
                viewer3d.update_cloud(g1=g1.cpu(),g2=g2.cpu(),d1=d1.cpu(),d2=d2.cpu(), frame = frame.cpu(), vel_arrow = vel_arrow.cpu())   
                viewer3d.tick()
                #o3d.visualization.draw([g1.cpu(),g2.cpu(),d1.cpu(),d2.cpu()])
            t1 = time.time()
            print("APP:", t1-t0)
            
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


            frame = frame_gpu.download()
            #valid_mask_cpu = valid_mask.download()
            valid_image = np.zeros((480, 848), dtype=np.uint8)
            valid_image[curr_mask] = 255
            prev_mask = curr_mask
            # visualization for dense
            cv2.imshow("original", frame)
            cv2.imshow("result", bgr)
            cv2.imshow("valid points", valid_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                if view_video: viewer3d.stop()
                break

            print("Time for one frame:",time_sec)

            # for _ in range(6):
            #     sys.stdout.write('\033[F')      # Move cursor up one line
            #     sys.stdout.write('\033[K')      # Clear the entire line

    finally:
        if Real_Time: rsManager.stop()
        #if view_video: viewer3d.stop()
        sys.exit()


if __name__ == "__main__":
    main()