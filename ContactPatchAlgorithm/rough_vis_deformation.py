import cv2
import open3d as o3d
import numpy as np
import robotpy_apriltag
import csv
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from capture_realsense import RealSenseManager
from replay_realsense import read_RGB_D_folder
from app_vis import Viewer3D
import time


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
    # print("Estimating model normals")
    # pcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
    # print("Completed")
    return pcd

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
        [k, idx, _] = pcd_tree.search_knn_vector_3d(m_points[tag],500)
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
    
def draw_registration_result(source, target, transformation):
    source_temp = source
    target_temp = target
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

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
    cropped_model = model_pcd.crop(t2cam_bounding_box)
    downsampled_t2cam = t2cam_pcd.uniform_down_sample(every_k_points=100)
    downsampled_cropped_model = cropped_model.uniform_down_sample(every_k_points=100)
    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #             t2cam_pcd, cropped_model, 0.02)
    # print(evaluation)
    print("Apply point-to-plane ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(t2cam_pcd, cropped_model, max_correspondence_distance = 0.00002, 
    #                                                       estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                                                       criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    reg_p2p = o3d.pipelines.registration.registration_icp(downsampled_t2cam, downsampled_cropped_model, max_correspondence_distance = 0.02, init = np.eye(4),
                                                          estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1.000000e-08,max_iteration=200))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    t2cam_pcd.transform(reg_p2p.transformation)
    #draw_registration_result(t2cam_pcd, cropped_model, reg_p2p.transformation)
    return cropped_model

def inner_deformed_to_inner_undeformed(t2cam_pcd,model_pcd):
    """
    Function to get the SIGNED corressponding distances between 
    the T2Cam Point Cloud and the Inner Tyre Model

    Args:
        t2cam_pcd (o3d.geometry.PointCloud): A Point Cloud of T2Cam image
        model_pcd (o3d.geometry.PointCloud): A Point Cloud of inner tyre model

    Returns:
        t2_d_pcd (o3d.geometry.PointCloud): Returns the T2Cam point Cloud with a colormap applied representing deformed distances
    """
    cropped_model_pcd = segment_pcd_using_bounding_box(t2cam_pcd,model_pcd)
    d_dist = np.asarray(t2cam_pcd.compute_point_cloud_distance(cropped_model_pcd))
    
    # num_bins = 20
    # plt.hist(d_dist, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
    # plt.title("Histogram of Deformed Distances")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.show()

    print(np.max(d_dist), np.min(d_dist))

    d_colors = plt.get_cmap('plasma')((d_dist - d_dist.min()) / (d_dist.max() - d_dist.min()))
    d_colors = d_colors[:, :3]

    t2_d_pcd = o3d.geometry.PointCloud()
    t2_d_pcd.points = t2cam_pcd.points
    t2_d_pcd.colors = o3d.utility.Vector3dVector(d_colors)
    return t2_d_pcd

def run_vis_app(t2cam_pcd,cropped_model_pcd):
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
    
def main():
    start_time = time.time()
    """Main function to excute the script"""
    Real_Time = False
    Run_on_Jetson = False
    debug_mode = False
    start_index = 100
    depth_profile = 3
    scale = 0.019390745853434508

    file_path_control_points = './4_row_model_control_points.csv'
    markers, m_points, numeric_markers = load_rc_control_points(file_path_control_points,scale)
    normalTag, RCcorTag = convert_rc_apriltag_hex_ids()
    correctTags = convert_rc_control_points(normalTag,numeric_markers)

    file_path_model = '4_row_model_HighPoly_Smoothed.ply'
    model_pcd = load_model_pcd(file_path_model,scale)

    tag_norm, model_correspondence = find_tag_point_ID_correspondence(model_pcd, m_points)
    tag_normals_line_set = draw_lines(m_points, tag_norm)

    if debug_mode: vis_window(model_pcd, tag_normals_line_set)
    
    if Real_Time:
        rsManager = RealSenseManager(depth_profile=depth_profile,color_profile=18,exposure=1000,gain=248,enable_spatial=False,enable_temporal=False)
    else:
        imageStream = read_RGB_D_folder(r'C:\Users\amalp\Desktop\MSS732\realsense',starting_index=start_index,depth_num=3,debug_mode=debug_mode)

    if Run_on_Jetson:
        from april_detect_jetson import DetectAprilTagsJetson
        detector = DetectAprilTagsJetson(depth_profile=depth_profile,debug_mode=debug_mode)
    else:
        from april_detect_windows import DetectAprilTagsWindows
        detector = DetectAprilTagsWindows(depth_profile=depth_profile,debug_mode=debug_mode)

    viewer3d = Viewer3D("Distance Deformation")
    load_time = time.time() - start_time
    print("Loading Completed in",load_time)
    try:
        while True:
            start_loop_time = time.time()
            if Real_Time:
                depth_image, color_image, rgbd_image, t2cam_pcd = rsManager.get_frames()
            elif imageStream.has_next():
                depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
            else:
                break
            
            # vis_window(t2cam_pcd)
            detector.input_frame(color_image)
            tag_IDs, tag_locations = detector.process_3D_locations(rgbd_image)
            
            time_at_detection = time.time()
            detect_time = time_at_detection - start_loop_time
            print("AprilTags Detected in", detect_time)
            #Super costly functions
            #t2cam_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            #t2cam_pcd_tree = o3d.geometry.KDTreeFlann(t2cam_pcd)

            t2cam_correspondence = find_t2cam_correspondence(t2cam_pcd,tag_locations,debug_mode)
            time_at_t2cam_corres = time.time()
            t2cam_corres_time = time_at_t2cam_corres - time_at_detection
            print("T2Cam Correspondence Point IDs calculated in:", t2cam_corres_time)

            correspondence_vector = make_correspondence_vector(t2cam_correspondence,model_correspondence,tag_IDs,correctTags,debug_mode)
            time_at_corres_vec = time.time()
            corres_vec_time = time_at_corres_vec - time_at_t2cam_corres
            print("Corres Vector made in", corres_vec_time)

            #vis_window(t2cam_pcd,model_pcd)
            register_t2cam_with_model(t2cam_pcd,model_pcd,correspondence_vector)
            time_at_registration = time.time()
            register_time = time_at_registration - time_at_corres_vec
            print("Registration in", register_time)
            #vis_window(t2cam_pcd,model_pcd)

            t2cam_dist_pcd = inner_deformed_to_inner_undeformed(t2cam_pcd,model_pcd)
            time_at_c2c = time.time()
            c2c_dist_time = time_at_c2c - time_at_registration
            print("C2C distance calculated in", c2c_dist_time)
            
            #o3d.visualization.draw_geometries([t2cam_dist_pcd])

            viewer3d.update_cloud(t2cam_dist_pcd)
            viewer3d.tick()
            time_at_app_view = time.time()
            app_view_time = time_at_app_view - time_at_c2c
            print("Time to update viewer", app_view_time)
            #time.sleep(0.1)

            

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
                    break

    finally:
        if Real_Time: rsManager.stop()


if __name__ == "__main__":
    main()