import open3d as o3d
import numpy as np
import copy
#______________________________________________________________________________________________________________________________________
# Loas and Visualize 3D model and current RGB-D image

mesh_inner = o3d.io.read_triangle_mesh("C:\\Users\\amalp\\Desktop\\MSS732\\betterinside\\scene\\cleaned_integrated.ply")
mesh_inner.compute_vertex_normals()

pcd_inner = o3d.io.read_point_cloud("C:\\Users\\amalp\\Desktop\\MSS732\\betterinside\\scene\\cleaned_integrated.ply")
print(pcd_inner.get_center())

pcd_small = o3d.io.read_point_cloud("C:\\Users\\amalp\\Desktop\\MSS732\\Intel_Stereo_Cam\\IntelStereoExplore\\x64\\Debug\\cloud_1.pcd")
pcd_small.scale(10, center=pcd_small.get_center())

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd_small,pcd_inner,mesh_frame])

#_______________________________________________________________________________________________________________________________________
# Pick point and find rough radius

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
#vis.add_geometry(pcd_small)
vis.add_geometry(pcd_inner+pcd_small)
vis.run()
vis.destroy_window()

rough_radius = np.linalg.norm(np.asarray(pcd_inner.points)[vis.get_picked_points()[0]])
print(rough_radius)

# pcd_small.translate(np.asarray(pcd_inner.points)[vis.get_picked_points()[0]]+pcd_small.get_center())
# small_center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=0.6, origin=pcd_small.get_center())

# o3d.visualization.draw_geometries([pcd_small,pcd_inner,mesh_frame,small_center_frame])

#_______________________________________________________________________________________

def compute_pca(point_cloud):
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)
    
    # Compute the mean of the points
    centroid = np.mean(points, axis=0)
    
    # Center the points around the mean
    centered_points = points - centroid
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    return centroid, eigenvalues, eigenvectors

#__________________________________________________________________________________
# Align 3D model eignvectors to x,y,z axis
cent_inner, eigval_inner, eigvec_inner = compute_pca(pcd_inner)
dir = eigvec_inner[:, np.argsort(-eigval_inner)]
rot_mat = dir.T

pcd_inner.rotate(rot_mat)
pcd_inner.translate(-cent_inner)
print("3D model PCA alignment to XYZ")

#__________________________________________________________________________________
# Find KNN for each point along the center
pcd_tree = o3d.geometry.KDTreeFlann(pcd_inner)
p_c = np.asarray(pcd_inner.points) 
query = p_c[np.abs(p_c[:,2]) < 5e-3]
print(query)

[k, idx, _] = pcd_tree.search_knn_vector_3d(query[0], 200)
query_pcd = o3d.geometry.PointCloud()
#query_pcd.points = o3d.utility.Vector3dVector(p_c[idx,:])
query_pcd.points = o3d.utility.Vector3dVector(query)
query_pcd.paint_uniform_color([1,0,0])
print(len(idx),len(query))
#__________________________________________________________________________________

#_____________________________________________________________________________

# Compute PCA
centroid_inner, eigenvalues_inner, eigenvectors_inner = compute_pca(pcd_inner)
centroid_small, eigenvalues_small, eigenvectors_small = compute_pca(pcd_small)

# Print the results
print("Centroid:", centroid_inner,centroid_small)
print("Eigenvalues:", eigenvalues_inner,eigenvalues_small)
print("Eigenvectors (principal axes):", eigenvectors_inner, eigenvectors_small)

# Visualize the principal axes
# Scale the eigenvectors by their eigenvalues for better visualization
axes = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=centroid_inner)

axis_inner = o3d.t.geometry.LineSet()
axis_inner.point.positions = o3d.core.Tensor([centroid_inner,
                                              centroid_inner + eigenvectors_inner[:, 0] * eigenvalues_inner[0], 
                                              centroid_inner + eigenvectors_inner[:, 1] * eigenvalues_inner[1], 
                                              centroid_inner + eigenvectors_inner[:, 2] * eigenvalues_inner[2]], o3d.core.float32)
axis_inner.line.indices = o3d.core.Tensor([[0, 1],[0, 2],[0, 3]], o3d.core.int32)
axis_inner.line.colors = o3d.core.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], o3d.core.float32)

axis_small = o3d.t.geometry.LineSet()
axis_small.point.positions = o3d.core.Tensor([centroid_small, 
                                              centroid_small + eigenvectors_small[:, 0] * eigenvalues_small[0], 
                                              centroid_small + eigenvectors_small[:, 1] * eigenvalues_small[1], 
                                              centroid_small + eigenvectors_small[:, 2] * eigenvalues_small[2]], o3d.core.float32)
axis_small.line.indices = o3d.core.Tensor([[0, 1],[0, 2],[0, 3]], o3d.core.int32)
axis_small.line.colors = o3d.core.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], o3d.core.float32)


# Visualize point cloud and principal axes
o3d.visualization.draw_geometries([mesh_frame,query_pcd, pcd_small, axis_inner.to_legacy(), axis_small.to_legacy()])


#__________________________________________________________________________________
#Visualise

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
#

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source,target):
    print(":: Load two point clouds and disturb initial pose.")

    #demo_icp_pcds = o3d.data.DemoICPPointClouds()
    #source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    #target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

voxel_size = 0.05  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size,pcd_small,pcd_inner)

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.499))
    return result

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)

    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 30, 14]
    # current_transformation = np.identity(4)
    # print("Colored point cloud registration ...\n")
    # for scale in range(3):
    #     iter = max_iter[scale]
    #     radius = voxel_radius[scale]
    #     print([iter, radius, scale])

    #     print("1. Downsample with a voxel size %.2f" % radius)
    #     source_down = source.voxel_down_sample(radius)
    #     target_down = target.voxel_down_sample(radius)

    #     print("2. Estimate normal")
    #     source_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #     target_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    #     print("3. Applying colored point cloud registration")
    #     result_icp = o3d.pipelines.registration.registration_colored_icp(
    #         source_down, target_down, radius, current_transformation,
    #         o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #         o3d.pipelines.registration.ICPConvergenceCriteria(
    #             relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
    #     current_transformation = result_icp.transformation
    #     print(result_icp, "\n")
        
    # draw_registration_result(source, target, result_icp.transformation)
    # print(current_transformation)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)