import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import time
from replay_realsense import read_RGB_D_folder

imageStream = read_RGB_D_folder('realsense2', starting_index=100, depth_num=3, debug_mode=3)
count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()

# Convert point cloud to numpy array
pcd_points = np.asarray(t2cam_pcd.points)

# Create a cKDTree for fast nearest neighbor search
pcd_tree = cKDTree(pcd_points)

# Generate multiple random query points
queries = pcd_points[np.random.choice(len(pcd_points), 100, replace=False)]

# Define KNN function using cKDTree
def knn_search(query):
    k = 1
    dist, idx = pcd_tree.query(query, k)
    return idx, dist

# Use joblib Parallel to parallelize the search
if __name__ == "__main__":
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(delayed(knn_search)(q) for q in queries)
    t1 = time.perf_counter()

    for i, (idx, dist) in enumerate(results):
        print(f"Query {i}: Neighbors {idx}, Distances {dist}")
    
    print("Time:", t1 - t0)


# import open3d as o3d
# import numpy as np
# from multiprocessing import Pool
# from replay_realsense import read_RGB_D_folder
# import time

# imageStream = read_RGB_D_folder('realsense2', starting_index=100, depth_num=3, debug_mode=3)
# count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()


# # Convert point cloud to numpy array
# pcd_points = np.asarray(t2cam_pcd.points)

# pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points)))

# # Generate multiple random query points
# queries = pcd_points[np.random.choice(len(pcd_points), 10, replace=False)]

# # Define KNN function (each process must create its own KDTree)
# def knn_search(query):
#     k = 1
#     [k_count, idx, dist] = pcd_tree.search_knn_vector_3d(query, k)
#     return list(idx), list(dist)

# # Use multiprocessing Pool to parallelize
# if __name__ == "__main__":
#     t0 = time.perf_counter()
#     with Pool(processes=6) as pool:  # Adjust based on CPU cores
#         results = pool.map(knn_search, queries)
#     t1 = time.perf_counter()

#     for i, (idx, dist) in enumerate(results):
#         print(f"Query {i}: Neighbors {idx}, Distances {dist}")
#     print("Time:", t1 - t0)