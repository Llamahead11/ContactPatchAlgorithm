import numpy as np
import cv2
import open3d as o3d
import os
import time
from numba import cuda
import numba

class CudaArrayInterface:
    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        type_map = {
            cv2.CV_8U: "|u1",
            cv2.CV_8UC1: "|u1",
            cv2.CV_8UC2: "|u1",
            cv2.CV_8UC3: "|u1",
            cv2.CV_8UC4: "|u1",
            cv2.CV_8S: "|i1",
            cv2.CV_16U: "<u2", cv2.CV_16S: "<i2",
            cv2.CV_32S: "<i4",
            cv2.CV_32F: "<f4", cv2.CV_64F: "<f8",
            cv2.CV_32FC1: "<f4",
            cv2.CV_32FC2: "<f4",
            cv2.CV_32FC3: "<f4"
        }
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (h, w, gpu_mat.channels()),
            "typestr": type_map[gpu_mat.type()],
            "descr": [("", type_map[gpu_mat.type()])],
            "stream": 1,
            "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
            "data": (gpu_mat.cudaPtr(), False),
        }


class read_RGB_D_folder:
    def __init__(self, folder, starting_index=0,depth_num=3,debug_mode=True):
        self.folder = folder
        self.depth_folder = os.path.join(os.path.dirname(__file__),folder, "depth")
        self.color_folder = os.path.join(os.path.dirname(__file__),folder, "color")
        self.depth_num = depth_num
        self.index = starting_index
        self.debug_mode = debug_mode
        self.depth_files = [os.path.join(self.depth_folder, f) for f in os.listdir(self.depth_folder) if f.endswith(".png")]
        self.color_files = [os.path.join(self.color_folder, f) for f in os.listdir(self.color_folder) if f.endswith(".jpg")]
        self.cu = o3d.core.Device("CUDA:0")
        self.cpu = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.float32
        self.arr_inc = np.arange(407040)
        self.old_to_new = -np.ones(407040, dtype=int)

        if self.debug_mode: print(self.index, self.depth_files[self.index],self.color_files[self.index])
        
        if self.depth_num == 3:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json").intrinsic_matrix).cuda()
        elif self.depth_num == 0:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json").intrinsic_matrix).cuda()


        # Step 2: build triangle indices
        self.triangles = []
        H = 480
        W = 848
        for i in range(H - 1):
            for j in range(W - 1):
                idx = i * W + j
                v0 = idx
                v1 = idx + 1
                v2 = idx + W
                v3 = idx + W + 1
                self.triangles.append([v0, v1, v2])
                self.triangles.append([v1, v3, v2])

        self.triangles = np.array(self.triangles)

    def has_next(self):
        return self.index < len(self.color_files)
    
    def get_next_frame(self):
        current_depth_np = np.asarray(o3d.io.read_image(self.depth_files[self.index]), np.float32) 
        current_color_np = np.asarray(o3d.io.read_image(self.color_files[self.index]), np.float32)
        current_depth_np /= 10000
        current_color_np /= 255

        # current_depth_np = current_depth.as_tensor().numpy()
        # current_color_np = current_color.as_tensor().numpy()

        current_depth = o3d.t.geometry.Image(o3d.core.Tensor(current_depth_np))
        current_color = o3d.t.geometry.Image(o3d.core.Tensor(current_color_np))
       
        #current_depth = o3d.t.io.read_image(self.depth_files[self.index])
        #current_color = o3d.t.io.read_image(self.color_files[self.index])

        current_depth_cuda = current_depth.cuda()
        current_color_cuda = current_color.cuda()

        self.index += 1
        depth_scale = 1/0.0001
       
        #filtered_depth = current_depth_cuda.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
        vertex_map = current_depth_cuda.create_vertex_map(self.intrinsic)
        normal_map = vertex_map.create_normal_map()

        pcd_cuda = o3d.t.geometry.PointCloud(self.cu)
        pcd_cuda.point.positions = vertex_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.normals = normal_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.colors = current_color_cuda.as_tensor().reshape((-1, 3))
        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # vertex_map_np = vertex_map.as_tensor().cpu().numpy()
        # normal_map_np = normal_map.as_tensor().cpu().numpy()

        #pcd = pcd_cuda.cpu()

        # vertices = vertex_map_np.reshape(-1, 3)
        # normals = normal_map_np.reshape(-1, 3)

        # mask = (vertices[:,2] != 0)
        # v = np.count_nonzero(mask)

        # valid_triangles = filter_triangles(self.triangles, mask)


        # mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CPU:0"))
        # mesh.vertex.positions = o3d.core.Tensor(vertices, o3d.core.float32, o3d.core.Device("CPU:0"))
        # mesh.triangle.indices = o3d.core.Tensor(valid_triangles, o3d.core.int32, o3d.core.Device("CPU:0"))
        # mesh.vertex.normals = o3d.core.Tensor(normals,o3d.core.float32, o3d.core.Device("CPU:0"))

        # t5 = time.time()
        #print("1: ", t1-t0)
        # print("2: ", t2-t1)
        # print("3: ", t3-t2)
        #print("4: ", t4-t3)
        # print("5: ", t5-t4)

        #mesh.triangle.normals = o3d.core.Tensor(...)
        # Step 5: clean up and visualize
        # mesh.remove_unreferenced_vertices()
        # #mesh.remove_degenerate_triangles()
        #mesh.remove_non_manifold_edges()

        #o3d.visualization.draw_geometries([mesh.to_legacy()])
        #mesh_cpu = mesh.cpu()
       
        return self.index, current_depth_np, current_color_cuda , pcd_cuda, vertex_map, normal_map

@numba.jit(nopython=True)
def filter_triangles(triangles, mask):
    out = np.empty((triangles.shape[0], 3), dtype=np.int32)
    count = 0
    for i in range(triangles.shape[0]):
        v0, v1, v2 = triangles[i]
        if mask[v0] and mask[v1] and mask[v2]:
            out[count, 0] = v0
            out[count, 1] = v1
            out[count, 2] = v2
            count += 1
    return out[:count]

@cuda.jit
def check_triangles_validity(triangles, mask, validity_flags):
    i = cuda.grid(1)
    if i < triangles.shape[0]:
        v0, v1, v2 = triangles[i][0], triangles[i][1], triangles[i][2]
        validity_flags[i] = mask[v0] and mask[v1] and mask[v2]

def filter_triangles_cuda(triangles, mask):
    N = triangles.shape[0]

    # Prepare GPU memory
    d_triangles = cuda.to_device(triangles)
    d_mask = cuda.to_device(mask)
    d_validity_flags = cuda.device_array(N, dtype=np.bool_)

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    check_triangles_validity[blocks_per_grid, threads_per_block](d_triangles, d_mask, d_validity_flags)

    # Copy flags back
    validity_flags = d_validity_flags.copy_to_host()

    # Filter on CPU
    filtered = triangles[validity_flags]

    return filtered