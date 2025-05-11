import cv2
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
import vidVisualiser as vV
import cupy as cp

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
            "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
            "data": (gpu_mat.cudaPtr(), False),
        }

class DenseOptFlow:
    def __init__(self, depth_profile, debug_mode, dense_method):
        self.dense_method = dense_method
        if dense_method == 'farne':
            self.dense_flow = cv2.cuda.FarnebackOpticalFlow.create(numLevels=5,
                                                        pyrScale=0.5,
                                                        fastPyramids=True,
                                                        winSize=15,
                                                        numIters=10,
                                                        polyN=7,
                                                        polySigma=1.5,
                                                        flags=0
            )
        elif dense_method == 'brox':
            self.dense_flow = cv2.cuda.BroxOpticalFlow.create(alpha=0.197,
                                               gamma=5.0,
                                               scale_factor=0.8,
                                               inner_iterations=5,
                                               outer_iterations=150,
                                               solver_iterations=10
            )
        elif dense_method == 'pyrLK':
            self.dense_flow = cv2.cuda.DensePyrLKOpticalFlow.create(winSize=(31,31),
                                                     maxLevel=10,
                                                     iters=300,
                                                     useInitialFlow=False
            )
        elif dense_method == 'dual':
            self.dense_flow = cv2.cuda.OpticalFlowDual_TVL1.create(tau=0.25,
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

        self.debug_mode = debug_mode
        self.depth_num = depth_profile
    
        if self.depth_num == 3:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
        elif self.depth_num == 0:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")

        self.fx = self.intrinsic.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic.intrinsic_matrix[1, 2]
        self.width = self.intrinsic.width
        self.height = self.intrinsic.height

        self.gpu_flow = cv2.cuda.GpuMat(rows = self.height, cols = self.width, type = cv2.CV_32FC2)
        self.gpu_flow_x = cv2.cuda_GpuMat(self.gpu_flow.size(), cv2.CV_32FC1)
        self.gpu_flow_y = cv2.cuda_GpuMat(self.gpu_flow.size(), cv2.CV_32FC1)

        self.gpu_prev_gray_f32 = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type = cv2.CV_32FC1)
        self.gpu_curr_gray_f32 = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type = cv2.CV_32FC1)

        self.gpu_hsv = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type=cv2.CV_32FC3)
        self.gpu_hsv_8u = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type=cv2.CV_8UC3)
        self.gpu_h = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type=cv2.CV_32FC1)
        self.gpu_s = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type=cv2.CV_32FC1)
        self.gpu_v = cv2.cuda_GpuMat(rows = self.height, cols = self.width, type=cv2.CV_32FC1)
        self.gpu_s.upload(np.ones_like(np.ones((self.height, self.width), dtype=np.float32))) # set saturation to 1

        self.traj_motion_2D_x = cv2.cuda.GpuMat()
        self.traj_motion_2D_y = cv2.cuda.GpuMat()
        grid_x_row = np.arange(self.width, dtype=np.float32)[None, :]  # (1, W)
        grid_y_col = np.arange(self.height, dtype=np.float32)[:, None]  # (H, 1)
        self.traj_motion_2D_x.upload(np.tile(grid_x_row, (self.height, 1)))
        self.traj_motion_2D_y.upload(np.tile(grid_y_col, (1, self.width)))

        self.cv2_vertex_map_gpu_prev = cv2.cuda.GpuMat()
        self.cv2_vertex_map_gpu_prev.upload(np.ones_like(np.ones((self.height, self.width,3), dtype=np.float32)))
        self.cv2_vertex_map_gpu_curr = cv2.cuda.GpuMat()
        self.cv2_normal_map_gpu_curr = cv2.cuda.GpuMat()

        self.traj_motion_3D_1 = cv2.cuda.GpuMat(rows = self.height, cols = self.width,type= cv2.CV_32FC3)
        self.traj_motion_3D_2 = cv2.cuda.GpuMat(rows = self.height, cols = self.width,type= cv2.CV_32FC3)
        self.traj_motion_3D_3 = cv2.cuda.GpuMat(rows = self.height, cols = self.width,type= cv2.CV_32FC3)
        self.traj_motion_3D_4 = cv2.cuda.GpuMat(rows = self.height, cols = self.width,type= cv2.CV_32FC3)
        self.traj_motion_3D_5 = cv2.cuda.GpuMat(rows = self.height, cols = self.width,type= cv2.CV_32FC3)

        self.cp_traj_motion_3D = cp.empty((self.height,self.width,3,5), dtype=np.float32)
        self.curr_traj_motion_3D = cp.empty((self.height,self.width,3), dtype=np.float32)
        self.prev_traj_motion_3D = cp.empty((self.height,self.width,3), dtype=np.float32)

        self.curr_normal_3D = cp.empty((self.height,self.width,3), dtype=np.float32)

        self.g1 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        self.g2 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        self.g3 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        self.g4 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        self.g5 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))

        self.d1 = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
        self.d2 = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
        self.d3 = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
        self.d4 = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
        self.vel_arrow = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))

        self.mesh_curr = o3d.t.geometry.TriangleMesh(o3d.core.Device("CUDA:0"))
        self.mesh_prev = o3d.t.geometry.TriangleMesh(o3d.core.Device("CUDA:0"))

        self.local_point_displacements_gpu = cp.empty((self.height,self.width,3), dtype=np.float32)
        self.local_point_velocities_gpu = cp.empty((self.height,self.width,3), dtype=np.float32)
        self.rotation_matrix_gpu = cp.empty((self.height,self.width,3,3), dtype=np.float32)
        self.point_disp_wrt_cam_gpu = cp.empty((self.height,self.width,3), dtype=np.float32)
        self.point_vel_wrt_cam_gpu = cp.empty((self.height,self.width,3), dtype=np.float32)
        self.mean_rotation_matrix_gpu = cp.empty((3,3), dtype = cp.float32)
        self.mean_local_point_velocities_gpu = cp.empty((1,3), dtype = cp.float32)

        self.norms_gpu = cp.empty((self.height,self.width,3), dtype = cp.float32)
        self.x_axis_gpu = cp.empty((self.height,self.width,3), dtype = cp.float32)
        self.y_axis_gpu = cp.empty((self.height,self.width,3), dtype = cp.float32)
        self.mean_norms_gpu = cp.empty((1,3), dtype = cp.float32)
        self.mean_x_axis_gpu = cp.empty((1,3), dtype = cp.float32)
        self.mean_y_axis_gpu = cp.empty((1,3), dtype = cp.float32)

        #self.triangles = cp.empty((self.height,self.width,3,2), dtype=cp.float32)
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

        self.triangles = cp.array(self.triangles, dtype=cp.int32)

                    

    def detect2D(self, gpu_prev_gray, gpu_curr_gray):
        if self.dense_method == 'brox':
            gpu_prev_gray.convertTo(dst=self.gpu_prev_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)
            gpu_curr_gray.convertTo(dst=self.gpu_curr_gray_f32, rtype=cv2.CV_32F, alpha=1.0 / 255.0)
            #print(self.gpu_curr_gray_f32.type(), self.gpu_prev_gray_f32.type())
            self.dense_flow.calc(
                I0 = self.gpu_prev_gray_f32, I1 = self.gpu_curr_gray_f32, flow = self.gpu_flow
            )
        else:
            self.dense_flow.calc(
                gpu_prev_gray, gpu_curr_gray, self.gpu_flow, None
            )

        cv2.cuda.split(self.gpu_flow, [self.gpu_flow_x, self.gpu_flow_y])

        self.gpu_magnitude, self.gpu_angle = cv2.cuda.cartToPolar(
            self.gpu_flow_x, self.gpu_flow_y, angleInDegrees=True,
        )

        #frame = gpu_curr_gray.download()
        return gpu_curr_gray, self.gpu_magnitude, self.gpu_angle,self.gpu_flow_x, self.gpu_flow_y
    
        # vV.plotAndSavePlt(local_point_displacements,prev_3D_points,vmin,vmax)

        # # Show the figure in a non-blocking way
        # plt.savefig("./Farne_video/{}.png".format(self.count))
        # #plt.show(block=False)
        # self.count +=1

    def vis_hsv_2D(self):
        self.gpu_v = cv2.cuda.normalize(self.gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)
    
        angle = self.gpu_angle.download()
        angle *= (1 / 360.0) * (180 / 255.0)

        gpu_mean_std = cv2.cuda.GpuMat((2,1),cv2.CV_64FC1)
        cv2.cuda.meanStdDev(mtx=self.gpu_angle,dst=gpu_mean_std)

        self.gpu_h.upload(angle)
        cv2.cuda.merge([self.gpu_h, self.gpu_s, self.gpu_v], self.gpu_hsv)
        self.gpu_hsv.convertTo(rtype = cv2.CV_8U, dst = self.gpu_hsv_8u, alpha=255.0)
        
        gpu_bgr = cv2.cuda.cvtColor(self.gpu_hsv_8u, cv2.COLOR_HSV2BGR)
        bgr = gpu_bgr.download()

        return bgr
    
    def init_disp(self, vertex_map_gpu):
        dl_vertex_map = vertex_map_gpu.as_tensor().clone().to_dlpack()
        self.prev_traj_motion_3D = cp.from_dlpack(dl_vertex_map)
                
    def detect3D(self,count, vertex_map_gpu_curr, normal_map_gpu_prev, normal_map_gpu_curr):
        t0 = time.time()
        #find the flow at the subpixels of the previous 2D flow trajectory vector
        interp_flow_x = cv2.cuda.remap(self.gpu_flow_x, self.traj_motion_2D_x, self.traj_motion_2D_y, interpolation=cv2.INTER_LINEAR)
        interp_flow_y = cv2.cuda.remap(self.gpu_flow_y, self.traj_motion_2D_x, self.traj_motion_2D_y, interpolation=cv2.INTER_LINEAR)

        # calc the next 2D trajectory 
        map_x_gpu = cv2.cuda.add(self.traj_motion_2D_x,interp_flow_x)
        map_y_gpu = cv2.cuda.add(self.traj_motion_2D_y,interp_flow_y)
        
        dl_vertex_map_curr = vertex_map_gpu_curr.as_tensor().to_dlpack()
        dl_normal_map_curr = normal_map_gpu_curr.as_tensor().to_dlpack()
        dl_normal_map_prev = normal_map_gpu_prev.as_tensor().to_dlpack()

        cp_vertex_map_curr = cp.from_dlpack(dl_vertex_map_curr)
        cp_normal_map_curr = cp.from_dlpack(dl_normal_map_curr)

        vertex_map_curr_ptr = cp_vertex_map_curr.data.ptr
        normal_map_curr_ptr = cp_normal_map_curr.data.ptr

        #self.cv2_vertex_map_gpu_curr.upload(vertex_map_curr)
        self.cv2_vertex_map_gpu_curr = cv2.cuda.createGpuMatFromCudaMemory(rows=self.height, cols=self.width, type=cv2.CV_32FC3, cudaMemoryAddress=vertex_map_curr_ptr)
        #correspondence_vertex_map_prev = vertex_map_prev[curr_image_pixels[:,:,0],curr_image_pixels[:,:,1]]
        self.cv2_normal_map_gpu_curr = cv2.cuda.createGpuMatFromCudaMemory(rows=self.height, cols=self.width, type=cv2.CV_32FC3, cudaMemoryAddress=normal_map_curr_ptr)

        # calc x,y,z 3D flow from 2D trajectory
        subpixel_interp_vertex_map_gpu = cv2.cuda.remap(self.cv2_vertex_map_gpu_curr, map_x_gpu, map_y_gpu, interpolation=cv2.INTER_LINEAR)
        subpixel_interp_normal_map_gpu = cv2.cuda.remap(self.cv2_normal_map_gpu_curr, map_x_gpu, map_y_gpu, interpolation=cv2.INTER_LINEAR)

        #assign next 2D traj
        self.traj_motion_2D_x = map_x_gpu.clone()  
        self.traj_motion_2D_y = map_y_gpu.clone() 

        # buffer_idx = count % 5
        # if buffer_idx == 0:
        #     self.traj_motion_3D_1 = subpixel_interp_vertex_map_gpu 
        #     print(0)
        # elif buffer_idx == 1:
        #     self.traj_motion_3D_2 = subpixel_interp_vertex_map_gpu 
        #     print(1)
        # elif buffer_idx == 2:
        #     self.traj_motion_3D_3 = subpixel_interp_vertex_map_gpu 
        #     print(2)
        # elif buffer_idx == 3:
        #     self.traj_motion_3D_4 = subpixel_interp_vertex_map_gpu 
        #     print(3)
        # else:
        #     self.traj_motion_3D_5 = subpixel_interp_vertex_map_gpu  
        #     print(4)  

        # cp_sub = cp.asarray(CudaArrayInterface(subpixel_interp_vertex_map_gpu))
        # self.cp_traj_motion_3D[...,count % 5] = cp_sub
        noncontigous_cp_vertex= cp.asarray(CudaArrayInterface(subpixel_interp_vertex_map_gpu))
        noncontigous_cp_normal= cp.asarray(CudaArrayInterface(subpixel_interp_normal_map_gpu))
        self.curr_traj_motion_3D  = cp.ascontiguousarray(noncontigous_cp_vertex)
        self.curr_normal_3D  = cp.ascontiguousarray(noncontigous_cp_normal)
        # print(self.curr_traj_motion_3D.__cuda_array_interface__)
        # print(self.prev_traj_motion_3D.__cuda_array_interface__)
        #point_disp_wrt_cam_gpu = cv2.cuda.subtract(subpixel_interp_vertex_map_gpu,self.cv2_vertex_map_gpu_prev) 
        self.point_disp_wrt_cam_gpu = self.curr_traj_motion_3D - self.prev_traj_motion_3D

        #subpixel_interp_vertex_map = subpixel_interp_vertex_map_gpu.download()
        #point_disp_wrt_cam = point_disp_wrt_cam_gpu.download()

    

        #create a (H,W) and add (gpu_flow_x and gpu_flow_y) array and mask it on the prev_frame vertex_map 
        #  pixel (1,1) prev moves (1,1) so pixel (2,2) curr
        #  so (1,1) -> (2,2) = [x1,y1,z1] -> [x2,y2,z2]
        #  [u1,v1,w1] = [x2,y2,z2] - [x1,y1,z1]  

        # possibly filter out subpixel flow < 1 maybe or do bilinear interpolation
        # need to compensate for camera movement through pose estimation 

        # therefore, vertex_map_prev(H+gpu_flow_y,W+gpu_flow_x) = vertex_map_curr(H,W) 

        # need normal_map_prev and normal_map_curr for coordinate system
        # use normal_maps (surface normal) -> z axis
        # calculate x and y axis for each point

        # transform [u1,v1,w1] vector in correct coordinates system

        #flow_vertex_map = vertex_map[image_pixe+(dx,dy)]
        
        normal_map_prev_gpu = cp.from_dlpack(dl_normal_map_prev)
        #normal_map_prev_gpu = cp.nan_to_num(normal_map_prev_gpu, nan=0.0)
        normal_map_prev_gpu /= cp.linalg.norm(normal_map_prev_gpu, axis=2, keepdims=True)
        #print(normal_map_prev_gpu.shape)

        # Calculate norms of the normal map
        self.norms_gpu = cp.linalg.norm(normal_map_prev_gpu, axis=2)
        self.mean_norms_gpu = cp.nanmean(normal_map_prev_gpu,axis=(0,1))
        #print(self.mean_norms_gpu.shape)
        self.mean_norms_gpu /= cp.linalg.norm(self.mean_norms_gpu, axis=0)
        #print(self.mean_norms_gpu.shape)

        # Create a reference vector based on the norm check
        reference_vector_gpu = cp.where(self.norms_gpu[..., None] < 0.9, cp.array([0, 1, 0], dtype=cp.float32)[None, None, :], 
                                        cp.array([1, 0, 0], dtype=cp.float32)[None, None, :])

        # Cross product to get the x axis of the local coordinate system
        self.x_axis_gpu = cp.cross(normal_map_prev_gpu, reference_vector_gpu)
        #self.x_axis_gpu = cp.nan_to_num(self.x_axis_gpu, nan=0.0)
        self.x_axis_gpu /= cp.linalg.norm(self.x_axis_gpu, axis=2, keepdims=True)
        self.mean_x_axis_gpu = cp.nanmean(self.x_axis_gpu,axis=(0,1))
        self.mean_x_axis_gpu /= cp.linalg.norm(self.mean_x_axis_gpu, axis=0)
        #print(self.x_axis_gpu.shape, self.mean_x_axis_gpu.shape)

        # Cross product again to get the y axis of the local coordinate system
        self.y_axis_gpu = cp.cross(normal_map_prev_gpu, self.x_axis_gpu)
        #self.y_axis_gpu = cp.nan_to_num(self.y_axis_gpu, nan=0.0)
        self.y_axis_gpu /= cp.linalg.norm(self.y_axis_gpu, axis=2, keepdims=True)
        self.mean_y_axis_gpu = cp.nanmean(self.y_axis_gpu, axis=(0,1))
        self.mean_y_axis_gpu /= cp.linalg.norm(self.mean_y_axis_gpu, axis=0)
        #print(self.y_axis_gpu.shape, self.mean_y_axis_gpu.shape)

        # Rotation matrix is constructed by stacking the axes
        self.rotation_matrix_gpu = cp.stack((self.x_axis_gpu, self.y_axis_gpu, normal_map_prev_gpu), axis=-1)
        self.mean_rotation_matrix_gpu = cp.stack((self.mean_x_axis_gpu, self.mean_y_axis_gpu, self.mean_norms_gpu), axis=-1)
        #print(self.rotation_matrix_gpu.shape, self.mean_rotation_matrix_gpu.shape)
        #print(self.mean_rotation_matrix_gpu)
        #cp_point_disp_wrt_cam = cp.asarray(point_disp_wrt_cam)
        #Use `einsum` to apply the rotation to point displacements on GPU
        self.local_point_displacements_gpu = cp.einsum('...ij,...j->...i', self.rotation_matrix_gpu, self.point_disp_wrt_cam_gpu)
        
        
        
        #Download the result if necessary
        #local_point_displacements = local_point_displacements_gpu.get()

        t1 = time.time()
        #x_axis = x_axis_gpu.get()
        #y_axis = y_axis_gpu.get()

        print("EINSUM:",t1-t0)

        # x_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+x_axis.reshape(-1,3)*0.0001)
        # y_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+y_axis.reshape(-1,3)*0.0001)
        # z_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+normal_map_prev.reshape(-1,3)*0.0001)
        
        #disp_lines = draw_lines(vertex_map_prev.reshape(-1,3), subpixel_warped_vertex_map.reshape(-1,3))

        t2 = time.time()
        #print("DRAW_LINES",t2-t1)
        
        
        return self.traj_motion_2D_x, self.traj_motion_2D_y, self.traj_motion_3D_1, self.traj_motion_3D_2,self.traj_motion_3D_3,self.traj_motion_3D_4,self.traj_motion_3D_5


    def get_3D_disp_wrt_cam(self):
        pass

    def get_3D_disp_local(self):
        pass

    def track_3D_vel(self,dt_ms):
        self.local_point_velocities_gpu = self.local_point_displacements_gpu/(dt_ms/1000)
        self.point_vel_wrt_cam_gpu = self.point_disp_wrt_cam_gpu/(dt_ms/1000)
        
        # mag_vel = cp.linalg.norm(self.local_point_velocities_gpu, axis = 2)
        mag_vel = cp.linalg.norm(self.point_vel_wrt_cam_gpu, axis = 2)
        #print(mag_vel.shape)
        mask = (mag_vel < 0.1) & (mag_vel > 0.001)

        #self.mean_local_point_velocities_gpu = cp.nanmean(self.local_point_velocities_gpu[mask], axis=0)
        self.mean_local_point_velocities_gpu = cp.nanmean(self.point_vel_wrt_cam_gpu[mask], axis=0)
        #print(self.mean_local_point_velocities_gpu.shape)
        #print(self.mean_local_point_velocities_gpu)
        # print(self.local_point_velocities_gpu)
        # print(dt_ms/1000)
        # print(self.local_point_displacements_gpu)
        print("ANGLE:",cp.degrees(cp.arctan2(self.mean_local_point_velocities_gpu[1],self.mean_local_point_velocities_gpu[0])))

    def make_tracked_mesh(self): 
        '''
        Naive approach to meshed tracking, does not account for the local convergence of flow
        also call once for init_disp with vertex map to make prev vertex map
        '''
        # vertices = vertex_map_np.reshape(-1, 3)
        # normals = normal_map_np.reshape(-1, 3)
        vertices = self.curr_traj_motion_3D.reshape(-1,3)
        normals = self.curr_normal_3D.reshape(-1,3)
        #print(vertices.shape, normals.shape)
        
        dl_vertices = vertices.toDlpack()
        dl_normals = normals.toDlpack()

        #need to interp also normal map subpixelly 
        indices = self.triangles.copy()
        mask_idx = cp.where(vertices[:,2] > 0.05)[0]
        mask = cp.isin(indices, mask_idx) 
        mask = cp.all(mask, axis=1) 
        indices = indices[mask]
        #print(indices.shape)

        dl_indices = indices.toDlpack()
        colors = cp.random.rand(*indices.shape)
        dl_colors = colors.toDlpack()
        #valid_triangles = filter_triangles(self.triangles, mask)

        self.mesh_curr.vertex.positions = o3d.core.Tensor.from_dlpack(dl_vertices)
        self.mesh_curr.triangle.indices = o3d.core.Tensor.from_dlpack(dl_indices)
        self.mesh_curr.vertex.normals = o3d.core.Tensor.from_dlpack(dl_normals)
        self.mesh_curr.triangle.colors = o3d.core.Tensor.from_dlpack(dl_colors)

#     @numba.jit(nopython=True)
# def filter_triangles(triangles, mask):
#     out = np.empty((triangles.shape[0], 3), dtype=np.int32)
#     count = 0
#     for i in range(triangles.shape[0]):
#         v0, v1, v2 = triangles[i]
#         if mask[v0] and mask[v1] and mask[v2]:
#             out[count, 0] = v0
#             out[count, 1] = v1
#             out[count, 2] = v2
#             count += 1
#     return out[:count]


    def vis_3D(self):
        prev = self.prev_traj_motion_3D
        curr = self.curr_traj_motion_3D
        vel = self.point_vel_wrt_cam_gpu
        local_vel = self.local_point_velocities_gpu

        g1_p = prev.reshape(-1,3).toDlpack()
        self.g1.point.positions = o3d.core.Tensor.from_dlpack(g1_p)
        g1d = self.g1.uniform_down_sample(every_k_points = 30)
        g2_p = curr.reshape(-1,3).toDlpack()
        self.g2.point.positions = o3d.core.Tensor.from_dlpack(g2_p)
        g2d = self.g2.uniform_down_sample(every_k_points = 30)

        draw_lines(prev, curr, self.d1)
        self.d1.paint_uniform_color(o3d.core.Tensor([0,1,0]))
        draw_lines_vel(prev, prev+vel, self.d2)
        self.d2.paint_uniform_color(o3d.core.Tensor([1,0,0]))

        #self.make_tracked_mesh()

        #x_local_axes = 
        # draw_lines(prev,prev + self.x_axis_gpu*0.001, self.d3)
        #y_local_axes = 
        # draw_lines(prev,prev + self.y_axis_gpu*0.001, self.d4)
        #z_local_axes = 
        # draw_lines(prev,prev + self.normal_map_prev*0.0001)
        dl_mean_rot = self.mean_rotation_matrix_gpu.toDlpack()
        frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.05, device = o3d.core.Device("CUDA:0"))
        frame.rotate(R = o3d.core.Tensor.from_dlpack(dl_mean_rot), center = o3d.core.Tensor([0,0,0], device=o3d.core.Device("CUDA:0")))
        #frame.rotate(R = o3d.core.Tensor([[1,0,0],[0,1,0],[0,0,1]], device=o3d.core.Device("CUDA:0")), center = o3d.core.Tensor([0,0,0], device=o3d.core.Device("CUDA:0")))
        # vel_arrow = o3d.t.geometry.TriangleMesh.create_arrow(cylinder_radius = 0.05, cone_radius = 0.075, cylinder_height = 0.25, cone_height = 0.2,resolution = 20, cylinder_split = 4,cone_split = 1,device = o3d.core.Device("CUDA:0"))
        # vel_arrow.rotate(R = o3d.core.Tensor.from_dlpack(dl_mean_rot), center = o3d.core.Tensor([0,0,0], device=o3d.core.Device("CUDA:0")))
        #print(self.mean_local_point_velocities_gpu)
        draw_simple_lines(cp.array([0,0,0]),self.mean_local_point_velocities_gpu,self.vel_arrow)

        self.prev_traj_motion_3D = self.curr_traj_motion_3D.copy()
        self.cv2_vertex_map_gpu_prev = self.cv2_vertex_map_gpu_curr.clone()
        self.mesh_prev = self.mesh_curr

        return g1d,self.g2,self.d1,self.d2,frame, self.vel_arrow
        #return g1d,g2d,self.d1,self.d2,frame, self.vel_arrow
        #return g1d,g2d,self.d1,self.d2, self.d3,self.d4, frame, self.vel_arrow

        # g1_p = self.cp_traj_motion_3D[...,0].reshape(-1,3).toDlpack()
        # self.g1.point.positions = o3d.core.Tensor.from_dlpack(g1_p)
        # g1d = self.g1.uniform_down_sample(every_k_points = 30)
        # g2_p = self.cp_traj_motion_3D[...,1].reshape(-1,3).toDlpack()
        # self.g2.point.positions = o3d.core.Tensor.from_dlpack(g2_p)
        # g2d = self.g2.uniform_down_sample(every_k_points = 30)
        # g3_p = self.cp_traj_motion_3D[...,2].reshape(-1,3).toDlpack()
        # self.g3.point.positions = o3d.core.Tensor.from_dlpack(g3_p)
        # g3d = self.g3.uniform_down_sample(every_k_points = 30)
        # g4_p = self.cp_traj_motion_3D[...,3].reshape(-1,3).toDlpack()
        # self.g4.point.positions = o3d.core.Tensor.from_dlpack(g4_p)
        # g4d = self.g4.uniform_down_sample(every_k_points = 30)
        # g5_p = self.cp_traj_motion_3D[...,4].reshape(-1,3).toDlpack()
        # self.g5.point.positions = o3d.core.Tensor.from_dlpack(g5_p)
        # g5d = self.g5.uniform_down_sample(every_k_points = 30)

        # draw_lines(self.cp_traj_motion_3D[...,0], self.cp_traj_motion_3D[...,1], self.d1)
        # self.d1.paint_uniform_color(o3d.core.Tensor([0,1,0]))
        # draw_lines(self.cp_traj_motion_3D[...,1], self.cp_traj_motion_3D[...,2], self.d2)
        # draw_lines(self.cp_traj_motion_3D[...,2], self.cp_traj_motion_3D[...,3], self.d3)
        # draw_lines(self.cp_traj_motion_3D[...,3], self.cp_traj_motion_3D[...,4], self.d4)
        # self.d4.paint_uniform_color(o3d.core.Tensor([1,0,0]))

        # return g1d,g2d,g3d,g4d,g5d,self.d1,self.d2,self.d3,self.d4
        #return self.g1,self.g2,self.g3,self.g4,self.g5,self.d1,self.d2,self.d3,self.d4
def draw_simple_lines(start_points, end_points, line_set):
    line_start = start_points
    line_end = end_points
    #print(line_end)
    n = 1
    #print(n)
    #print(np.max(line_start[:,2]),np.max(line_end[:,2]))
    lines = cp.empty((n, 2), dtype=np.int32)
    lines[:, 0] = cp.arange(n, dtype=np.int32)
    lines[:, 1] = cp.arange(n, 2 * n, dtype=np.int32)
    #lines = np.column_stack((np.arange(n), np.arange(n, 2*n)))
    #line_points = np.concatenate((start_points, end_points), axis=0)

    line_points = cp.empty((2 * n, 3), dtype=np.float32)
    line_points[:n] = line_start
    line_points[n:] = line_end

    #print(lines)
    #print(line_points)
    lps = line_points.toDlpack()
    ls = lines.toDlpack() 

    line_set.point.positions = o3d.core.Tensor.from_dlpack(lps)
    line_set.line.indices = o3d.core.Tensor.from_dlpack(ls)

def draw_lines(start_points, end_points, line_set):
    line_start = start_points.reshape(-1,3)[::30]
    line_end = end_points.reshape(-1,3)[::30]
    dist = cp.linalg.norm(line_end-line_start,axis=1)
    
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

def draw_lines_vel(start_points, end_points, line_set):
    line_start = start_points.reshape(-1,3)[::30]
    line_end = end_points.reshape(-1,3)[::30]
    vel = cp.linalg.norm(line_end-line_start,axis=1)
    
    # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # line_points = np.vstack((start_points, end_points))
    valid_start = ((line_start[:,2] <= 0.7) & (line_start[:,2] >= 0.07)) 
    #valid_end = ((line_end[:,2] <= 7) & (line_end[:,2] >= 0))
    # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.0))
    valid_vel = (vel < 0.1) & (vel > 0.001)

    # mask = valid_start & valid_end & valid_dist
    nan_mask = ~ (cp.isnan(line_end).any(axis=1) | cp.isnan(line_start).any(axis=1))

    mask = nan_mask  & valid_start & valid_vel
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 

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
