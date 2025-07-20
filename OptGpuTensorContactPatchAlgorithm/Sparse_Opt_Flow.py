import cv2 
import time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import vidVisualiser as vV
import cupy as cp

class SparseOptFlow:
    def __init__(self, depth_profile, debug_mode, grid_step):
        self.depth_num = depth_profile
        self.debug_mode = debug_mode
        self.grid_step = grid_step

        self.sparse = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize=(21,21),
                                                        maxLevel=50,
                                                        iters=150,
                                                        useInitialFlow=False
        )

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

        row_mat = (self.height-self.grid_step//2)//self.grid_step
        col_mat = (self.width-self.grid_step//2)//self.grid_step

        self.traj_motion_2D_x = cv2.cuda.GpuMat(rows = row_mat, cols = col_mat,type= cv2.CV_32FC1)
        self.traj_motion_2D_y = cv2.cuda.GpuMat(rows = row_mat, cols = col_mat,type= cv2.CV_32FC1)

        # grid_x_row = np.arange(self.width//self.grid_step, dtype=np.float32)[None, :]  # (1, W)
        # grid_y_col = np.arange(self.height//self.grid_step, dtype=np.float32)[:, None]  # (H, 1)
        # self.traj_motion_2D_x.upload(np.tile(grid_x_row, (self.height//self.grid_step, 1)))
        # self.traj_motion_2D_y.upload(np.tile(grid_y_col, (1, self.width//self.grid_step)))

        self.cv2_vertex_map_gpu_prev = cv2.cuda.GpuMat()
        self.cv2_vertex_map_gpu_prev.upload(np.ones_like(np.ones((self.height, self.width,3), dtype=np.float32)))
        self.cv2_vertex_map_gpu_curr = cv2.cuda.GpuMat()

        p0 = init_grid((self.height, self.width),step=self.grid_step)
        #p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY), mask=None, **dict(maxCorners=40000, qualityLevel=0.3, minDistance=7, blockSize=7))
        self.gpu_prev_p = cv2.cuda.GpuMat()
        self.gpu_prev_p.upload(p0)
        
        self.gpu_next_p = cv2.cuda.GpuMat(self.gpu_prev_p.size(), cv2.CV_32FC2)

        self.gpu_status = cv2.cuda.GpuMat()
        self.gpu_err = cv2.cuda.GpuMat()

        self.gpu_flow_x = cv2.cuda.GpuMat(self.gpu_next_p.size(), cv2.CV_32FC1)
        self.gpu_flow_y = cv2.cuda.GpuMat(self.gpu_next_p.size(), cv2.CV_32FC1)

        self.cp_traj_motion_3D = cp.empty((self.height,self.width,3,5), dtype=np.float32)

        self.traj_motion_3D_1 = cv2.cuda.GpuMat(self.gpu_prev_p.size(),type= cv2.CV_32FC3)
        self.traj_motion_3D_2 = cv2.cuda.GpuMat(self.gpu_prev_p.size(),type= cv2.CV_32FC3)
        self.traj_motion_3D_3 = cv2.cuda.GpuMat(self.gpu_prev_p.size(),type= cv2.CV_32FC3)
        self.traj_motion_3D_4 = cv2.cuda.GpuMat(self.gpu_prev_p.size(),type= cv2.CV_32FC3)
        self.traj_motion_3D_5 = cv2.cuda.GpuMat(self.gpu_prev_p.size(),type= cv2.CV_32FC3)
    
        self.g1 = o3d.t.geometry.PointCloud()
        self.g2 = o3d.t.geometry.PointCloud()
        self.g3 = o3d.t.geometry.PointCloud()
        self.g4 = o3d.t.geometry.PointCloud()
        self.g5 = o3d.t.geometry.PointCloud()

        self.d1 = o3d.t.geometry.LineSet()
        self.d2 = o3d.t.geometry.LineSet()
        self.d3 = o3d.t.geometry.LineSet()
        self.d4 = o3d.t.geometry.LineSet()

    def init_prev_frame(self):
        pass

    def detect2D(self, gpu_prev_gray, gpu_curr_gray):
        self.sparse.calc(
        prevImg = gpu_prev_gray, nextImg = gpu_curr_gray, prevPts = self.gpu_prev_p, nextPts = self.gpu_next_p, status = self.gpu_status, err = self.gpu_err 
        )
        print(self.gpu_prev_p.size(), self.gpu_next_p.size(), self.gpu_flow_x.size(), self.gpu_flow_y.size())
        cv2.cuda.split(self.gpu_next_p, [self.gpu_flow_x, self.gpu_flow_y])
        
        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
            self.gpu_flow_x, self.gpu_flow_y, angleInDegrees=True,
        )

        return gpu_magnitude, gpu_angle, self.gpu_flow_x, self.gpu_flow_y, self.gpu_next_p, self.gpu_status
    
    def vis_grid_2D(self, gpu_curr_gray):
        prev_p = self.gpu_prev_p.download()
        next_p = self.gpu_next_p.download()
        frame = gpu_curr_gray.download()
        bgr = frame.copy()
        self.gpu_prev_p = self.gpu_next_p.clone()

        p0 = prev_p[0]
        p1 = next_p[0]

        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel()
            c, d = old.ravel()
            bgr = cv2.line(bgr, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            bgr = cv2.circle(bgr, (int(a), int(b)), 5, (0, 0, 255), -1)

        return bgr, frame
    
    def detect3D(self, count, vertex_map_curr, normal_map_prev):
        t0 = time.time()

        self.cv2_vertex_map_gpu_curr.upload(vertex_map_curr)

        # calc x,y,z 3D flow from 2D trajectory
        subpixel_interp_vertex_map_gpu = cv2.cuda.remap(self.cv2_vertex_map_gpu_curr, self.gpu_flow_x, self.gpu_flow_y, interpolation=cv2.INTER_LINEAR)
        
        print(subpixel_interp_vertex_map_gpu.size())

        #assign next 2D traj
        self.traj_motion_2D_x = self.gpu_flow_x  
        self.traj_motion_2D_y = self.gpu_flow_y

        buffer_idx = count % 5
        if buffer_idx == 0:
            self.traj_motion_3D_1 = subpixel_interp_vertex_map_gpu 
            print(0)
        elif buffer_idx == 1:
            self.traj_motion_3D_2 = subpixel_interp_vertex_map_gpu 
            print(1)
        elif buffer_idx == 2:
            self.traj_motion_3D_3 = subpixel_interp_vertex_map_gpu 
            print(2)
        elif buffer_idx == 3:
            self.traj_motion_3D_4 = subpixel_interp_vertex_map_gpu 
            print(3)
        else:
            self.traj_motion_3D_5 = subpixel_interp_vertex_map_gpu  
            print(4)  

        print(self.traj_motion_3D_1.size(),self.traj_motion_3D_2.size(),self.traj_motion_3D_3.size(),self.traj_motion_3D_4.size(),self.traj_motion_3D_5.size() )

        # point_disp_wrt_cam_gpu = cv2.cuda.subtract(subpixel_interp_vertex_map_gpu,self.cv2_vertex_map_gpu_prev) 
        # subpixel_interp_vertex_map = subpixel_interp_vertex_map_gpu.download()
        # point_disp_wrt_cam = point_disp_wrt_cam_gpu.download()

    

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
        
        # normal_map_prev_gpu = cp.asarray(normal_map_prev)
        # normal_map_prev_gpu /= cp.linalg.norm(normal_map_prev_gpu, axis=2, keepdims=True)

        # # Calculate norms of the normal map
        # norms_gpu = cp.linalg.norm(normal_map_prev_gpu, axis=2)

        # # Create a reference vector based on the norm check
        # reference_vector_gpu = cp.where(norms_gpu[..., None] < 0.9, cp.array([0, 1, 0], dtype=cp.float32)[None, None, :], 
        #                                 cp.array([1, 0, 0], dtype=cp.float32)[None, None, :])

        # # Cross product to get the x axis of the local coordinate system
        # x_axis_gpu = cp.cross(normal_map_prev_gpu, reference_vector_gpu)
        # x_axis_gpu /= cp.linalg.norm(x_axis_gpu, axis=2, keepdims=True)

        # # Cross product again to get the y axis of the local coordinate system
        # y_axis_gpu = cp.cross(normal_map_prev_gpu, x_axis_gpu)

        # # Rotation matrix is constructed by stacking the axes
        # rotation_matrix_gpu = cp.stack((x_axis_gpu, y_axis_gpu, normal_map_prev_gpu), axis=-1)

        # cp_point_disp_wrt_cam = cp.asarray(point_disp_wrt_cam)
        # #Use `einsum` to apply the rotation to point displacements on GPU
        # local_point_displacements_gpu = cp.einsum('...ji,...j->...i', rotation_matrix_gpu, cp_point_disp_wrt_cam)

        # #Download the result if necessary
        # local_point_displacements = local_point_displacements_gpu.get()

        t1 = time.time()
        #x_axis = x_axis_gpu.get()
        #y_axis = y_axis_gpu.get()

        print("EINSUM:",t1-t0)

        # x_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+x_axis.reshape(-1,3)*0.0001)
        # y_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+y_axis.reshape(-1,3)*0.0001)
        # z_local_axes = draw_lines(vertex_map_prev.reshape(-1,3),vertex_map_prev.reshape(-1,3)+normal_map_prev.reshape(-1,3)*0.0001)
        
        #disp_lines = draw_lines(vertex_map_prev.reshape(-1,3), subpixel_warped_vertex_map.reshape(-1,3))

        t2 = time.time()
        print("DRAW_LINES",t2-t1)
        self.cv2_vertex_map_gpu_prev = self.cv2_vertex_map_gpu_curr
        return self.traj_motion_2D_x, self.traj_motion_2D_y, self.traj_motion_3D_1, self.traj_motion_3D_2,self.traj_motion_3D_3,self.traj_motion_3D_4,self.traj_motion_3D_5

    def get_3D_disp_wrt_cam(self):
        pass

    def get_3D_disp_local(self):
        pass

    def vis_3D(self):
        g1_p = self.traj_motion_3D_1.download()
        self.g1.point.positions = o3d.core.Tensor(g1_p.reshape(-1,3),dtype=o3d.core.float32)
        g1d = self.g1.uniform_down_sample(every_k_points = 10)

        g2_p = self.traj_motion_3D_2.download()
        self.g2.point.positions = o3d.core.Tensor(g2_p.reshape(-1,3),dtype=o3d.core.float32)
        g2d = self.g2.uniform_down_sample(every_k_points = 10)

        g3_p = self.traj_motion_3D_3.download()
        self.g3.point.positions = o3d.core.Tensor(g3_p.reshape(-1,3),dtype=o3d.core.float32)
        g3d = self.g3.uniform_down_sample(every_k_points = 10)

        g4_p = self.traj_motion_3D_4.download()
        self.g4.point.positions = o3d.core.Tensor(g4_p.reshape(-1,3),dtype=o3d.core.float32)
        g4d = self.g4.uniform_down_sample(every_k_points = 10)

        g5_p = self.traj_motion_3D_5.download()
        self.g5.point.positions = o3d.core.Tensor(g5_p.reshape(-1,3),dtype=o3d.core.float32)
        g5d = self.g5.uniform_down_sample(every_k_points = 10)

        draw_lines(self.g1.point.positions.numpy(), self.g2.point.positions.numpy(), self.d1)
        self.d1.paint_uniform_color(o3d.core.Tensor([0,1,0]))
        draw_lines(self.g2.point.positions.numpy(), self.g3.point.positions.numpy(), self.d2)
        draw_lines(self.g3.point.positions.numpy(), self.g4.point.positions.numpy(), self.d3)
        draw_lines(self.g4.point.positions.numpy(), self.g5.point.positions.numpy(), self.d4)
        self.d4.paint_uniform_color(o3d.core.Tensor([1,0,0]))
        return self.g1,self.g2,self.g3,self.g4,self.g5,self.d1,self.d2,self.d3,self.d4

    
def init_grid(image_shape, step = 2):
    h,w = image_shape
    y,x = np.mgrid[step//2:h:step, step//2:w:step]
    grid = np.stack((x, y), axis=-1).astype(np.float32)
    return grid.reshape(1, -1, 2)

def draw_lines(start_points, end_points, line_set):
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

    line_set.point.positions = o3d.core.Tensor(line_points, dtype = o3d.core.float32)
    line_set.line.indices = o3d.core.Tensor(lines,dtype = o3d.core.int32)



        
