import cv2
import numpy as np
import open3d as o3d
import time 
import matplotlib.pyplot as plt
import vidVisualiser as vV

class lkDense:
    def __init__(self,depth_profile,debug_mode):
        self.debug_mode = debug_mode
        self.depth_num = depth_profile
        self.count = 0
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

    def detect(self, prev_img, curr_img, prev_rgbd_img, curr_rgbd_img, max_def):
        cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        flow = cv2.optflow.calcOpticalFlowSparseToDense(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY),cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY),None, 
                                    grid_step = 4,
                                    k = 128,
                                    sigma = 0.5,
                                    use_post_proc =  True,
                                    fgs_lambda = 500,
                                    fgs_sigma = 1.5)
        magnitude, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
        mask = np.zeros_like(prev_img) 
        mask[..., 1] = 255
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        #cv2.imshow("dense optical flow", rgb) 
        k = cv2.waitKey(1)
        
        prev_depth = np.asarray(prev_rgbd_img.depth)
        curr_depth = np.asarray(curr_rgbd_img.depth)

        dx = np.round(flow[:,:,0]).astype(int)
        dy = np.round(flow[:,:,1]).astype(int)
        
        y_coords, x_coords = np.where(magnitude >= 0.1)
        #print(np.where(magnitude>=1))
        dx = dx[y_coords,x_coords].flatten()
        dy = dy[y_coords,x_coords].flatten()

        x_coords2 = x_coords + dx
        y_coords2 = y_coords + dy

        x_coords2 = np.clip(x_coords2, a_min=None, a_max=self.width-1)
        y_coords2 = np.clip(y_coords2, a_min=None, a_max=self.height-1)

        z_pts1 = prev_depth[y_coords, x_coords]
        x_pts1 = self.x_calc(x_coords,z_pts1)
        y_pts1 = self.y_calc(y_coords,z_pts1)

        z_pts2 = curr_depth[y_coords2,x_coords2]
        x_pts2 = self.x_calc(x_coords2,z_pts2)
        y_pts2 = self.y_calc(y_coords2,z_pts2)

        prev_3D_points = np.vstack((x_pts1,-y_pts1,z_pts2)).T
        curr_3D_points = np.vstack((x_pts2,-y_pts2,z_pts2)).T

        # filter out matches that deform over the max deformation calculated using registration
        mask = (np.linalg.norm(prev_3D_points-curr_3D_points, axis=1) <= max_def) & (prev_3D_points[:,2] > 0.07)
        prev_3D_points = prev_3D_points[mask]
        curr_3D_points = curr_3D_points[mask]
        prev_3D_points = prev_3D_points[::10]
        curr_3D_points = curr_3D_points[::10]

        print(prev_3D_points)
        
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        prev_pcd = o3d.t.geometry.PointCloud(device)
        prev_pcd.point.positions = o3d.core.Tensor(prev_3D_points, dtype, device)
        #prev_pcd = prev_pcd.uniform_down_sample(every_k_points=10)
        prev_pcd.estimate_normals(radius=0.05, max_nn=30)
        prev_pcd.orient_normals_consistent_tangent_plane(k=10)
        prev_normals = prev_pcd.point.normals.numpy()
        curr_pcd = o3d.t.geometry.PointCloud(device)
        curr_pcd.point.positions = o3d.core.Tensor(curr_3D_points, dtype, device)
        #curr_pcd = curr_pcd.uniform_down_sample(every_k_points=10)
        curr_pcd.estimate_normals(radius=0.05, max_nn=30)
        curr_pcd.orient_normals_consistent_tangent_plane(k=10)
        def_lines = self.draw_lines(prev_3D_points,curr_3D_points)
        
        #o3d.visualization.draw_geometries([prev_pcd.to_legacy(),curr_pcd.to_legacy(),def_lines])

        #Determine local coordinate 
        average_normal = prev_normals
        average_normal /= np.linalg.norm(prev_normals,axis=1,keepdims=True)

        norms = np.linalg.norm(average_normal, axis=1)
        reference_vector = np.where(norms[:, None] < 0.9, np.array([0, 1, 0]), np.array([1, 0, 0]))
        
        x_axis = np.cross(average_normal, reference_vector)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)  
        
        y_axis = np.cross(average_normal, x_axis)

        # Create a rotation matrix with z-axis aligned to the average normal
        rotation_matrix =np.stack((x_axis, y_axis, average_normal), axis=-1)
        point_displacements = curr_3D_points - prev_3D_points
        #takes into account that rotation matrix is tranposed as calculation goes from old to new coordinate system 
        local_point_displacements = np.einsum('nji,nj->ni', rotation_matrix, point_displacements) 
        print(local_point_displacements)
        
        # x_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+x_axis*0.007)
        # y_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+y_axis*0.007)
        # z_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+average_normal*0.007)

        x_component = self.draw_lines(prev_3D_points,prev_3D_points+x_axis*local_point_displacements[:,0][:,None])
        y_component = self.draw_lines(prev_3D_points,prev_3D_points+y_axis*local_point_displacements[:,1][:,None])
        z_component = self.draw_lines(prev_3D_points,prev_3D_points+average_normal*local_point_displacements[:,2][:,None])
        
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin = [0,0,0])
        
        #o3d.visualization.draw_geometries([prev_pcd.to_legacy(),curr_pcd.to_legacy(),def_lines,x_component,y_component,z_component,origin])

        vmin, vmax = -0.002, 0.002 # only show values with +/-5mm of deformation
        
        end_time = time.time()
        vV.plotAndSavePlt(local_point_displacements,prev_3D_points,vmin,vmax)
        
        # Show the figure in a non-blocking way
        plt.savefig("./LKD_video/{}.png".format(self.count))
        #plt.show(block=False)
        self.count +=1
        print("LK_Dense detection",end_time-start_time)
        return prev_3D_points, curr_3D_points


    def x_calc(self, x_pts, z_pts):
        return (x_pts - self.cx) * z_pts / self.fx 
    
    def y_calc(self, y_pts, z_pts):
        return (y_pts - self.cy) * z_pts / self.fy 
    
    def draw_lines(self, start_points, end_points):
        line_starts = start_points
        line_ends = end_points

        lines = [[i, i + len(start_points)] for i in range(len(start_points))]
        line_points = np.vstack((line_starts, line_ends))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        return line_set