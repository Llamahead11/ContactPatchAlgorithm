import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time

class surfDetect:
    def __init__(self, depth_profile, debug_mode):
        self.surf = cv2.cuda.SURF_CUDA_create(5000,1,1)
        #self.surf.hessianThreshold = 400

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
        self.prev_img = None
        self.gpu_prev = cv2.cuda.GpuMat()
        self.gpu_prev_gray = cv2.cuda.GpuMat()
        self.ret = False
        self.matcher = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_L2)
        self.stream = cv2.cuda.Stream()
        self.gpu_curr = cv2.cuda.GpuMat()
        self.gpu_curr_gray = cv2.cuda.GpuMat()  

    def detect(self, curr_img, prev_rgbd_img, curr_rgbd_img, max_def, pcd):
        if self.ret == False:
            self.prev_img = curr_img
            self.gpu_prev.upload(curr_img, stream =self.stream)
            self.gpu_prev_gray = cv2.cuda.cvtColor(self.gpu_prev, cv2.COLOR_BGR2GRAY, stream = self.stream)
            self.ret = True
            return np.array([]),np.array([])

        start_time = time.time()
        self.gpu_curr.upload(curr_img, stream = self.stream)
        self.gpu_curr_gray = cv2.cuda.cvtColor(self.gpu_curr,cv2.COLOR_BGR2GRAY,stream=self.stream)

        #kp1, des1 = self.surf.detectAndCompute(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), None)
        #kp2, des2 = self.surf.detectAndCompute(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), None)

        kp1, des1 = self.surf.detectWithDescriptors(self.gpu_prev_gray, None)
        kp2, des2 = self.surf.detectWithDescriptors(self.gpu_curr_gray, None)
        matches = self.matcher.match(des1,des2)
        #knnMatchAsync(des1,des2,k=2,stream = self.stream)
        #match(des1, des2)

        # Download keypoints to CPU
        self.stream.waitForCompletion()
        cpu_kp1 = self.surf.downloadKeypoints(kp1)
        cpu_kp2 = self.surf.downloadKeypoints(kp2)

        #print("dscccccccccccccc", cpu_kp1)

        # Convert matched keypoints to point arrays
        pts1 = np.int32([cpu_kp1[m.queryIdx].pt for m in matches])
        pts2 = np.int32([cpu_kp2[m.trainIdx].pt for m in matches])
        #print("dcsddddddddddddddddddddddddddddddddddddddddd",pts1)
        end_time = time.time()
        # # Draw matches (optional)
        # img_matches = cv2.drawMatches(self.prev_img, cpu_kp1, curr_img, cpu_kp2, matches, None,
        #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # cv2.imshow('SURF', img_matches)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)

        # flann = cv2.FlannBasedMatcher(index_params,search_params)
    
        # matches = flann.knnMatch(des1,des2,k=2)

        matchesMask = [[0,0] for i in range(len(matches))]
        # pts1 = []
        # pts2 = []

        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.7*n.distance:
        #         matchesMask[i]=[1,0]
        #         pts1.append((int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])))
        #         pts2.append((int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1]))) 

        # pts1 = np.asarray(pts1)
        # pts2 = np.asarray(pts2)

        if self.debug_mode:
            draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

            mask = [m[0] for m in matchesMask]
            # Filter the keypoints using the mask
            filtered_kp1 = [kp for kp, m in zip(kp1, mask) if m == 1]
            prev_img = cv2.drawKeypoints(prev_img,filtered_kp1,None, color = (0,255,0), flags = 0)

            img = cv2.drawMatchesKnn(prev_img,kp1,curr_img,kp2,matches,None,**draw_params)
            plt.imshow(img), plt.show()
            
            for i in range(len(pts1)):
                prev_img = cv2.line(prev_img, pts1[i], pts2[i], (0,255,0), 2)
            plt.imshow(prev_img), plt.show()

        prev_depth = np.asarray(prev_rgbd_img.depth)
        curr_depth = np.asarray(curr_rgbd_img.depth)
    
        z_pts1 = prev_depth[pts1[:,1],pts1[:,0]] 
        z_pts2 = curr_depth[pts2[:,1],pts2[:,0]] 

        x_pts1 = self.x_calc(pts1,z_pts1)
        y_pts1 = self.y_calc(pts1,z_pts1) 

        x_pts2 = self.x_calc(pts2,z_pts2)
        y_pts2 = self.y_calc(pts2,z_pts2) 

        prev_3D_points = np.vstack((x_pts1,-y_pts1,z_pts2)).T
        curr_3D_points = np.vstack((x_pts2,-y_pts2,z_pts2)).T

        # filter out matches that deform over the max deformation calculated using registration
        mask = (np.linalg.norm(prev_3D_points-curr_3D_points, axis=1) <= max_def) & (prev_3D_points[:,2] > 0.07)
        prev_3D_points = prev_3D_points[mask]
        curr_3D_points = curr_3D_points[mask]

        #print(prev_3D_points)
        
        # device = o3d.core.Device("CPU:0")
        # dtype = o3d.core.float32
        # prev_pcd = o3d.t.geometry.PointCloud(device)
        # prev_pcd.point.positions = o3d.core.Tensor(prev_3D_points, dtype, device)
        # prev_pcd.estimate_normals(radius=0.05, max_nn=30)
        # prev_pcd.orient_normals_consistent_tangent_plane(k=10)
        # prev_normals = prev_pcd.point.normals.numpy()
        # curr_pcd = o3d.t.geometry.PointCloud(device)
        # curr_pcd.point.positions = o3d.core.Tensor(curr_3D_points, dtype, device)
        # curr_pcd.estimate_normals(radius=0.05, max_nn=30)
        # curr_pcd.orient_normals_consistent_tangent_plane(k=10)
        # def_lines = self.draw_lines(prev_3D_points,curr_3D_points)
        
        # #Determine local coordinate 
        # average_normal = prev_normals
        # average_normal /= np.linalg.norm(prev_normals,axis=1,keepdims=True)

        # norms = np.linalg.norm(average_normal, axis=1)
        # reference_vector = np.where(norms[:, None] < 0.9, np.array([0, 1, 0]), np.array([1, 0, 0]))
        
        # x_axis = np.cross(average_normal, reference_vector)
        # x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)  
        
        # y_axis = np.cross(average_normal, x_axis)

        # # Create a rotation matrix with z-axis aligned to the average normal
        # rotation_matrix =np.stack((x_axis, y_axis, average_normal), axis=-1)
        # point_displacements = curr_3D_points - prev_3D_points
        # #takes into account that rotation matrix is tranposed as calculation goes from old to new coordinate system 
        # local_point_displacements = np.einsum('nji,nj->ni', rotation_matrix, point_displacements) 
        # print(local_point_displacements)
        
        # # x_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+x_axis*0.007)
        # # y_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+y_axis*0.007)
        # # z_local_axes = self.draw_lines(prev_3D_points,prev_3D_points+average_normal*0.007)

        # x_component = self.draw_lines(prev_3D_points,prev_3D_points+x_axis*local_point_displacements[:,0][:,None])
        # y_component = self.draw_lines(prev_3D_points,prev_3D_points+y_axis*local_point_displacements[:,1][:,None])
        # z_component = self.draw_lines(prev_3D_points,prev_3D_points+average_normal*local_point_displacements[:,2][:,None])
        
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin = [0,0,0])
        
        # o3d.visualization.draw_geometries([prev_pcd.to_legacy(),curr_pcd.to_legacy(),def_lines,x_component,y_component,z_component,origin])

        # vmin, vmax = -0.005, 0.005 # only show values with +/-5mm of deformation
        # #vmin, vmax = np.min([local_point_displacements[:, 0].min(),local_point_displacements[:, 1].min(),local_point_displacements[:, 2].min()])

        # clipped_x = np.clip(local_point_displacements[:, 0], vmin, vmax)
        # clipped_y = np.clip(local_point_displacements[:, 1], vmin, vmax)
        # clipped_z = np.clip(local_point_displacements[:, 2], vmin, vmax)
        
        # x_colors = plt.get_cmap('plasma')((clipped_x - clipped_x.min()) / (clipped_x.max() - clipped_x.min()))
        # y_colors = plt.get_cmap('plasma')((clipped_y - clipped_y.min()) / (clipped_y.max() - clipped_y.min()))
        # z_colors = plt.get_cmap('plasma')((clipped_z - clipped_z.min()) / (clipped_z.max() - clipped_z.min()))
        # x_colors = x_colors[:, :3]
        # y_colors = y_colors[:, :3]
        # z_colors = z_colors[:, :3]

        # # Create a Matplotlib figure with three 3D subplots
        # fig = plt.figure(figsize=(15, 5))

        # # X Displacement Plot
        # ax1 = fig.add_subplot(131, projection='3d')
        # sc1 = ax1.scatter(prev_3D_points[:, 0], prev_3D_points[:, 1], prev_3D_points[:, 2], c=clipped_x, cmap='plasma', s=2)
        # ax1.set_title("X Displacement")
        # ax1.set_xlabel("X")
        # ax1.set_ylabel("Y")
        # ax1.set_zlabel("Z")
        # fig.colorbar(sc1, ax=ax1, orientation='vertical', label='X Displacement [m]')

        # # Y Displacement Plot
        # ax2 = fig.add_subplot(132, projection='3d')
        # sc2 = ax2.scatter(prev_3D_points[:, 0], prev_3D_points[:, 1], prev_3D_points[:, 2], c=clipped_y, cmap='plasma', s=2)
        # ax2.set_title("Y Displacement")
        # ax2.set_xlabel("X")
        # ax2.set_ylabel("Y")
        # ax2.set_zlabel("Z")
        # fig.colorbar(sc2, ax=ax2, orientation='vertical', label='Y Displacement [m]')

        # # Z Displacement Plot
        # ax3 = fig.add_subplot(133, projection='3d')
        # sc3 = ax3.scatter(prev_3D_points[:, 0], prev_3D_points[:, 1], prev_3D_points[:, 2], c=clipped_z, cmap='plasma', s=2)
        # ax3.set_title("Z Displacement")
        # ax3.set_xlabel("X")
        # ax3.set_ylabel("Y")
        # ax3.set_zlabel("Z")
        # fig.colorbar(sc3, ax=ax3, orientation='vertical', label='Z Displacement [m]')

        # # Show the figure in a non-blocking way
        # plt.show(block=False)

        # # Create a Matplotlib figure with three subplots for the colorbars
        # fig, ax = plt.subplots()

        # # Set up colorbars
        # cmap = plt.get_cmap('plasma')
        # norm = plt.Normalize(vmin, vmax)

        # # Scalar mappers for colorbars
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        # # Add colorbars to subplots
        # fig.colorbar(sm, ax=ax, orientation='vertical', label='Displacement [m]')

        # # Show the colorbar figure in non-blocking mode
        # plt.show(block=False)

        # t2_x_pcd = o3d.t.geometry.PointCloud(device)
        # t2_x_pcd.point.positions = o3d.core.Tensor(prev_3D_points, dtype, device)
        # t2_x_pcd.point.colors = o3d.core.Tensor(x_colors, dtype, device)
        # t2_x_pcd.point.normals = prev_pcd.point.normals

        # o3d.visualization.draw_geometries([t2_x_pcd.to_legacy(), origin])

        # t2_y_pcd = o3d.t.geometry.PointCloud(device)
        # t2_y_pcd.point.positions = o3d.core.Tensor(prev_3D_points, dtype, device)
        # t2_y_pcd.point.colors = o3d.core.Tensor(y_colors, dtype, device)
        # t2_y_pcd.point.normals = prev_pcd.point.normals

        # o3d.visualization.draw_geometries([t2_y_pcd.to_legacy(), origin])

        # t2_z_pcd = o3d.t.geometry.PointCloud(device)
        # t2_z_pcd.point.positions = o3d.core.Tensor(prev_3D_points, dtype, device)
        # t2_z_pcd.point.colors = o3d.core.Tensor(z_colors, dtype, device)
        # t2_z_pcd.point.normals = prev_pcd.point.normals

        # o3d.visualization.draw_geometries([t2_z_pcd.to_legacy(), origin])
        self.gpu_prev_gray = self.gpu_curr_gray
        print("surf detection",end_time-start_time)
        return prev_3D_points, curr_3D_points
    
    def x_calc(self, xy_pts, z_pts):
        return (xy_pts[:,0] - self.cx) * z_pts / self.fx 
    
    def y_calc(self, xy_pts, z_pts):
        return (xy_pts[:,1] - self.cy) * z_pts / self.fy 
    
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
    
    def uv_pixel_to_xyz_point(self, pixel_coords):
        return np.array([pixel_coords[:,0] + pixel_coords[:,1]*(self.width)])