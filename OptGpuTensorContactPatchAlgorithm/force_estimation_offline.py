import open3d as o3d
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import median_filter

def draw_lines_lineset(start_points, end_points, line_set):
    line_start = start_points.copy() #cp.ascontiguousarray(cp.from_dlpack(start_points.to_dlpack()))
    line_end = end_points.copy() #cp.ascontiguousarray(cp.from_dlpack(end_points.to_dlpack()))

    dist = cp.linalg.norm(line_end-line_start,axis=1)
    # # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # # line_points = np.vstack((start_points, end_points))
    # # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    valid_dist = dist < 0.1
    mask = valid_dist #(line_start != 0.0).all(axis=1) & (line_end != 0.0).all(axis=1) &
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 
    
    disp = line_valid_end[:,0] - line_valid_start[:,0]
    #disp = (((local.reshape(-1,3))[::20])[mask])[:,2]
    #disp = cp.linalg.norm(line_end-line_start,axis=1)

    if disp.shape[0] != 0:
        normalized =  (disp - disp.min()) / (disp.max() - disp.min())
        print(disp.min(),disp.max())
        #(vel[mask] - vel[mask].mean()) / vel[mask].std() #
        # Use a colormap (e.g., viridis, jet, plasma)
        colormap = plt.cm.get_cmap('viridis')
        colors = colormap(normalized.get())[:, :3]  # Drop alpha channel
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
        line_set.line.colors = o3d.core.Tensor(colors, dtype=o3d.core.float32, device = o3d.core.Device("CUDA:0"))

def draw_lines_lineset_color(start_points, end_points, line_set, F_colors):
    line_start = start_points.copy() #cp.ascontiguousarray(cp.from_dlpack(start_points.to_dlpack()))
    line_end = end_points.copy() #cp.ascontiguousarray(cp.from_dlpack(end_points.to_dlpack()))

    dist = cp.linalg.norm(line_end-line_start,axis=1)
    # # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # # line_points = np.vstack((start_points, end_points))
    # # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    valid_dist = (dist < 0.1)
    mask = (F_colors.squeeze() < 2.5) & (F_colors.squeeze() > 0)#(line_start != 0.0).all(axis=1) & (line_end != 0.0).all(axis=1) &
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start[mask]
    line_valid_end = line_end[mask] 
    
    disp = F_colors.squeeze()[mask]
    #disp = (((local.reshape(-1,3))[::20])[mask])[:,2]
    #disp = cp.linalg.norm(line_end-line_start,axis=1)

    if disp.shape[0] != 0:
        normalized =  (disp - disp.min()) / (disp.max() - disp.min())
        print(disp.min(),disp.max())
        #(vel[mask] - vel[mask].mean()) / vel[mask].std() #
        # Use a colormap (e.g., viridis, jet, plasma)
        colormap = plt.cm.get_cmap('viridis')
        colors = colormap(normalized.get())[:, :3]  # Drop alpha channel
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
        line_set.line.colors = o3d.core.Tensor(colors, dtype=o3d.core.float32, device = o3d.core.Device("CUDA:0"))
        print(line_set.point.positions.shape)
        print(line_set.line.indices.shape)
        print(line_set.line.colors.shape)

# Current Config (Deformed Configuration)
# linspace of origins to outer_def

data = np.load("saved_arrays/iteration_159.npz")
origins_uh = cp.asarray(data['orig']) #.reshape(480,848,3)
hit_point_uh = cp.asarray(data['hit_p']) #.reshape(480,848,3)
hit_point_o_uh = cp.asarray(data['hit_p_o']) #.reshape(480,848,3)
inv_full_T = cp.asarray(data['inv_T'])

origins_h = cp.hstack([origins_uh, cp.ones((origins_uh.shape[0], 1))])
hit_point_h = cp.hstack([hit_point_uh, cp.ones((hit_point_uh.shape[0], 1))])  # (N, 4)
hit_point_o_h = cp.hstack([hit_point_o_uh, cp.ones((hit_point_o_uh.shape[0], 1))])  # (N, 4)

# Apply transformation
transformed_origins_h = origins_h @ inv_full_T  # still (N, 4)
transformed_hit_point_h = hit_point_h @ inv_full_T  # still (N, 4)
transformed_hit_point_o_h = hit_point_o_h @ inv_full_T  # still (N, 4)

# Drop homogeneous coordinate → back to (N, 3)
origins = transformed_origins_h[:, :3].reshape(480,848,3)
hit_point = transformed_hit_point_h[:, :3].reshape(480,848,3)
hit_point_o = transformed_hit_point_o_h[:, :3].reshape(480,848,3)

image_inner = np.ones((480,848),dtype=np.uint8)
image_outer = np.ones((480,848),dtype=np.uint8)
mask_valid_inner = hit_point[:,:,2] != 0 
mask_valid_outer = hit_point_o[:,:,2] != 0 
image_inner[mask_valid_inner.get()] = 0
image_outer[mask_valid_outer.get()] = 0
plt.figure()
plt.imshow(image_inner)
plt.figure()
plt.imshow(image_outer)

#Order knn ?

#Filter image to fill in small gaps
filtered_hitpoint = cp.empty_like(hit_point)
filtered_hitpoint_o = cp.empty_like(hit_point_o)

temp_hp = hit_point
temp_hpo = hit_point_o
for i in range(3):
    for i in range(3):  # Apply median filter to each of X, Y, Z separately
        filtered_hitpoint[..., i] = median_filter(temp_hp[..., i], size=3)
        filtered_hitpoint_o[..., i] = median_filter(temp_hpo[..., i],size=3)
        mask_valid_inner = hit_point[:,:,2] != 0 
        mask_valid_outer = hit_point_o[:,:,2] != 0 
        hit_point[~mask_valid_inner] = filtered_hitpoint[~mask_valid_inner]
        hit_point_o[~mask_valid_outer] = filtered_hitpoint_o[~mask_valid_outer]
    temp_hp = filtered_hitpoint.copy()
    temp_hpo = filtered_hitpoint_o.copy()



image_inner_f = np.ones((480,848),dtype=np.uint8)
image_outer_f = np.ones((480,848),dtype=np.uint8)
mask_valid_inner_f = filtered_hitpoint[:,:,2] != 0 
mask_valid_outer_f = filtered_hitpoint_o[:,:,2] != 0 
image_inner_f[mask_valid_inner_f.get()] = 0
image_outer_f[mask_valid_outer_f.get()] = 0

pcd1 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
pcd1.point.positions = o3d.core.Tensor.from_dlpack(hit_point.reshape(-1,3, order='C').toDlpack())
pcd2 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
pcd2.point.positions = o3d.core.Tensor.from_dlpack(hit_point_o.reshape(-1,3, order='C').toDlpack())
pcd3 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
pcd3.point.positions = o3d.core.Tensor.from_dlpack(filtered_hitpoint.reshape(-1,3, order='C').toDlpack())
pcd4 = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
pcd4.point.positions = o3d.core.Tensor.from_dlpack(filtered_hitpoint_o.reshape(-1,3, order='C').toDlpack())

# colors = np.linspace(0, 1, pcd.point.positions.shape[0])
# colors = np.stack([colors, np.zeros_like(colors), 1 - colors], axis=1)
# pcd.point.colors = o3d.core.Tensor.from_numpy(colors).cuda()

o3d.visualization.draw([pcd1.cpu(),pcd2.cpu(),pcd3.cpu(),pcd4.cpu()])

plt.figure()
plt.imshow(image_inner_f)
plt.figure()
plt.imshow(image_outer_f)

#Maximal Rectangle


# from sklearn.neighbors import NearestNeighbors

# def detect_unordered_knn(pts, k=4, max_dist=0.05):
#     nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
#     dists, _ = nbrs.kneighbors(pts)
#     # Skip first column (distance to self is 0)
#     avg_dist = dists[:, 1:].mean(axis=1)
#     unordered_mask = avg_dist > max_dist
#     unordered_indices = np.where(unordered_mask)[0]
#     return unordered_indices

# pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
# pcd.point.positions = o3d.core.Tensor.from_dlpack(hit_point.reshape(-1,3, order='C')[:400000].toDlpack())

# colors = np.linspace(0, 1, pcd.point.positions.shape[0])
# colors = np.stack([colors, np.zeros_like(colors), 1 - colors], axis=1)
# pcd.point.colors = o3d.core.Tensor.from_numpy(colors).cuda()

# o3d.visualization.draw([pcd.cpu()])

div = 4#4
layers = 10#20
origins = origins[60:380:div,312:740:div,:].reshape(-1,3)
hit_point = hit_point[60:380:div,312:740:div,:].reshape(-1,3)
hit_point_o = hit_point_o[60:380:div,312:740:div,:].reshape(-1,3)

sh1 = 380 - 60
sh2 = 740 - 312

print("origins.shape", origins.shape)
print("hit_point.shape", hit_point.shape)
print("hit_point_o.shape", hit_point_o.shape)

dist = hit_point - origins
outer_def = hit_point_o - dist

#volumetric_def_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
volumetric_def_points = cp.linspace(origins,outer_def,layers,True,False,cp.float64,axis=0)
print("volumetric_def_points shape", volumetric_def_points.shape)

# Reference Config (Undeformed Configuration)
#linspace of hit_point to hit_point_o
volumetric_undef_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
print("volumetric_def_points shape", volumetric_undef_points.shape)



#Deformation Gradient F using Finite Differences Method
volumetric_def_points = volumetric_def_points.reshape(layers,sh1//div,sh2//div,3)
volumetric_undef_points = volumetric_undef_points.reshape(layers,sh1//div,sh2//div,3)

dx_dzeta1 = cp.empty((layers,sh1//div,sh2//div),cp.float32)
dx_dzeta2 = cp.empty_like(dx_dzeta1)
dx_dzeta3 = cp.empty_like(dx_dzeta1)
dy_dzeta1 = cp.empty_like(dx_dzeta1)
dy_dzeta2 = cp.empty_like(dx_dzeta1)
dy_dzeta3 = cp.empty_like(dx_dzeta1)
dz_dzeta1 = cp.empty_like(dx_dzeta1)
dz_dzeta2 = cp.empty_like(dx_dzeta1)
dz_dzeta3 = cp.empty_like(dx_dzeta1)
dX_dzeta1 = cp.empty_like(dx_dzeta1)
dX_dzeta2 = cp.empty_like(dx_dzeta1)
dX_dzeta3 = cp.empty_like(dx_dzeta1)
dY_dzeta1 = cp.empty_like(dx_dzeta1)
dY_dzeta2 = cp.empty_like(dx_dzeta1)
dY_dzeta3 = cp.empty_like(dx_dzeta1)
dZ_dzeta1 = cp.empty_like(dx_dzeta1)
dZ_dzeta2 = cp.empty_like(dx_dzeta1)
dZ_dzeta3 = cp.empty_like(dx_dzeta1)

##Foward Difference
# dx[:,:,:-1] = volumetric_def_points[:,:,1:,0] - volumetric_def_points[:,:,0:-1,0]
# dy[:,:-1,:] = volumetric_def_points[:,1:,:,1] - volumetric_def_points[:,0:-1,:,1]
# dz[:-1,:,:] = volumetric_def_points[1:,:,:,2] - volumetric_def_points[0:-1,:,:,2]

# dX[:,:,:-1] = volumetric_undef_points[:,:,1:,0] - volumetric_undef_points[:,:,0:-1,0]
# dY[:,:-1,:] = volumetric_undef_points[:,1:,:,1] - volumetric_undef_points[:,0:-1,:,1]
# dZ[:-1,:,:] = volumetric_undef_points[1:,:,:,2] - volumetric_undef_points[0:-1,:,:,2]

## Central Difference
dx_dzeta1[:,:,1:-1] = (volumetric_def_points[:,:,2:,0] - volumetric_def_points[:,:,0:-2,0])/2
dx_dzeta2[:,1:-1,:] = (volumetric_def_points[:,2:,:,0] - volumetric_def_points[:,0:-2,:,0])/2
dx_dzeta3[1:-1,:,:] = (volumetric_def_points[2:,:,:,0] - volumetric_def_points[0:-2,:,:,0])/2

dy_dzeta1[:,:,1:-1] = (volumetric_def_points[:,:,2:,1] - volumetric_def_points[:,:,0:-2,1])/2
dy_dzeta2[:,1:-1,:] = (volumetric_def_points[:,2:,:,1] - volumetric_def_points[:,0:-2,:,1])/2
dy_dzeta3[1:-1,:,:] = (volumetric_def_points[2:,:,:,1] - volumetric_def_points[0:-2,:,:,1])/2

dz_dzeta1[:,:,1:-1] = (volumetric_def_points[:,:,2:,2] - volumetric_def_points[:,:,0:-2,2])/2
dz_dzeta2[:,1:-1,:] = (volumetric_def_points[:,2:,:,2] - volumetric_def_points[:,0:-2,:,2])/2
dz_dzeta3[1:-1,:,:] = (volumetric_def_points[2:,:,:,2] - volumetric_def_points[0:-2,:,:,2])/2

dX_dzeta1[:,:,1:-1] = (volumetric_undef_points[:,:,2:,0] - volumetric_undef_points[:,:,0:-2,0])/2
dX_dzeta2[:,1:-1,:] = (volumetric_undef_points[:,2:,:,0] - volumetric_undef_points[:,0:-2,:,0])/2
dX_dzeta3[1:-1,:,:] = (volumetric_undef_points[2:,:,:,0] - volumetric_undef_points[0:-2,:,:,0])/2

dY_dzeta1[:,:,1:-1] = (volumetric_undef_points[:,:,2:,1] - volumetric_undef_points[:,:,0:-2,1])/2
dY_dzeta2[:,1:-1,:] = (volumetric_undef_points[:,2:,:,1] - volumetric_undef_points[:,0:-2,:,1])/2
dY_dzeta3[1:-1,:,:] = (volumetric_undef_points[2:,:,:,1] - volumetric_undef_points[0:-2,:,:,1])/2

dZ_dzeta1[:,:,1:-1] = (volumetric_undef_points[:,:,2:,2] - volumetric_undef_points[:,:,0:-2,2])/2
dZ_dzeta2[:,1:-1,:] = (volumetric_undef_points[:,2:,:,2] - volumetric_undef_points[:,0:-2,:,2])/2
dZ_dzeta3[1:-1,:,:] = (volumetric_undef_points[2:,:,:,2] - volumetric_undef_points[0:-2,:,:,2])/2

dx_dzeta1[:,:,0] = volumetric_def_points[:,:,1,0] - volumetric_def_points[:,:,0,0]
dx_dzeta2[:,0,:] = volumetric_def_points[:,1,:,0] - volumetric_def_points[:,0,:,0]
dx_dzeta3[0,:,:] = volumetric_def_points[1,:,:,0] - volumetric_def_points[0,:,:,0]
dy_dzeta1[:,:,0] = volumetric_def_points[:,:,1,1] - volumetric_def_points[:,:,0,1]
dy_dzeta2[:,0,:] = volumetric_def_points[:,1,:,1] - volumetric_def_points[:,0,:,1]
dy_dzeta3[0,:,:] = volumetric_def_points[1,:,:,1] - volumetric_def_points[0,:,:,1]
dz_dzeta1[:,:,0] = volumetric_def_points[:,:,1,2] - volumetric_def_points[:,:,0,2]
dz_dzeta2[:,0,:] = volumetric_def_points[:,1,:,2] - volumetric_def_points[:,0,:,2]
dz_dzeta3[0,:,:] = volumetric_def_points[1,:,:,2] - volumetric_def_points[0,:,:,2]

dX_dzeta1[:,:,0] = volumetric_undef_points[:,:,1,0] - volumetric_undef_points[:,:,0,0]
dX_dzeta2[:,0,:] = volumetric_undef_points[:,1,:,0] - volumetric_undef_points[:,0,:,0]
dX_dzeta3[0,:,:] = volumetric_undef_points[1,:,:,0] - volumetric_undef_points[0,:,:,0]
dY_dzeta1[:,:,0] = volumetric_undef_points[:,:,1,1] - volumetric_undef_points[:,:,0,1]
dY_dzeta2[:,0,:] = volumetric_undef_points[:,1,:,1] - volumetric_undef_points[:,0,:,1]
dY_dzeta3[0,:,:] = volumetric_undef_points[1,:,:,1] - volumetric_undef_points[0,:,:,1]
dZ_dzeta1[:,:,0] = volumetric_undef_points[:,:,1,2] - volumetric_undef_points[:,:,0,2]
dZ_dzeta2[:,0,:] = volumetric_undef_points[:,1,:,2] - volumetric_undef_points[:,0,:,2]
dZ_dzeta3[0,:,:] = volumetric_undef_points[1,:,:,2] - volumetric_undef_points[0,:,:,2]

dx_dzeta1[:,:,-1] = volumetric_def_points[:,:,-1,0] - volumetric_def_points[:,:,-2,0]
dx_dzeta2[:,-1,:] = volumetric_def_points[:,-1,:,0] - volumetric_def_points[:,-2,:,0]
dx_dzeta3[-1,:,:] = volumetric_def_points[-1,:,:,0] - volumetric_def_points[-2,:,:,0]
dy_dzeta1[:,:,-1] = volumetric_def_points[:,:,-1,1] - volumetric_def_points[:,:,-2,1]
dy_dzeta2[:,-1,:] = volumetric_def_points[:,-1,:,1] - volumetric_def_points[:,-2,:,1]
dy_dzeta3[-1,:,:] = volumetric_def_points[-1,:,:,1] - volumetric_def_points[-2,:,:,1]
dz_dzeta1[:,:,-1] = volumetric_def_points[:,:,-1,2] - volumetric_def_points[:,:,-2,2]
dz_dzeta2[:,-1,:] = volumetric_def_points[:,-1,:,2] - volumetric_def_points[:,-2,:,2]
dz_dzeta3[-1,:,:] = volumetric_def_points[-1,:,:,2] - volumetric_def_points[-2,:,:,2]

dX_dzeta1[:,:,-1] = volumetric_undef_points[:,:,-1,0] - volumetric_undef_points[:,:,-2,0]
dX_dzeta2[:,-1,:] = volumetric_undef_points[:,-1,:,0] - volumetric_undef_points[:,-2,:,0]
dX_dzeta3[-1,:,:] = volumetric_undef_points[-1,:,:,0] - volumetric_undef_points[-2,:,:,0]
dY_dzeta1[:,:,-1] = volumetric_undef_points[:,:,-1,1] - volumetric_undef_points[:,:,-2,1]
dY_dzeta2[:,-1,:] = volumetric_undef_points[:,-1,:,1] - volumetric_undef_points[:,-2,:,1]
dY_dzeta3[-1,:,:] = volumetric_undef_points[-1,:,:,1] - volumetric_undef_points[-2,:,:,1]
dZ_dzeta1[:,:,-1] = volumetric_undef_points[:,:,-1,2] - volumetric_undef_points[:,:,-2,2]
dZ_dzeta2[:,-1,:] = volumetric_undef_points[:,-1,:,2] - volumetric_undef_points[:,-2,:,2]
dZ_dzeta3[-1,:,:] = volumetric_undef_points[-1,:,:,2] - volumetric_undef_points[-2,:,:,2]

dx_dzeta = cp.stack([dx_dzeta1, dx_dzeta2, dx_dzeta3], axis=-1)
dy_dzeta = cp.stack([dy_dzeta1, dy_dzeta2, dy_dzeta3], axis=-1)
dz_dzeta = cp.stack([dz_dzeta1, dz_dzeta2, dz_dzeta3], axis=-1)

print("dx_dzeta shape",dx_dzeta.shape)

# Jx shape: (..., 3, 3)
Jx = cp.stack([dx_dzeta, dy_dzeta, dz_dzeta], axis=-2)
inv_Jx = cp.linalg.inv(Jx)
print("inv_Jx shape", inv_Jx.shape)

dX_dzeta = cp.stack([dX_dzeta1, dX_dzeta2, dX_dzeta3], axis=-1)
dY_dzeta = cp.stack([dY_dzeta1, dY_dzeta2, dY_dzeta3], axis=-1)
dZ_dzeta = cp.stack([dZ_dzeta1, dZ_dzeta2, dZ_dzeta3], axis=-1)

JX = cp.stack([dX_dzeta, dY_dzeta, dZ_dzeta], axis=-2)

F = Jx @ cp.linalg.inv(JX)
print("F shape", F.shape)
I = cp.eye(3, dtype=cp.float32)
diff = cp.abs(F - I)
print("Mean deviation from identity in undeformed case:", cp.mean(diff))

# cond_JX = cp.linalg.cond(JX)
# print("Max condition number:", cond_JX.max().item())
# print("Mean condition number:", cond_JX.mean().item())

det_JX = cp.linalg.det(JX)
print("Min determinant of JX:", det_JX.min().item())

#eps = 1e-4

# dX_safe = cp.where(cp.abs(dX) < eps, eps, dX)
# dY_safe = cp.where(cp.abs(dY) < eps, eps, dY)
# dZ_safe = cp.where(cp.abs(dZ) < eps, eps, dZ)

# F = cp.stack([
#     cp.stack([dx/dX_safe, dx/dY_safe, dx/dZ_safe], axis=-1),
#     cp.stack([dy/dX_safe, dy/dY_safe, dy/dZ_safe], axis=-1),
#     cp.stack([dz/dX_safe, dz/dY_safe, dz/dZ_safe], axis=-1),
#     ], axis=-2)
# print("F shape", F.shape) #(10,480,848,3,3)


# del dx, dy, dz, dX, dY, dZ
# del volumetric_def_points, volumetric_undef_points

F_T = F.transpose(0, 1, 2, 4, 3)
C = cp.matmul(F_T, F)
print("C shape", C.shape)
#del F
I = cp.eye(3, dtype=cp.float32)[None, None, None, :, :]
E = 0.5*(C-I) 
print("E shape", E.shape)

#Yeoh model
#U = C10(I1 − 3) + C20(I1 − 3)2 + C30(I1 − 3)3                   +1/D1(J^el− 3)2 +1/D2(J^el − 3)4 +1/D3(J^el − 3)6

C10 = 473.685
C20 = -119.853
C30 = 34.293

D1 = 5.085*(10^(-8))

# W = C1(I1 - 3) + C2(I1 − 3)^2 + C3(I1 − 3)^3
# But I1 = trace(C) 
#  But C = 2E + I (E = (1/2)*(C-I))
# So I1 = trace(2E + I) = 2*trace(E) + 3
# Therefore, I1 - 3 = 2*trace(E)

# W = C1(2tr(E)) + C2(2tr(E))^2 + C3(2tr(E))^3
# now that W is a function of the trace of E = E11 + E22 + E33

# let x = tr(E), the W = C1*2x + C2*(2x)^2 + C3*(2x)^3

# Now dW/dx = 2*C1 + 2*C2*2x*2 + 3*C3*(2x)^2 * 2
#           = 2*C1 + 8*C2*x + 24*C3*x^2

# Now S = dW/dE = (dW/dx)*(dx/dE) but dx/dE = dtr(E)/dE = E11 + E22 + E33 / dE = I

# Therefore = S = (2*C1 + 8*C2*tr(E) + 24*C3*(tr(E))^2)*I

trE = cp.trace(E, axis1=-2, axis2=-1)

dWdx = 2*C10 + 8*C20*trE + 24*C30*(trE)**2
dxdE = cp.eye(3)[None,None,None,:,:] 

S = dWdx[:,:,:,None,None] * dxdE
print("S shape", S.shape)

F_reshaped = F.reshape(-1, 3, 3)              
detF = cp.linalg.det(F_reshaped)              
detF = detF.reshape(F.shape[:-2])
detF = cp.clip(detF, 1e-5, 1e5)
print("detF shape", detF.shape)

#T = (1/detF)*F_T*S*F
FS = cp.matmul(F,S)
T = cp.matmul(FS,F_T)
T /= detF[:,:,:,None,None]
print("T shape", T.shape)

divT = cp.empty((layers,sh1//div,sh2//div,3),cp.float32)

eps = 1e-4

# # dx_safe = cp.where(cp.abs(dx) < eps, eps, dx)
# # dy_safe = cp.where(cp.abs(dy) < eps, eps, dy)
# # dz_safe = cp.where(cp.abs(dz) < eps, eps, dz)
# print("dx_safe stats", cp.min(dx_safe).item(), cp.max(dx_safe).item(), cp.mean(dx_safe).item())
# plt.figure(figsize=(6, 4))
# plt.hist(dx_safe.reshape(-1,1).get(), bins=500, color='steelblue', edgecolor='black')
# plt.xlabel("Distance to fitted plane (m)")
# plt.ylabel("Number of points")
# # plt.xlim([-0.020,0.010])
# # plt.ylim([0,1400])
# plt.title("Histogram of distances")
# plt.grid(True)
# plt.tight_layout()
# plt.show(block = False)
#Central diff
# divT[:,:,1:-1,0] = (T[:,:,2:,0,0]-T[:,:,0:-2,0,0])/dx[:,:,1:-1]+ (T[:,:,2:,0,1]-T[:,:,0:-2,0,1])/dy[:,:,1:-1] + (T[:,:,2:,0,2]-T[:,:,0:-2,0,2])/dz[:,:,1:-1]
# divT[:,1:-1,:,1] = (T[:,2:,:,1,0]-T[:,0:-2,:,1,0])/dx[:,1:-1,:] + (T[:,2:,:,1,1]-T[:,0:-2,:,1,1])/dy[:,1:-1,:] + (T[:,2:,:,1,2]-T[:,0:-2,:,1,2])/dz[:,1:-1,:]
# divT[1:-1,:,:,2] = (T[2:,:,:,2,0]-T[0:-2,:,:,2,0])/dx[1:-1,:,:] + (T[2:,:,:,2,1]-T[0:-2,:,:,2,1])/dy[1:-1,:,:] + (T[2:,:,:,2,2]-T[0:-2,:,:,2,2])/dz[1:-1,:,:]

# divT[:,:,0,0] = (T[:,:,1,0,0]-T[:,:,0,0,0])/dx[:,:,0] + (T[:,:,1,0,1]-T[:,:,0,0,1])/dy[:,:,0] + (T[:,:,1,0,2]-T[:,:,0,0,2])/dz[:,:,0]
# divT[:,0,:,1] = (T[:,1,:,1,0]-T[:,0,:,1,0])/dx[:,0,:] + (T[:,1,:,1,1]-T[:,0,:,1,1])/dy[:,0,:] + (T[:,1,:,1,2]-T[:,0,:,1,2])/dz[:,0,:]
# divT[0,:,:,2] = (T[1,:,:,2,0]-T[0,:,:,2,0])/dx[0,:,:] + (T[1,:,:,2,1]-T[0,:,:,2,1])/dy[0,:,:] + (T[1,:,:,2,2]-T[0,:,:,2,2])/dz[0,:,:]

# divT[:,:,-1,0] = (T[:,:,-1,0,0]-T[:,:,-2,0,0])/dx[:,:,-1] + (T[:,:,-1,0,1]-T[:,:,-2,0,1])/dy[:,:,-1] + (T[:,:,-1,0,2]-T[:,:,-2,0,2])/dz[:,:,-1]
# divT[:,-1,:,1] = (T[:,-1,:,1,0]-T[:,-2,:,1,0])/dx[:,-1,:] + (T[:,-1,:,1,1]-T[:,-2,:,1,1])/dy[:,-1,:] + (T[:,-1,:,1,2]-T[:,-2,:,1,2])/dz[:,-1,:]
# divT[-1,:,:,2] = (T[-1,:,:,2,0]-T[-2,:,:,2,0])/dx[-1,:,:] + (T[-1,:,:,2,1]-T[-2,:,:,2,1])/dy[-1,:,:] + (T[-1,:,:,2,2]-T[-2,:,:,2,2])/dz[-1,:,:]

# divT[:,:,:,0] = ((T[:,:,2:,0,0]-T[:,:,0:-2,0,0])/2) + ((T[:,2:,:,0,0]-T[:,0:-2,:,0,0])/2) + ((T[2:,:,:,0,0]-T[0:-2,:,:,0,0])/2)
# print("divT shape", divT.shape)

# divT[1:-1,1:-1,1:-1,0] = ((T[:,:,2:,0,0] - T[:,:,:-2,0,0])/2)*(inv_Jx[:,:,1:-1,0,0]) + ((T[:,2:,:,0,0] - T[:,:-2,:,0,0])/2)*(inv_Jx[:,1:-1,:,0,0]) + ((T[2:,:,:,0,0] - T[:-2,:,:,0,0])/2)*(inv_Jx[1:-1,:,:,0,0]) + \
#                   ((T[:,:,2:,0,1] - T[:,:,:-2,0,1])/2)*(inv_Jx[:,:,1:-1,0,1]) + ((T[:,2:,:,0,1] - T[:,:-2,:,0,1])/2)*(inv_Jx[:,1:-1,:,0,1]) + ((T[2:,:,:,0,1] - T[:-2,:,:,0,1])/2)*(inv_Jx[1:-1,:,:,0,1]) + \
#                   ((T[:,:,2:,0,2] - T[:,:,:-2,0,2])/2)*(inv_Jx[:,:,1:-1,0,2]) + ((T[:,2:,:,0,2] - T[:,:-2,:,0,2])/2)*(inv_Jx[:,1:-1,:,0,2]) + ((T[2:,:,:,0,2] - T[:-2,:,:,0,2])/2)*(inv_Jx[1:-1,:,:,0,2]) 


def compute_divT(T, inv_Jx):
    D1, D2, D3, _, _ = T.shape
    divT = cp.zeros((D1, D2, D3, 3), dtype=T.dtype)

    # Central differencing (interior points)
    dT_dzeta = cp.zeros_like(T)
    dT_dzeta[1:-1, :, :, :, :] += (T[2:, :, :, :, :] - T[:-2, :, :, :, :]) / 2  # dzeta3
    dT_dzeta[:, 1:-1, :, :, :] += (T[:, 2:, :, :, :] - T[:, :-2, :, :, :]) / 2  # dzeta2
    dT_dzeta[:, :, 1:-1, :, :] += (T[:, :, 2:, :, :] - T[:, :, :-2, :, :]) / 2  # dzeta1

    # Forward/backward differencing (boundaries)
    dT_dzeta[0, :, :, :, :] += (T[1, :, :, :, :] - T[0, :, :, :, :])  # fwd dzeta3
    dT_dzeta[-1, :, :, :, :] += (T[-1, :, :, :, :] - T[-2, :, :, :, :])  # bwd dzeta3

    dT_dzeta[:, 0, :, :, :] += (T[:, 1, :, :, :] - T[:, 0, :, :, :])  # fwd dzeta2
    dT_dzeta[:, -1, :, :, :] += (T[:, -1, :, :, :] - T[:, -2, :, :, :])  # bwd dzeta2

    dT_dzeta[:, :, 0, :, :] += (T[:, :, 1, :, :] - T[:, :, 0, :, :])  # fwd dzeta1
    dT_dzeta[:, :, -1, :, :] += (T[:, :, -1, :, :] - T[:, :, -2, :, :])  # bwd dzeta1

    # Einstein summation: divT_i = ∂T_ij/∂ζ_k * ∂ζ_k/∂x_j
    divT = cp.einsum('...ij,...kj->...ik', dT_dzeta, inv_Jx).sum(axis=-1)
    # dT_dzeta[..., i, j], inv_Jx[..., k, j] → contraction over j
    # for i in range(3):
    #     for k in range(3):
    #         divT[..., i] += dT_dzeta[..., i, :] * inv_Jx[..., k, :]

    # # Final contraction: sum over k
    # divT = divT.sum(axis=-1)

    return divT

divT = compute_divT(T, inv_Jx)

force_per_unit_volume = -divT # Quasi-static assumption (inertia negligible) my justification is that mass is very small but idk a can be very big, even with high accelerations
forces = force_per_unit_volume.reshape(-1,3)

# force_magnitudes = np.linalg.norm(forces, axis=1)
# nonzero = force_magnitudes > 1e-8
# unit_forces = np.zeros_like(forces)
# unit_forces= forces / force_magnitudes[:, None]
# unit_forces = unit_forces.reshape(10,480,848,3) 

eps = 1e-4
force_magnitudes = cp.linalg.norm(forces, axis=1)
mask = (force_magnitudes < 1e7)#(force_magnitudes > 100) & 
correct = force_magnitudes.get()[mask.get()]
print(forces.shape,correct.shape)
plt.figure(figsize=(6, 4))
plt.hist(correct, bins=1000, color='steelblue', edgecolor='black')
plt.xlabel("Distance to fitted plane (m)")
plt.ylabel("Number of points")
# plt.xlim([-0.020,0.010])
# plt.ylim([0,1400])
plt.title("Histogram of distances")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)
    # plt.savefig(f"hist/{count:04d}.png")
    # plt.close()
safe_force_magnitudes = np.where(force_magnitudes < eps, eps, force_magnitudes)
unit_forces = forces / safe_force_magnitudes[:, None]
unit_forces = unit_forces.reshape(layers, sh1//div,sh2//div, 3) * 0.001

#physical_forces = forces_per_unit_volume * dV

cnan=cp.isinf(force_magnitudes).sum()
cinf=cp.isnan(force_magnitudes).sum()
print("unit forces", unit_forces.shape)
print(cnan,cinf)

undef_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
undef_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_undef_points[:,:,:,:].reshape(-1,3)))

def_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
def_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_def_points[:,:,:,:].reshape(-1,3)))

fpuv_x = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_y = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_z = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))

draw_lines_lineset(volumetric_def_points[1:,:,:,:].reshape(-1,3),volumetric_def_points[:-1,:,:,:].reshape(-1,3),fpuv_x)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,1],fpuv_y)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,2],fpuv_z)

F_magnitudes = cp.linalg.norm(F, axis=(-2, -1))
F_x_mag = cp.linalg.norm(F[:,:,:-1, 0], axis=-1).reshape(-1,1) #.flatten()
F_y_mag = cp.linalg.norm(F[:,:-1,:, 1], axis=-1).reshape(-1,1) #.flatten()
F_z_mag = cp.linalg.norm(F[:-1,:,:, 2], axis=-1).reshape(-1,1) #.flatten()
print("F_x_mag shape", F_x_mag.shape)
print("F_y_mag shape", F_y_mag.shape)
print("F_z_mag shape", F_z_mag.shape)
F_colors = cp.vstack([F_x_mag,F_y_mag,F_z_mag])
print("F_colors shape", F_colors.shape)
mask_x = F_x_mag < 20
mask_y = F_y_mag < 20
mask_z = F_z_mag < 20
plt.figure(figsize=(6, 4))
plt.hist(F_x_mag.get()[mask_x.get()], bins=1000, color='steelblue', edgecolor='black')
plt.hist(F_y_mag.get()[mask_y.get()], bins=1000, color='green', edgecolor='black')
plt.hist(F_z_mag.get()[mask_z.get()], bins=1000, color='red', edgecolor='black')
plt.xlabel("Distance to fitted plane (m)")
plt.ylabel("Number of points")
# plt.xlim([-0.020,0.010])
# plt.ylim([0,1400])
plt.title("Histogram of distances")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)

mesh = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
mesh_start_p = cp.vstack([volumetric_def_points[:,:,:-1,:].reshape(-1,3),volumetric_def_points[:,:-1,:,:].reshape(-1,3),volumetric_def_points[:-1,:,:,:].reshape(-1,3)])
print(mesh_start_p.shape)
mesh_end_p_x = volumetric_def_points[:,:,:-1,:] + (volumetric_def_points[:,:,1:,:] - volumetric_def_points[:,:,:-1,:]) #cp.hstack((dx_dzeta1.reshape(-1,1),dy_dzeta1.reshape(-1,1),dz_dzeta1.reshape(-1,1)))
print(mesh_end_p_x.shape)
mesh_end_p_y = volumetric_def_points[:,:-1,:,:] + (volumetric_def_points[:,1:,:,:] - volumetric_def_points[:,:-1,:,:]) #cp.hstack((dx_dzeta2.reshape(-1,1),dy_dzeta2.reshape(-1,1),dz_dzeta2.reshape(-1,1)))
mesh_end_p_z = volumetric_def_points[:-1,:,:,:] + (volumetric_def_points[1:,:,:,:] - volumetric_def_points[:-1,:,:,:]) #cp.hstack((dx_dzeta3.reshape(-1,1),dy_dzeta3.reshape(-1,1),dz_dzeta3.reshape(-1,1)))
mesh_end_p = cp.vstack([mesh_end_p_x.reshape(-1,3),mesh_end_p_y.reshape(-1,3),mesh_end_p_z.reshape(-1,3)])
draw_lines_lineset_color(mesh_start_p,mesh_end_p,mesh,F_colors)
o3d.visualization.draw([def_vol.to_legacy(),mesh.cpu().to_legacy()])
#print(force_per_unit_volume[3,220:230,450:460,:])
#print(unit_forces[3,220:230,450:460,:])
#print(divT[3,220:230,450:460,:])
#print(detF[3,220:230,450:460])
#print(F[3,220:230,450:460,:,:])

#o3d.visualization.draw_geometries([fpuv_x.cpu().to_legacy()])
#o3d.visualization.draw([def_vol,fpuv_x.cpu(),mesh.cpu()])

#loop with 3Dviewer
#use knn and calc dx = F dX using affine approx?

