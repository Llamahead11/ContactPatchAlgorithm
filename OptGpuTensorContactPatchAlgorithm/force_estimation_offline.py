import open3d as o3d
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def draw_lines_lineset(start_points, end_points, line_set):
    line_start = start_points.copy()#cp.ascontiguousarray(cp.from_dlpack(start_points.to_dlpack()))
    line_end = end_points.copy()#cp.ascontiguousarray(cp.from_dlpack(end_points.to_dlpack()))

    # dist = cp.linalg.norm(line_end-line_start,axis=1)
    # # lines = [[i, i + len(start_points)] for i in range(len(start_points))]
    # # line_points = np.vstack((start_points, end_points))
    # # valid_start = ((line_start[:,2] <= 1) & (line_start[:,2] >= 0.07)) 
    # # valid_end = ((line_end[:,2] <= 1) & (line_end[:,2] >= 0.07))
    # valid_dist = dist < 0.1
    # mask = (line_start != 0.0).all(axis=1) & (line_end != 0.0).all(axis=1) & valid_dist
    
    # Replace invalid start points with corresponding end points
    line_valid_start = line_start #[mask]
    line_valid_end = line_end #[mask] 
    
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


# Current Config (Deformed Configuration)
# linspace of origins to outer_def

data = np.load("saved_arrays/iteration_160.npz")
origins = cp.asarray(data['orig']).reshape(480,848,3)
hit_point = cp.asarray(data['hit_p']).reshape(480,848,3)
hit_point_o = cp.asarray(data['hit_p_o']).reshape(480,848,3)


div = 4
layers = 20
origins = origins[::div,::div,:].reshape(-1,3)
hit_point = hit_point[::div,::div,:].reshape(-1,3)
hit_point_o = hit_point_o[::div,::div,:].reshape(-1,3)

print("origins.shape", origins.shape)
print("hit_point.shape", hit_point.shape)
print("hit_point_o.shape", hit_point_o.shape)

dist = hit_point - origins
outer_def = hit_point_o - dist

volumetric_def_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
#volumetric_def_points = cp.linspace(origins,outer_def,layers,True,False,cp.float64,axis=0)
print("volumetric_def_points shape", volumetric_def_points.shape)

# Reference Config (Undeformed Configuration)
#linspace of hit_point to hit_point_o
volumetric_undef_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
print("volumetric_def_points shape", volumetric_undef_points.shape)

#Deformation Gradient F using Finite Differences Method
volumetric_def_points = volumetric_def_points.reshape(layers,480//div,848//div,3)
volumetric_undef_points = volumetric_undef_points.reshape(layers,480//div,848//div,3)

dx = cp.empty((layers,480//div,848//div),cp.float32)
dy = cp.empty((layers,480//div,848//div),cp.float32)
dz = cp.empty((layers,480//div,848//div),cp.float32)
dX = cp.empty((layers,480//div,848//div),cp.float32)
dY = cp.empty((layers,480//div,848//div),cp.float32)
dZ = cp.empty((layers,480//div,848//div),cp.float32)

##Foward Difference
# dx[:,:,:-1] = volumetric_def_points[:,:,1:,0] - volumetric_def_points[:,:,0:-1,0]
# dy[:,:-1,:] = volumetric_def_points[:,1:,:,1] - volumetric_def_points[:,0:-1,:,1]
# dz[:-1,:,:] = volumetric_def_points[1:,:,:,2] - volumetric_def_points[0:-1,:,:,2]

# dX[:,:,:-1] = volumetric_undef_points[:,:,1:,0] - volumetric_undef_points[:,:,0:-1,0]
# dY[:,:-1,:] = volumetric_undef_points[:,1:,:,1] - volumetric_undef_points[:,0:-1,:,1]
# dZ[:-1,:,:] = volumetric_undef_points[1:,:,:,2] - volumetric_undef_points[0:-1,:,:,2]

## Central Difference
dx[:,:,1:-1] = (volumetric_def_points[:,:,2:,0] - volumetric_def_points[:,:,0:-2,0])/2
dy[:,1:-1,:] = (volumetric_def_points[:,2:,:,1] - volumetric_def_points[:,0:-2,:,1])/2
dz[1:-1,:,:] = (volumetric_def_points[2:,:,:,2] - volumetric_def_points[0:-2,:,:,2])/2

dX[:,:,1:-1] = (volumetric_undef_points[:,:,2:,0] - volumetric_undef_points[:,:,0:-2,0])/2
dY[:,1:-1,:] = (volumetric_undef_points[:,2:,:,1] - volumetric_undef_points[:,0:-2,:,1])/2
dZ[1:-1,:,:] = (volumetric_undef_points[2:,:,:,2] - volumetric_undef_points[0:-2,:,:,2])/2

dx[:,:,0] = volumetric_def_points[:,:,1,0] - volumetric_def_points[:,:,0,0]
dy[:,0,:] = volumetric_def_points[:,1,:,1] - volumetric_def_points[:,0,:,1]
dz[0,:,:] = volumetric_def_points[1,:,:,2] - volumetric_def_points[0,:,:,2]

dX[:,:,0] = volumetric_undef_points[:,:,1,0] - volumetric_undef_points[:,:,0,0]
dY[:,0,:] = volumetric_undef_points[:,1,:,1] - volumetric_undef_points[:,0,:,1]
dZ[0,:,:] = volumetric_undef_points[1,:,:,2] - volumetric_undef_points[0,:,:,2]

dx[:,:,-1] = volumetric_def_points[:,:,-1,0] - volumetric_def_points[:,:,-2,0]
dy[:,-1,:] = volumetric_def_points[:,-1,:,1] - volumetric_def_points[:,-2,:,1]
dz[-1,:,:] = volumetric_def_points[-1,:,:,2] - volumetric_def_points[-2,:,:,2]

dX[:,:,-1] = volumetric_undef_points[:,:,-1,0] - volumetric_undef_points[:,:,-2,0]
dY[:,-1,:] = volumetric_undef_points[:,-1,:,1] - volumetric_undef_points[:,-2,:,1]
dZ[-1,:,:] = volumetric_undef_points[-1,:,:,2] - volumetric_undef_points[-2,:,:,2]

eps = 1e-4

dX_safe = cp.where(cp.abs(dX) < eps, eps, dX)
dY_safe = cp.where(cp.abs(dY) < eps, eps, dY)
dZ_safe = cp.where(cp.abs(dZ) < eps, eps, dZ)

F = cp.stack([
    cp.stack([dx/dX_safe, dx/dY_safe, dx/dZ_safe], axis=-1),
    cp.stack([dy/dX_safe, dy/dY_safe, dy/dZ_safe], axis=-1),
    cp.stack([dz/dX_safe, dz/dY_safe, dz/dZ_safe], axis=-1),
    ], axis=-2)
print("F shape", F.shape) #(10,480,848,3,3)


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

divT = cp.empty((layers,480//div,848//div,3),cp.float32)

eps = 1e-4

dx_safe = cp.where(cp.abs(dx) < eps, eps, dx)
dy_safe = cp.where(cp.abs(dy) < eps, eps, dy)
dz_safe = cp.where(cp.abs(dz) < eps, eps, dz)
print("dx_safe stats", cp.min(dx_safe).item(), cp.max(dx_safe).item(), cp.mean(dx_safe).item())
plt.figure(figsize=(6, 4))
plt.hist(dx_safe.reshape(-1,1).get(), bins=500, color='steelblue', edgecolor='black')
plt.xlabel("Distance to fitted plane (m)")
plt.ylabel("Number of points")
# plt.xlim([-0.020,0.010])
# plt.ylim([0,1400])
plt.title("Histogram of distances")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)
#Central diff
divT[:,:,1:-1,0] = (T[:,:,2:,0,0]-T[:,:,0:-2,0,0])/dx_safe[:,:,1:-1]+ (T[:,:,2:,0,1]-T[:,:,0:-2,0,1])/dy_safe[:,:,1:-1] + (T[:,:,2:,0,2]-T[:,:,0:-2,0,2])/dz_safe[:,:,1:-1]
divT[:,1:-1,:,1] = (T[:,2:,:,1,0]-T[:,0:-2,:,1,0])/dx_safe[:,1:-1,:] + (T[:,2:,:,1,1]-T[:,0:-2,:,1,1])/dy_safe[:,1:-1,:] + (T[:,2:,:,1,2]-T[:,0:-2,:,1,2])/dz_safe[:,1:-1,:]
divT[1:-1,:,:,2] = (T[2:,:,:,2,0]-T[0:-2,:,:,2,0])/dx_safe[1:-1,:,:] + (T[2:,:,:,2,1]-T[0:-2,:,:,2,1])/dy_safe[1:-1,:,:] + (T[2:,:,:,2,2]-T[0:-2,:,:,2,2])/dz_safe[1:-1,:,:]

divT[:,:,0,0] = (T[:,:,1,0,0]-T[:,:,0,0,0])/dx_safe[:,:,0] + (T[:,:,1,0,1]-T[:,:,0,0,1])/dy_safe[:,:,0] + (T[:,:,1,0,2]-T[:,:,0,0,2])/dz_safe[:,:,0]
divT[:,0,:,1] = (T[:,1,:,1,0]-T[:,0,:,1,0])/dx_safe[:,0,:] + (T[:,1,:,1,1]-T[:,0,:,1,1])/dy_safe[:,0,:] + (T[:,1,:,1,2]-T[:,0,:,1,2])/dz_safe[:,0,:]
divT[0,:,:,2] = (T[1,:,:,2,0]-T[0,:,:,2,0])/dx_safe[0,:,:] + (T[1,:,:,2,1]-T[0,:,:,2,1])/dy_safe[0,:,:] + (T[1,:,:,2,2]-T[0,:,:,2,2])/dz_safe[0,:,:]

divT[:,:,-1,0] = (T[:,:,-1,0,0]-T[:,:,-2,0,0])/dx_safe[:,:,-1] + (T[:,:,-1,0,1]-T[:,:,-2,0,1])/dy_safe[:,:,-1] + (T[:,:,-1,0,2]-T[:,:,-2,0,2])/dz_safe[:,:,-1]
divT[:,-1,:,1] = (T[:,-1,:,1,0]-T[:,-2,:,1,0])/dx_safe[:,-1,:] + (T[:,-1,:,1,1]-T[:,-2,:,1,1])/dy_safe[:,-1,:] + (T[:,-1,:,1,2]-T[:,-2,:,1,2])/dz_safe[:,-1,:]
divT[-1,:,:,2] = (T[-1,:,:,2,0]-T[-2,:,:,2,0])/dx_safe[-1,:,:] + (T[-1,:,:,2,1]-T[-2,:,:,2,1])/dy_safe[-1,:,:] + (T[-1,:,:,2,2]-T[-2,:,:,2,2])/dz_safe[-1,:,:]

print("divT shape", divT.shape)

force_per_unit_volume = -divT # Quasi-static assumption (inertia negligible) my justification is that mass is very small but idk a can be very big
forces = force_per_unit_volume.reshape(-1,3)

# force_magnitudes = np.linalg.norm(forces, axis=1)
# nonzero = force_magnitudes > 1e-8
# unit_forces = np.zeros_like(forces)
# unit_forces= forces / force_magnitudes[:, None]
# unit_forces = unit_forces.reshape(10,480,848,3) 

eps = 1e-4
force_magnitudes = cp.linalg.norm(forces, axis=1)
mask = (force_magnitudes > 100) & (force_magnitudes < 1e20)
correct = force_magnitudes.get()[mask.get()]
print(forces.shape,correct.shape)
plt.figure(figsize=(6, 4))
plt.hist(correct, bins=500, color='steelblue', edgecolor='black')
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
unit_forces = unit_forces.reshape(layers, 480//div,848//div, 3) * 0.001

#physical_forces = forces_per_unit_volume * dV

cnan=cp.isinf(unit_forces).sum()
cinf=cp.isnan(unit_forces).sum()
print("unit forces", unit_forces.shape)
print(cnan,cinf)

undef_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
undef_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_undef_points[9,:,:,:].reshape(-1,3)))

def_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
def_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_def_points[9,:,:,:].reshape(-1,3)))

fpuv_x = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_y = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_z = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))

draw_lines_lineset(volumetric_def_points[:,:,:,:].reshape(-1,3),volumetric_def_points[:,:,:,:].reshape(-1,3)+unit_forces[:,:,:,:].reshape(-1,3),fpuv_x)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,1],fpuv_y)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,2],fpuv_z)

#print(force_per_unit_volume[3,220:230,450:460,:])
#print(unit_forces[3,220:230,450:460,:])
#print(divT[3,220:230,450:460,:])
#print(detF[3,220:230,450:460])
#print(F[3,220:230,450:460,:,:])

#o3d.visualization.draw_geometries([fpuv_x.cpu().to_legacy()])
o3d.visualization.draw([def_vol,fpuv_x.cpu()])


