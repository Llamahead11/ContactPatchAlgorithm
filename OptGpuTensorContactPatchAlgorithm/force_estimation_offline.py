import open3d as o3d
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import median_filter
from scipy.spatial import Delaunay 
import shapely
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, MultiPolygon
import alphashape
from scipy.spatial import Voronoi
from scipy.io import savemat


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

def normalised_colors_press(pressure):
    low_val = (-2.3)*1e6
    high_val = (-1.5)*1e6
    normalized =  (pressure - pressure[pressure > low_val].min()) / (pressure[pressure < high_val].max() - pressure[pressure > low_val].min())
    print(pressure[pressure > low_val].min(),pressure[pressure < high_val].max())
    colormap = plt.cm.get_cmap('viridis')
    colors = colormap(normalized.get())[:, :3]
    return colors
    
def normalised_colors_vec(force):
    val = 2
    normalized =  (force - force[force > -val].min()) / (force[force < val].max() - force[force > -val].min())
    print(force[force > -val].min(),force[force < val].max())
    colormap = plt.cm.get_cmap('viridis')
    colors = colormap(normalized.get())[:, :3]
    return colors

def normalised_colors(E):
    trE = cp.trace(E, axis1=3, axis2=4)[..., None, None]
    print("Trace E shape", trE.shape)
    E_dev = E - (trE/3) * cp.eye(3, dtype=E.dtype)[None, None, None, :, :]
    print("E_dev shape", E_dev.shape)
    #E_dev_norm = cp.linalg.norm(E_dev, axis=(3,4)).flatten()
    E_dev_norm = E_dev[:,:,:,2,2].flatten()
    #E_norm = np.linalg.norm(E, axis=(-2,-1)).flatten()
    #E_dev_norm = E_norm
    #normalized =  (E_dev_norm - E_dev_norm.min()) / (E_dev_norm.max() - E_dev_norm.min())
    #print(E_dev_norm.min(),E_dev_norm.max())

    maskE = cp.linalg.norm(E_dev, axis=(3,4)).flatten() < 100

    # plt.figure(figsize=(12, 6))
    # plt.subplot(2,3,1)
    # plt.hist(E_dev[:,:,:,0,0].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_xx")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_xx")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(2,3,2)
    # plt.hist(E_dev[:,:,:,1,1].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_yy")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_yy")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(2,3,3)
    # plt.hist(E_dev[:,:,:,2,2].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_zz")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_zz")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(2,3,4)
    # plt.hist(E_dev[:,:,:,0,1].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_xy")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_xy")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(2,3,5)
    # plt.hist(E_dev[:,:,:,0,2].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_xz")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_xz")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(2,3,6)
    # plt.hist(E_dev[:,:,:,1,2].flatten().get()[maskE.get()], bins=1000, color='steelblue', edgecolor='black')
    # plt.xlabel("E_yz")
    # plt.ylabel("Number of points")
    # plt.title("Histogram of Local E_yz")
    # plt.grid(True)
    # plt.tight_layout()
    #plt.show(block = False)

    normalized =  (E_dev_norm - E_dev_norm[E_dev_norm > -0.5].min()) / (E_dev_norm[E_dev_norm <0.5].max() - E_dev_norm[E_dev_norm > -0.5].min())
    print(E_dev_norm[E_dev_norm > -0.5].min(),E_dev_norm[E_dev_norm < 0.5].max())
    colormap = plt.cm.get_cmap('viridis')
    colors = colormap(normalized.get())[:, :3]
    return colors

def generate_orthog_vects(norms):
    eps = 1e-10
    # Create a reference vector based on the norm check
    norms_gpu = cp.linalg.norm(norms,axis=-1,keepdims=True)
    reference_vector_gpu = cp.where(norms_gpu < 0.9, cp.array([0, 1, 0], dtype=cp.float32), 
                                    cp.array([1, 0, 0], dtype=cp.float32))
    print("REF",reference_vector_gpu.shape)
    # Cross product to get the x axis of the local coordinate system
    x_axis_gpu = cp.cross(norms, reference_vector_gpu)
    x_axis_gpu /= cp.linalg.norm(x_axis_gpu, axis=-1, keepdims=True) + eps

    # Cross product again to get the y axis of the local coordinate system
    y_axis_gpu = cp.cross(norms, x_axis_gpu)
    y_axis_gpu /= cp.linalg.norm(y_axis_gpu, axis=-1, keepdims=True) + eps

    # Rotation matrix is constructed by stacking the axes
    rotation_matrix_gpu = cp.stack((x_axis_gpu, y_axis_gpu, norms), axis=-1)
    print("ROT",rotation_matrix_gpu.shape)
    #print(rotation_matrix_gpu)
    # #Use `einsum` to apply the rotation to point displacements on GPU
    # local_point_displacements_gpu = cp.einsum('...ji,...j->...i', rotation_matrix_gpu, point_disp_wrt_cam_gpu)
    return rotation_matrix_gpu


def triangle_area(p1, p2, p3):
    # p1, p2, p3 are 3D points as (x,y,z) arrays
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

def clipped_voronoi_areas(vor, boundary):
    areas = np.zeros(len(vor.points))
    for i, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if len(vertices) == 0 or -1 in vertices:
            # fallback: assign a small area (or approximate from nearest finite vertices)
            # or clip manually using the boundary's bounding box
            # simplest: approximate with small value to keep array shape
            areas[i] = 0.0
            continue

        poly = Polygon(vor.vertices[vertices])
        clipped = poly.intersection(boundary)
        
        if clipped.is_empty:
            areas[i] = 0.0
        elif isinstance(clipped, MultiPolygon):
            # sum areas of all disconnected pieces
            areas[i] = sum(p.area for p in clipped.geoms)
        else:
            areas[i] = clipped.area
    return areas

def find_contact_points(def_pcd_last_layer,A,B,C,D,o3d_inv_full_T,centroid):
    #centroid = def_pcd_last_layer.get_center()
    
    pcd_centre = def_pcd_last_layer.clone()
    normal_def_pcd_last_layer = def_pcd_last_layer.clone()
    normal_def_pcd_last_layer.transform(o3d_inv_full_T.inv())
    pcd_centre.transform(o3d_inv_full_T.inv())
    pcd_centre.translate(-centroid)
    #o3d.visualization.draw([pcd_centre])
    plane = o3d.t.geometry.TriangleMesh(device = o3d.core.Device("CUDA:0"))
    if pcd_centre.point.positions.shape[0] != 0:
        # U,S,VT = pcd_centre.point.positions.svd()
        # #print(VT)
        # #print(VT.shape)
        # normal = VT[-1]
        # print(normal)
        # print(centroid)
        # A, B, C = normal
        # print("A:",A,"B:",B,"C:",C)
        # D = (-normal.mul(centroid)).sum(dim=0)
        #print("D:",D)
        normal = (A.append(B).append(C))
        # print(normal)
        # #normal_transform = normal.matmul(o3d_inv_full_T[:3, :3].T)
        # print(o3d_inv_full_T)
        # normal_transform = (o3d_inv_full_T[:3, :3]).matmul(normal)
        # print(normal_transform)
        # print((o3d_inv_full_T[:3, :3]).matmul(normal))
        # print(o3d_inv_full_T[:3, -1])
        # A,B,C = normal_transform[:3]
        # D = (-normal_transform[:3].mul(centroid)).sum()
        #A*x+B*y+C*z+D = 0
        #print("A:",A,"B:",B,"C:",C,"D:",D)
        x = o3d.core.Tensor([[-0.5,0.5,-0.5,0.5]],dtype=o3d.core.float64).cuda()
        z = o3d.core.Tensor([[-0.5,-0.5,0.5,0.5]],dtype=o3d.core.float64).cuda()
        y = - (A*x + C*z + D) / B
        #print((x.append(y,axis = 0)).append(z,axis = 0).T())
        
        plane.vertex.positions = (x.append(y,axis = 0)).append(z,axis = 0).T()
        plane.triangle.indices = o3d.core.Tensor([[0,1,2],[1,2,3]],dtype=o3d.core.int64).cuda()
        
        dist_to_plane = ((normal_def_pcd_last_layer.point.positions).matmul(normal) + D).flatten()
        print(dist_to_plane.shape)
        mask = dist_to_plane.abs() < 0.0021

        contact_patch = normal_def_pcd_last_layer.select_by_mask(mask)
        # contact_patch.point.positions = contact_patch.point.positions - (dist_to_plane[mask].reshape((-1, 1))).mul(normal.reshape((1, 3)))
        n = normal.cpu().numpy()
        z_axis = np.array([0,0,1], dtype=np.float32)

        v = np.cross(n, z_axis)
        c = np.dot(n, z_axis)
        if np.linalg.norm(v) < 1e-8:  # already aligned
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1/(1+c))
        #contact_patch.point.positions = contact_patch.point.positions - (dist_to_plane[mask].reshape((-1, 1))).mul(normal.reshape((1, 3)))
        contact_patch.rotate(o3d.core.Tensor(R),center=[0,0,0])
        
        lugs = contact_patch.clone().cpu()
        # num_partitions = lugs.pca_partition(max_points=contact_patch.point.positions.shape[0]//2)
        # print(np.unique(lugs.point.partition_ids.numpy(), return_counts=True))
        # print(num_partitions)
        # lug0 = lugs.select_by_mask(lugs.point.partition_ids == 0)
        # lug1 = lugs.select_by_mask(lugs.point.partition_ids == 1)
        # lug2 = lugs.select_by_mask(lugs.point.partition_ids == 2)
        # o3d.visualization.draw([contact_patch.cpu()])
        labels = lugs.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)

        num_lugs, counts = np.unique(labels.numpy(), return_counts=True)
        print("Number of lugs",num_lugs)
        num_lugs = num_lugs[num_lugs != -1]
        
        #print(labels)
        max_label = labels.max().item()
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(
                labels.numpy() / (max_label if max_label > 0 else 1))
        colors = o3d.core.Tensor(colors[:, :3], o3d.core.float32)
        colors[labels < 0] = 0
        lugs.point.colors = colors
        #o3d.visualization.draw([lugs])
        # lug0 = lugs.select_by_mask(labels == 0)
        # lug1 = lugs.select_by_mask(labels == 1)
        # lug2 = lugs.select_by_mask(labels == 2)
        # lug3 = lugs.select_by_mask(labels == 3)
        lug_list = []
        alpha_list = []
        vor_list = []
        boundary_list = []
        dA_list = []
        for lug in num_lugs:
            lug_pcd = lugs.select_by_mask(labels == lug)
            if lug_pcd.point.positions.shape[0] > 4:
                lug_list.append(lug_pcd)
                poly_alpha = alphashape.alphashape(lug_pcd.point.positions.numpy()[:,:2], alpha=0.08)
                alpha_list.append(poly_alpha)
                vor = Voronoi(lug_pcd.point.positions.numpy()[:,:2])
                vor_list.append(vor)
                boundary_polygon = Polygon(poly_alpha)
                boundary_list.append(boundary_polygon)
                dA = clipped_voronoi_areas(vor, boundary_polygon)
                dA_list.append(dA)
            else:
                num_lugs = num_lugs[num_lugs != lug]
                lug_list.append(0)
                alpha_list.append(0)
                vor_list.append(0)
                dA_list.append(0)
        print(dA_list)

        #dA_list = np.array(dA_list)
        cpa = np.sum([dA.sum() for dA in dA_list])
        # poly0_alpha = alphashape.alphashape(lug0.point.positions.numpy()[:,:2], alpha=0.08)
        # poly1_alpha = alphashape.alphashape(lug1.point.positions.numpy()[:,:2], alpha=0.08)
        # poly2_alpha = alphashape.alphashape(lug2.point.positions.numpy()[:,:2], alpha=0.08)
        

        # vor0 = Voronoi(lug0.point.positions.numpy()[:,:2])
        # vor1 = Voronoi(lug1.point.positions.numpy()[:,:2])
        # vor2 = Voronoi(lug2.point.positions.numpy()[:,:2])
        
        # boundary_polygon0 = Polygon(poly0_alpha)
        # boundary_polygon1 = Polygon(poly1_alpha)
        # boundary_polygon2 = Polygon(poly2_alpha)
        
        # dA0 = clipped_voronoi_areas(vor0, boundary_polygon0)
        # dA1 = clipped_voronoi_areas(vor1, boundary_polygon1)
        # dA2 = clipped_voronoi_areas(vor2, boundary_polygon2)

        # mesh_lug0_alpha = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(lug0.to_legacy(),alpha=0.08))
        # mesh_lug1_alpha = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(lug1.to_legacy(),alpha=0.08))
        # mesh_lug2_alpha = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(lug2.to_legacy(),alpha=0.08))

        # mesh_lug0 = mesh_lug0_alpha #.boolean_intersection(mesh_lug0_del)
        # mesh_lug1 = mesh_lug1_alpha #.boolean_intersection(mesh_lug1_del)
        # mesh_lug2 = mesh_lug2_alpha #.boolean_intersection(mesh_lug2_del)

        # lug0_area = mesh_lug0.get_surface_area()
        # lug1_area = mesh_lug1.get_surface_area()
        # lug2_area = mesh_lug2.get_surface_area()

        # mesh_lug0.compute_triangle_areas()
        # mesh_lug1.compute_triangle_areas()
        # mesh_lug2.compute_triangle_areas()

        # print(mesh_lug0.triangle.areas.shape)
        # print(mesh_lug1.triangle.areas.shape)
        # print(mesh_lug2.triangle.areas.shape)
        
        # dA3 = -1
        # # mesh_lug3_del = o3d.t.geometry.TriangleMesh()
        # if lug3.point.positions.shape[0] != 0:
        #     poly3_alpha = alphashape.alphashape(lug3.point.positions.numpy()[:,:2], alpha=0.08)
        #     vor3 = Voronoi(lug3.point.positions.numpy()[:,:2])
        #     boundary_polygon3 = Polygon(poly3_alpha)
        #     dA3 = clipped_voronoi_areas(vor3, boundary_polygon3)
        #     # tri_lug3 = Delaunay(lug3.point.positions.numpy())
        #     # mesh_lug3_del.vertex.positions = lug3.point.positions
        #     # mesh_lug3_del.triangle.indices = o3d.core.Tensor(tri_lug3.simplices,dtype=o3d.core.int64)
        #     mesh_lug3_alpha = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(lug3.to_legacy(),alpha=0.08))
        #     mesh_lug3 = mesh_lug3_alpha #.boolean_intersection(mesh_lug3_del)

        #     lug3_area = mesh_lug3.get_surface_area()
        #     mesh_lug3.compute_triangle_areas()
        #     print(mesh_lug3.triangle.areas.shape)
        #     print("dA shape", dA0.shape, dA1.shape, dA2.shape, dA3.shape)
        #     print("Total dA",mesh_lug0.triangle.areas.shape,mesh_lug1.triangle.areas.shape,mesh_lug2.triangle.areas.shape,mesh_lug3.triangle.areas.shape)
        #     cpa = lug0_area + lug1_area + lug2_area + lug3_area
        # else:
        #     lug3_area = 0
        #     print("Total dA",mesh_lug0.triangle.areas.shape,mesh_lug1.triangle.areas.shape,mesh_lug2.triangle.areas.shape)
        #     cpa = lug0_area + lug1_area + lug2_area

        print("The contact patch area is: ", cpa)

        # o3d.visualization.draw([lug0,lug1,lug2, lug3,mesh_lug0,mesh_lug1,mesh_lug2, mesh_lug3, lugs, contact_patch.cpu()])

        contact_patch_points_indices = mask.nonzero(as_tuple=False)    

        return contact_patch, contact_patch_points_indices, plane, cpa, dA_list, num_lugs, labels
    
def triangle_indices():
    triangles = []
    H = 480
    W = 848
    for i in range(H - 1):
        for j in range(W - 1):
            idx = i * W + j
            v0 = idx
            v1 = idx + 1
            v2 = idx + W
            v3 = idx + W + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)
    return triangles

def force_preload(first_layer):
    pressure = 2 * 100000
    # first_layer_mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CUDA:0"))
    # first_layer_mesh.vertex.positions = first_layer.point.positions
    # first_layer_mesh.vertex.normals = first_layer.point.normals
    # first_layer_mesh.normalize_normals()
    # first_layer_mesh.triangle.indices = o3d.core.Tensor(triangle_indices, o3d.core.int32).cuda()
    # first_layer_mesh.compute_triangle_areas()
    first_layer_mesh_legacy = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        first_layer.cpu().to_legacy(), alpha=0.08
    )
    first_layer_mesh = o3d.t.geometry.TriangleMesh.from_legacy(first_layer_mesh_legacy).cuda()
    first_layer_mesh.compute_triangle_areas()
    dA = first_layer_mesh.triangle.areas.cpu().numpy()
    normals = first_layer_mesh.vertex.normals.cpu().numpy()
    triangle_indices = first_layer_mesh.triangle.indices.cpu().numpy()
    vertex_area = np.zeros((normals.shape[0],))
    #print(triangle_indices.min(), triangle_indices.max(), first_layer.point.positions.shape[0])
    # for tri, area in zip(triangle_indices, dA):
    #     vertex_area[tri] += area / 3.0
    tri_flat = triangle_indices.flatten()  # shape: (M*3,)

    # Repeat the corresponding triangle area 3 times and divide by 3
    area_flat = np.repeat(dA / 3.0, 3)    # shape: (M*3,)

    # Accumulate per vertex
    np.add.at(vertex_area, tri_flat, area_flat)
    # poly_alpha = alphashape.alphashape(first_layer.point.positions.numpy(), alpha=0.08)
    # vor = Voronoi(first_layer.point.positions.numpy())
    # boundary_polygon = Polygon(poly_alpha)
    # dA = clipped_voronoi_areas(vor, boundary_polygon)
    Force_preload_array = pressure * normals * vertex_area[:,None]
    force_mask = (Force_preload_array < (35)).all(axis = 1) & (Force_preload_array > (-35)).all(axis = 1)
    Force_preload = np.sum(Force_preload_array[force_mask], axis=0)
    print(Force_preload)
    print(0)
    return Force_preload



# Current Config (Deformed Configuration)
# linspace of origins to outer_def
contact_patch_area_arr = []
traction_arr = []
net_force_est_arr = []
tri_ind = triangle_indices()
try:
    for i in range(50,250+250):
        if (i == 0) | (i == 1):
            continue
        else:
            data = np.load(f"test_7/iteration_{i:03d}.npz")
            print("Iteration", i)
            origins_uh = cp.asarray(data['orig']) #.reshape(480,848,3)
            hit_point_uh = cp.asarray(data['hit_p']) #.reshape(480,848,3)
            hit_point_o_uh = cp.asarray(data['hit_p_o']) #.reshape(480,848,3)
            inv_full_T = cp.asarray(data['inv_T'])
            o3d_inv_full_T = o3d.core.Tensor(data['inv_T']).cuda()
            centroid = o3d.core.Tensor(data['cent'],dtype=o3d.core.float64).cuda()
            A = o3d.core.Tensor(data['Ap'],dtype=o3d.core.float64).cuda()
            B = o3d.core.Tensor(data['Bp'],dtype=o3d.core.float64).cuda()
            C = o3d.core.Tensor(data['Cp'],dtype=o3d.core.float64).cuda()
            D = o3d.core.Tensor(data['Dp'],dtype=o3d.core.float64).cuda()

            origins_h = cp.hstack([origins_uh, cp.ones((origins_uh.shape[0], 1))])
            hit_point_h = cp.hstack([hit_point_uh, cp.ones((hit_point_uh.shape[0], 1))])  # (N, 4)
            hit_point_o_h = cp.hstack([hit_point_o_uh, cp.ones((hit_point_o_uh.shape[0], 1))])  # (N, 4)

            # Apply transformation
            transformed_origins_h = origins_h @ inv_full_T.T  # still (N, 4)
            transformed_hit_point_h = hit_point_h @ inv_full_T.T  # still (N, 4)
            transformed_hit_point_o_h = hit_point_o_h @ inv_full_T.T  # still (N, 4)

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
            # plt.figure()
            # plt.imshow(image_inner)
            # plt.figure()
            # plt.imshow(image_outer)

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

            #o3d.visualization.draw([pcd1.cpu(),pcd2.cpu(),pcd3.cpu(),pcd4.cpu()])

            # plt.figure()
            # plt.imshow(image_inner_f)
            # plt.figure()
            # plt.imshow(image_outer_f)

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

            div = 2#4
            layers = 15#20
            # origins = origins[60:380:div,312:740:div,:].reshape(-1,3)
            # hit_point = hit_point[60:380:div,312:740:div,:].reshape(-1,3)
            # hit_point_o = hit_point_o[60:380:div,312:740:div,:].reshape(-1,3)

            origins = origins[::div,::div,:].reshape(-1,3)
            hit_point = hit_point[::div,::div,:].reshape(-1,3)
            hit_point_o = hit_point_o[::div,::div,:].reshape(-1,3)

            # sh1 = 380 - 60
            # sh2 = 740 - 312

            sh1 = 480
            sh2 = 848

            print("origins.shape", origins.shape)
            print("hit_point.shape", hit_point.shape)
            print("hit_point_o.shape", hit_point_o.shape)

            dist = hit_point - origins
            outer_def = hit_point_o - dist

            #perturb = arr = (cp.random.rand(136960, 3) * 2 - 1) * 0.0005

            #volumetric_def_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
            volumetric_def_points = cp.linspace(origins,outer_def,layers,True,False,cp.float64,axis=0)
            print("volumetric_def_points shape", volumetric_def_points.shape)

            # Reference Config (Undeformed Configuration)
            #linspace of hit_point to hit_point_o
            volumetric_undef_points = cp.linspace(hit_point,hit_point_o,layers,True,False,cp.float32,axis=0)
            print("volumetric_def_points shape", volumetric_undef_points.shape)

            dir = hit_point - hit_point_o
            undef_surface_normals = dir/cp.linalg.norm(dir,axis=-1, keepdims=True)
            undef_vol_surface_normals = cp.tile(undef_surface_normals[None, ...], (layers, 1, 1))

            dir_def = dist - outer_def
            def_surface_normals = dir_def/cp.linalg.norm(dir_def,axis=-1, keepdims=True)
            def_vol_surface_normals = cp.tile(def_surface_normals[None, ...], (layers, 1, 1))

            volumetric_def_points = volumetric_def_points.reshape(layers,sh1//div,sh2//div,3)
            volumetric_undef_points = volumetric_undef_points.reshape(layers,sh1//div,sh2//div,3)
            undef_vol_surface_normals = undef_vol_surface_normals.reshape(layers,sh1//div,sh2//div,3)
            def_vol_surface_normals = def_vol_surface_normals.reshape(layers,sh1//div,sh2//div,3)

            undef_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            undef_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_undef_points.reshape(-1,3)))
            undef_vol_s = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            undef_vol_s.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_undef_points[1:3,14000:14800,:].reshape(-1,3)))

            def_vol = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            def_vol.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_def_points.reshape(-1,3)))
            def_vol_first_layer = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
            def_vol_first_layer.point.positions = o3d.core.Tensor.from_dlpack(volumetric_def_points[1,:,:].reshape(-1,3).toDlpack())
            def_vol_first_layer.point.normals = o3d.core.Tensor.from_dlpack(def_vol_surface_normals[1,:,:].reshape(-1,3).toDlpack())
            def_vol_last_layer = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
            def_vol_last_layer.point.positions = o3d.core.Tensor.from_dlpack(volumetric_def_points[-1,:,:].reshape(-1,3).toDlpack())
            def_vol_last_layer.point.normals = o3d.core.Tensor.from_dlpack(def_vol_surface_normals[-1,:,:].reshape(-1,3).toDlpack())
            def_vol_s = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            def_vol_s.point.positions = o3d.core.Tensor.from_numpy(cp.asnumpy(volumetric_def_points[1:3,14000:14800,:].reshape(-1,3)))

            contact_patch_pcd, contact_patch_points_indices, plane, contact_patch_area, dA_list, num_lugs,labels = find_contact_points(def_vol_last_layer,A,B,C,D,o3d_inv_full_T,centroid)
            #force_pre = force_preload(def_vol_first_layer)
            contact_patch_area_arr.append(contact_patch_area)
            #o3d.visualization.draw([contact_patch_pcd.transform(o3d_inv_full_T).cpu(),plane.transform(o3d_inv_full_T).cpu(),def_vol_last_layer.cpu()])
            #print("indices:",(def_vol_surface_normals[-1,:,:].flatten())[cp.from_dlpack(contact_patch_points_indices[0].to_dlpack())])
            # exit()
            # o3d.visualization.draw([undef_vol,def_vol,undef_vol_s,def_vol_s])
            # exit()

            #Deformation Gradient F using Finite Differences Method
            volumetric_def_points = volumetric_def_points.reshape(layers,sh1//div,sh2//div,3)
            volumetric_undef_points = volumetric_undef_points.reshape(layers,sh1//div,sh2//div,3)
            undef_vol_surface_normals = undef_vol_surface_normals.reshape(layers,sh1//div,sh2//div,3)

            t1 = cp.array([1,0,0], dtype=cp.float32)
            t1_field = cp.broadcast_to(t1, (layers,sh1//div,sh2//div, 3))

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

            # point_wise_spatial_volumes = cp.zeros((layers,sh1//div,sh2//div),cp.float32)

            # #dV = (left/2+right/2)*(top/2+bottom/2)*(front/2+back/2)
            # dX = volumetric_undef_points[:,:,:-1,0] - volumetric_undef_points[:,:,1:,0]
            # dY = volumetric_undef_points[:,:-1,:,1] - volumetric_undef_points[:,1:,:,1]
            # dZ = volumetric_undef_points[:-1,:,:,2] - volumetric_undef_points[1:,:,:,2]

            # left = dX[:,:,:-1]/2
            # right = dX[:,:,1:]/2
            # front = dY[:,:-1,:]/2
            # back = dY[:,1:,:]/2
            # top = dZ[:-1,:,:]/2
            # bottom = dZ[1:,:,:]/2

            # point_wise_spatial_volumes[1:-1,1:-1,1:-1] = (left[1:-1,1:-1,:]+right[1:-1,1:-1,:])*(top[:,1:-1,1:-1]+bottom[:,1:-1,1:-1])*(front[1:-1,:,1:-1]+back[1:-1,:,1:-1])
            # point_wise_spatial_volumes[:,:,0] = left[:,:,0]*2
            # point_wise_spatial_volumes[:,:,-1] = right[:,:,-1]*2
            # point_wise_spatial_volumes[:,0,:] = front[:,0,:]*2
            # point_wise_spatial_volumes[:,-1,:] = back[:,-1,:]*2
            # point_wise_spatial_volumes[0,:,:] = top[0,:,:]*2
            # point_wise_spatial_volumes[-1,:,:] = bottom[-1,:,:]*2

            # Jx shape: (..., 3, 3)
            Jx = cp.stack([dx_dzeta, dy_dzeta, dz_dzeta], axis=-2)
            inv_Jx = cp.linalg.inv(Jx)
            print("inv_Jx shape", inv_Jx.shape)

            dX_dzeta = cp.stack([dX_dzeta1, dX_dzeta2, dX_dzeta3], axis=-1)
            dY_dzeta = cp.stack([dY_dzeta1, dY_dzeta2, dY_dzeta3], axis=-1)
            dZ_dzeta = cp.stack([dZ_dzeta1, dZ_dzeta2, dZ_dzeta3], axis=-1)

            JX = cp.stack([dX_dzeta, dY_dzeta, dZ_dzeta], axis=-2)
            det_JX = cp.linalg.det(JX)
            print("Min determinant of JX:", det_JX.min().item())
            maskdetJX = det_JX < 1e-6

            # plt.figure(figsize=(6, 4))
            # plt.hist(cp.abs(det_JX).get()[maskdetJX.get()], bins=1000, color='steelblue', edgecolor='black')
            # plt.xlabel("Volume [m^3]")
            # plt.ylabel("Number of points")
            # # plt.xlim([-0.020,0.010])
            # # plt.ylim([0,1400])
            # plt.title("Histogram of Determinant of JX or Volume of Elements [m^3]")
            # plt.grid(True)
            # plt.tight_layout()
            #plt.show(block = False)

            #F = Jx @ cp.linalg.inv(JX)
            F = cp.linalg.solve(JX, Jx)
            print("F shape", F.shape)
            I = cp.eye(3, dtype=cp.float32)
            diff = cp.abs(F - I)
            print("Mean deviation from identity in undeformed case:", cp.mean(diff))

            # cond_JX = cp.linalg.cond(JX)
            # print("Max condition number:", cond_JX.max().item())
            # print("Mean condition number:", cond_JX.mean().item())
            F_reshaped = F[:,:,:,:,:].reshape(-1, 3, 3)              
            detF = cp.linalg.det(F_reshaped)              
            #detF = detF.reshape(F.shape[:-2])
            #detF = cp.clip(detF, 1e-5, 1e5)
            print("detF shape", detF.shape)


            mask_F = (detF < 10) & (detF > -10)
            # plt.figure(figsize=(6, 4))
            # plt.hist(detF.get()[mask_F.get()], bins=1000, color='steelblue', edgecolor='black')
            # plt.xlabel("Histogram of Determinant F Deformation Gradient")
            # plt.ylabel("Number of points")
            # # plt.xlim([-0.020,0.010])
            # # plt.ylim([0,1400])
            # plt.title("Det(F)")
            # plt.grid(True)
            # plt.tight_layout()
            #plt.show(block = False)

            o3d_detF = o3d.core.Tensor.from_numpy(detF.get()) 
            o3d_bad_mask = (o3d_detF > 1.5) | (o3d_detF < 0.5)  
            o3d_good_mask = o3d_bad_mask != True

            bad_pcd = undef_vol.select_by_mask(o3d_bad_mask)
            bad_def_pcd = def_vol.select_by_mask(o3d_bad_mask)
            print("Number of invalid points",bad_pcd.point.positions.shape)
            #o3d.visualization.draw([bad_pcd, bad_def_pcd])

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

            Q = generate_orthog_vects(undef_vol_surface_normals)

            #E_local = Q.transpose(0, 1, 2, 4, 3) @ E @ Q
            E_local = cp.einsum("...ji,...jk,...kl->...il", Q, E, Q)
            print("E_local",E_local.shape)

            trE = cp.trace(E_local, axis1=-2, axis2=-1)
            masktrE = trE < 100

            # plt.figure(figsize=(6, 4))
            # plt.hist(trE.get()[masktrE.get()], bins=1000, color='steelblue', edgecolor='black')
            # plt.xlabel("Trace of E Green-Lagrange Strain")
            # plt.ylabel("Number of points")
            # # plt.xlim([-0.020,0.010])
            # # plt.ylim([0,1400])
            # plt.title("tr(E)")
            # plt.grid(True)
            # plt.tight_layout()
            #plt.show(block = False)

            c = normalised_colors(E_local)
            undef_vol.point.colors = o3d.core.Tensor.from_numpy(c)
            clean_undef_vol = undef_vol.select_by_mask(o3d_good_mask)
            #o3d.visualization.draw([clean_undef_vol])


            #==================================================================================================
            #Yeoh model
            #U = C10(I1 − 3) + C20(I1 − 3)2 + C30(I1 − 3)3 + 1/D1(J^el− 3)2 +1/D2(J^el − 3)4 +1/D3(J^el − 3)6

            # C10 = 473.685
            # C20 = -119.853
            # C30 = 34.293

            # D1 = 5.085*(10^(-8))

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

            # dWdx = 2*C10 + 8*C20*trE + 24*C30*(trE)**2
            # dxdE = cp.eye(3)[None,None,None,:,:] 

            # S = dWdx[:,:,:,None,None] * dxdE
            # print("S shape", S.shape)

            #=====================================================================================

            #=====================================================================================
            #Mooney Rivlin Model with axial and shear contributions (I1,I2)

            C10 = 23666        #0.6e6
            C01 = 178697
            D = 1

            # W = C10(I1-3) + C01(I2-3), I1 = tr(C) , I2 = 0.5(tr(C)^2 - tr(C^2)) with C = 2E + I

            # I1 - 3 = 2*tr(E) : I1 = tr(C) :
            #                    I1 = tr(2E + I) : 
            #                    I1 = tr(2E) + tr(I) : 
            #                    I1 = 2*tr(E) + 3

            # I2 - 3 = 4tr(E) + 2*(tr(E)^2 - tr(E^2)) : I2 = 0.5(tr(C)^2 - tr(C^2)) :
            #                                           I2 = 0.5((2*tr(E)+3)^2) - tr((2E+I)^2)) :
            #                                           2*I2 = (4*tr(E)^2 + 12*tr(E) + 9) - tr(4E^2 + 4E + I))
            #                                           2*I2 = (4*tr(E)^2 + 12*tr(E) + 9) - (4*tr(E^2) + 4*tr(E) + 3)
            #                                           2*I2 = 4*(tr(E)^2 - tr(E^2)) + 8*tr(E) + 6
            #                                           I2 = 2*(tr(E)^2 - tr(E^2)) + 4*tr(E) + 3
            #                                           I2 - 3 = 4*tr(E) + 2*(tr(E)^2 - tr(E^2)) 

            # Now, W = C10*2*tr(E) + C01*(4*tr(E) + 2*(tr(E)^2 - tr(E^2))) 

            # But dtr(E)/dE = I : d/dE (E11+E22+E33) = 1 if i = j and 0 if i != j

            # But dtr(E^2)/dE = d/dE (E11*E11+E12*E21+E13*E31+E21*E12+E22*E22+E23*E32+E31*E13+E32*E23+E33*E33) and E is symmetric :
            #                 = d/dE (E11^2 + E12^2 + E13^2 + E21^2 + E22^2 + E23^2 + E31^2 + E32^2 + E33^2)
            #                 = d/dE (Eij ^2) 
            #                 = 2*Eij
            #                 = 2E

            # Therefore, S = dW/dE = C10*2*tr(E)/dE + C01*(4*tr(E)/dE + 2*((tr(E)/dE)^2 - tr(E^2)/dE)) 
            #                      = C10*2*I + C01*(4*I + 2*(2*tr(E)*I - 2E)) 
            #                      = (C10*2 + C01*4 + 4*C01*tr(E))*I - 4*C01*E


            S = (C10*2 + C01*4 + 4*C01*trE[...,None,None])*cp.eye(3)[None,None,None,:,:] - 4*C01*E
            print("S shape", S.shape)

            #T = (1/detF)*F_T*S*F
            detF = detF.reshape(F.shape[:-2])
            FS = cp.matmul(F,S)
            T = cp.matmul(FS,F_T)
            T /= detF[:,:,:,None,None]
            print("T shape", T.shape)

            # traction = T[-1,:,:]*def_vol_surface_normals[-1,:,:]
            contact_patch_points_indices_cp = cp.from_dlpack(contact_patch_points_indices[0].to_dlpack())
            traction = cp.einsum('...ij,...j->...i', T[-1, :, :].reshape(-1,3,3)[contact_patch_points_indices_cp], def_vol_surface_normals[-1, :, :].reshape(-1,3)[contact_patch_points_indices_cp])
            mean_t = traction.mean(axis=0)
        
            mask_traction = (traction < (5)*1e6).all(axis = 1) & (traction > (-5)*1e6).all(axis = 1)
            print("Number of traction points",traction[mask_traction].shape)

            net_force_est_lug = []
            mask_force_lug = []
            #dA_list = np.array(dA_list)
            print(num_lugs)
            for lug in num_lugs:
                #print(dA_list[lug].shape)
                #print(traction.get()[labels.numpy() == lug,:].shape)
                net_force_est_lug_i = (traction.get()[labels.numpy() == lug,:] * dA_list[lug][:,None])
                #print(net_force_est_lug_i)
                net_force_est_lug.append(net_force_est_lug_i)
                mask_force_lug_i = (net_force_est_lug_i < (35)).all(axis = 1) & (net_force_est_lug_i > (-35)).all(axis = 1)
                mask_force_lug.append(mask_force_lug_i)

            force_est = np.concatenate([
                net_force_est_lug[i][mask_force_lug[i]]
                for i in num_lugs
            ])

            net_force_est = np.array([
                net_force_est_lug[i][mask_force_lug[i]].sum(axis=0)
                for i in num_lugs
            ]).sum(axis=0) #- force_pre
            # net_force_est_lug0 = (traction.get()[labels.numpy() == 0,:] * ml0[:,None])
            # net_force_est_lug1 = (traction.get()[labels.numpy() == 1,:] * ml1[:,None])
            # net_force_est_lug2 = (traction.get()[labels.numpy() == 2,:] * ml2[:,None])

            # mask_force_lug0 = (net_force_est_lug0 < (35)).all(axis = 1) & (net_force_est_lug0 > (-35)).all(axis = 1)
            # mask_force_lug1 = (net_force_est_lug1 < (35)).all(axis = 1) & (net_force_est_lug1 > (-35)).all(axis = 1)
            # mask_force_lug2 = (net_force_est_lug2 < (35)).all(axis = 1) & (net_force_est_lug2 > (-35)).all(axis = 1)

            # if np.sum(labels.numpy() == 3) != 0:
            #     net_force_est_lug3 = (traction.get()[labels.numpy() == 3,:] * ml3[:,None])
            #     mask_force_lug3 = (net_force_est_lug3 < (35)).all(axis = 1) & (net_force_est_lug3 > (-35)).all(axis = 1)
            #     force_est = np.concatenate([net_force_est_lug0[mask_force_lug0],net_force_est_lug1[mask_force_lug1],net_force_est_lug2[mask_force_lug2],net_force_est_lug3[mask_force_lug3]])
            # else:
            #     force_est = np.concatenate([net_force_est_lug0[mask_force_lug0],net_force_est_lug1[mask_force_lug1],net_force_est_lug2[mask_force_lug2]])
            
            # force_est = np.concatenate([net_force_est_lug0[mask_force_lug0],net_force_est_lug1[mask_force_lug1],net_force_est_lug2[mask_force_lug2],net_force_est_lug3[mask_force_lug3]])
            #plt.figure(figsize=(6, 4))
            # plt.hist(force_est[:,0], bins=100, color='blue')
            # plt.hist(force_est[:,1], bins=100, color='red')
            # plt.hist(force_est[:,2], bins=100, color='yellow')
            # plt.xlabel("Contact Patch Force [N]")
            # plt.ylabel("Number of points")
            # plt.title("Force [N]")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

            #net_force_est = net_force_est_lug0[mask_force_lug0].sum(axis=0) + net_force_est_lug1[mask_force_lug1].sum(axis=0) + net_force_est_lug2[mask_force_lug2].sum(axis=0) + net_force_est_lug3[mask_force_lug3].sum(axis=0)
            #[labels.numpy()[mask_traction.get()] == 0,:] #mean_t * contact_patch_area
            print("net force estimation",net_force_est)
            traction_arr.append(traction[mask_traction].sum(axis=0).get())
            # plt.figure(figsize=(6, 4))
            # plt.hist(traction[:,0].get()[mask_traction.get()], bins=1000, color='blue')
            # plt.hist(traction[:,1].get()[mask_traction.get()], bins=1000, color='red')
            # plt.hist(traction[:,2].get()[mask_traction.get()], bins=1000, color='yellow')
            # plt.xlabel("Traction [N/m^2]")
            # plt.ylabel("Number of points")
            # # plt.xlim([-0.020,0.010])
            # # plt.ylim([0,1400])
            # plt.title("Histogram of Traction [N/m^2]")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

            net_force_est_arr.append(net_force_est)

            # pressure = -traction*def_vol_surface_normals[-1,:,:] 
            pressure = -(traction * def_vol_surface_normals[-1, :, :].reshape(-1,3)[contact_patch_points_indices_cp]).sum(axis=-1)
            mask_pressure = (pressure < (-1.5)*1e6) & (pressure > (-2.3)*1e6)
            # cpres = normalised_colors_press(pressure)
            # contact_patch_pcd.point.colors = o3d.core.Tensor.from_numpy(cpres).cuda() 

            # shear = traction+pressure*def_vol_surface_normals
            shear = traction+pressure[:, None]*def_vol_surface_normals[-1,:,:].reshape(-1,3)[contact_patch_points_indices_cp]

            # plt.figure(figsize=(6, 4))
            # plt.hist(pressure.get()[mask_pressure.get()], bins=1000, color='steelblue', edgecolor='black')
            # plt.xlabel("Histogram of Pressure Distribution [Pa]")
            # plt.ylabel("Number of points")
            # # plt.xlim([-0.020,0.010])
            # # plt.ylim([0,1400])
            # plt.title("Pa")
            # plt.grid(True)
            # plt.tight_layout()
            #plt.show(block = False)

            print("before")
            #o3d.visualization.draw([contact_patch_pcd.cpu()])
except Exception as e:
    import traceback
    traceback.print_exc()
    print("CUDA ERROR:", e)
finally:
    contact_patch_area_arr = np.array(contact_patch_area_arr)
    print("Mean cpa: ", contact_patch_area_arr.mean())
    plt.figure()
    plt.plot(contact_patch_area_arr)
    plt.title("Contact Patch Area [m^2]")
    plt.xlabel("Time [s]")
    plt.ylabel("Area [m^2]")
    plt.grid()
    plt.show(block=False)

    # np.save('cpa_test_7.npy', contact_patch_area_arr)
    #savemat("cpa_test_21.mat", {"data4": contact_patch_area_arr})
    # force_contact_patch = sum(traction*area)
    # force_contact_patch = traction*
    traction_arr = np.array(traction_arr)
    plt.figure()
    plt.plot(traction_arr[:,0])
    plt.plot(traction_arr[:,1])
    plt.plot(traction_arr[:,2])
    plt.legend(['x','y','z'])
    plt.title("Contact Patch Traction [N/m^2]")
    plt.xlabel("Time [s]")
    plt.ylabel("Traction [N/m^2]")
    plt.grid()
    plt.show(block=False)

    net_force_est_arr = np.array(net_force_est_arr)
    plt.figure()
    plt.plot(net_force_est_arr[:,0])
    plt.plot(net_force_est_arr[:,1])
    plt.plot(net_force_est_arr[:,2])
    plt.legend(['x','y','z'])
    plt.title("net_force [N]")
    plt.xlabel("Time [s]")
    plt.ylabel("net_force [N]")
    plt.grid()
    plt.show(block=True)

exit(0)

#========================
#Mooney Rivlin Material model parameterisation 
#C10,C01,D

#take RMSE from 5000N and 7000N vertical plate tests + RSME from 6000N and 4000N Cleat tests
# 
# use scipy . opt . min to get the C10, C01 and D
#  

#========================



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

forces_puv = force_per_unit_volume.reshape(-1,3)

forces = force_per_unit_volume*det_JX[...,None] 
forces_local = forces #cp.einsum("...ji,...j->...i", Q, forces)
f_x_mask = cp.abs(forces_local[...,0]) < 200
f_y_mask = cp.abs(forces_local[...,1]) < 200
f_z_mask = cp.abs(forces_local[...,2]) < 200

good_points = (detF < 1.5) & (detF > 0.5)
for i in range(0,20,1):
    print(f"Fx sum {i/1}", forces_local[...,0][(cp.abs(forces_local[...,0]) < i/1) & good_points].sum())
    print(f"Fy sum {i/1}", forces_local[...,1][(cp.abs(forces_local[...,1]) < i/1) & good_points].sum())
    print(f"Fz sum {i/1}", forces_local[...,2][(cp.abs(forces_local[...,2]) < i/1) & good_points].sum())

plt.figure(figsize=(6, 4))
plt.hist(forces_local[...,0].get()[f_x_mask.get()], bins=1000, color='steelblue', edgecolor='black')
plt.hist(forces_local[...,1].get()[f_y_mask.get()], bins=1000, color='green', edgecolor='black')
plt.hist(forces_local[...,2].get()[f_z_mask.get()], bins=1000, color='red', edgecolor='black')
plt.xlabel("F_x, F_y, F_z [N]")
plt.ylabel("Number of points")
# plt.xlim([-0.020,0.010])
# plt.ylim([0,1400])
plt.title("F_x, F_y , F_z Force Histogram")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)

cf = normalised_colors_vec(forces[...,2].flatten())
def_vol.point.colors = o3d.core.Tensor.from_numpy(cf)
clean_def_vol_forces = def_vol.select_by_mask(o3d_good_mask)
#o3d.visualization.draw([clean_def_vol_forces])

# force_magnitudes = np.linalg.norm(forces, axis=1)
# nonzero = force_magnitudes > 1e-8
# unit_forces = np.zeros_like(forces)
# unit_forces= forces / force_magnitudes[:, None]
# unit_forces = unit_forces.reshape(10,480,848,3) 

eps = 1e-4
force_magnitudes = cp.linalg.norm(forces_puv, axis=1)
mask = (force_magnitudes < 1e9)#(force_magnitudes > 100) & 
correct = force_magnitudes.get()[mask.get()]
print(forces_puv.shape,correct.shape)
plt.figure(figsize=(6, 4))
plt.hist(correct, bins=1000, color='steelblue', edgecolor='black')
plt.xlabel("Force Density Magnitudes [N/m^3]")
plt.ylabel("Number of points")
# plt.xlim([-0.020,0.010])
# plt.ylim([0,1400])
plt.title("Histogram of Force Density Magnitudes (N/m^3)")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)
    # plt.savefig(f"hist/{count:04d}.png")
    # plt.close()
safe_force_magnitudes = np.where(force_magnitudes < eps, eps, force_magnitudes)
unit_forces = forces_puv / safe_force_magnitudes[:, None]
unit_forces = unit_forces.reshape(layers, sh1//div,sh2//div, 3) * 0.001

#physical_forces = forces_per_unit_volume * dV

cnan=cp.isinf(force_magnitudes).sum()
cinf=cp.isnan(force_magnitudes).sum()
print("unit forces", unit_forces.shape)
print(cnan,cinf)

fpuv_x = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_y = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
fpuv_z = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))

draw_lines_lineset(volumetric_def_points[:,:,:,:].reshape(-1,3)[::100],volumetric_def_points[:,:,:,:].reshape(-1,3)[::100] + Q[...,0].reshape(-1,3)[::100]/100,fpuv_x)
draw_lines_lineset(volumetric_def_points[:,:,:,:].reshape(-1,3)[::100],volumetric_def_points[:,:,:,:].reshape(-1,3)[::100] + Q[...,1].reshape(-1,3)[::100]/100,fpuv_y)
draw_lines_lineset(volumetric_def_points[:,:,:,:].reshape(-1,3)[::100],volumetric_def_points[:,:,:,:].reshape(-1,3)[::100] + Q[...,2].reshape(-1,3)[::100]/100,fpuv_z)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,1],fpuv_y)
# draw_lines_lineset(volumetric_undef_points,volumetric_undef_points+force_per_unit_volume[...,2],fpuv_z)
o3d.visualization.draw([fpuv_x.cpu(),fpuv_y.cpu()])
o3d.visualization.draw([fpuv_z.cpu()])

# F_magnitudes = cp.linalg.norm(F, axis=(-2, -1))
# F_x_mag = cp.linalg.norm(F[:,:,:-1, 0], axis=-1).reshape(-1,1) #.flatten()
# F_y_mag = cp.linalg.norm(F[:,:-1,:, 1], axis=-1).reshape(-1,1) #.flatten()
# F_z_mag = cp.linalg.norm(F[:-1,:,:, 2], axis=-1).reshape(-1,1) #.flatten()
# print("F_x_mag shape", F_x_mag.shape)
# print("F_y_mag shape", F_y_mag.shape)
# print("F_z_mag shape", F_z_mag.shape)
# F_colors = cp.vstack([F_x_mag,F_y_mag,F_z_mag])
# print("F_colors shape", F_colors.shape)
# mask_x = F_x_mag < 20
# mask_y = F_y_mag < 20
# mask_z = F_z_mag < 20
# plt.figure(figsize=(6, 4))
# plt.hist(F_x_mag.get()[mask_x.get()], bins=1000, color='steelblue', edgecolor='black')
# plt.hist(F_y_mag.get()[mask_y.get()], bins=1000, color='green', edgecolor='black')
# plt.hist(F_z_mag.get()[mask_z.get()], bins=1000, color='red', edgecolor='black')
# plt.xlabel("Global F Deformation Gradients Magnitudes")
# plt.ylabel("Number of points")
# # plt.xlim([-0.020,0.010])
# # plt.ylim([0,1400])
# plt.title("F_x, F_y , F_z Deformation Gradients Magnitudes")
# plt.grid(True)
# plt.tight_layout()
# plt.show(block = False)

# mesh = o3d.t.geometry.LineSet(o3d.core.Device("CUDA:0"))
# mesh_start_p = cp.vstack([volumetric_def_points[:,:,:-1,:].reshape(-1,3),volumetric_def_points[:,:-1,:,:].reshape(-1,3),volumetric_def_points[:-1,:,:,:].reshape(-1,3)])
# print(mesh_start_p.shape)
# mesh_end_p_x = volumetric_def_points[:,:,:-1,:] + (volumetric_def_points[:,:,1:,:] - volumetric_def_points[:,:,:-1,:]) #cp.hstack((dx_dzeta1.reshape(-1,1),dy_dzeta1.reshape(-1,1),dz_dzeta1.reshape(-1,1)))
# print(mesh_end_p_x.shape)
# mesh_end_p_y = volumetric_def_points[:,:-1,:,:] + (volumetric_def_points[:,1:,:,:] - volumetric_def_points[:,:-1,:,:]) #cp.hstack((dx_dzeta2.reshape(-1,1),dy_dzeta2.reshape(-1,1),dz_dzeta2.reshape(-1,1)))
# mesh_end_p_z = volumetric_def_points[:-1,:,:,:] + (volumetric_def_points[1:,:,:,:] - volumetric_def_points[:-1,:,:,:]) #cp.hstack((dx_dzeta3.reshape(-1,1),dy_dzeta3.reshape(-1,1),dz_dzeta3.reshape(-1,1)))
# mesh_end_p = cp.vstack([mesh_end_p_x.reshape(-1,3),mesh_end_p_y.reshape(-1,3),mesh_end_p_z.reshape(-1,3)])
# draw_lines_lineset_color(mesh_start_p,mesh_end_p,mesh,F_colors)
# o3d.visualization.draw([def_vol.to_legacy(),mesh.cpu().to_legacy()])
#print(force_per_unit_volume[3,220:230,450:460,:])
#print(unit_forces[3,220:230,450:460,:])
#print(divT[3,220:230,450:460,:])
#print(detF[3,220:230,450:460])
#print(F[3,220:230,450:460,:,:])

#o3d.visualization.draw_geometries([fpuv_x.cpu().to_legacy()])
#o3d.visualization.draw([def_vol,fpuv_x.cpu(),mesh.cpu()])

#loop with 3Dviewer
#use knn and calc dx = F dX using affine approx?

