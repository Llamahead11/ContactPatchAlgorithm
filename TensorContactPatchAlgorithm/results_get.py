import open3d as o3d
import numpy as np
from replay_realsense_tensor import read_RGB_D_folder
import time 
import cv2
import matplotlib.pyplot as plt

# imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
# while imageStream.has_next():
#     start_cv2 = cv2.getTickCount()
#     if imageStream.has_next():
#         count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu= imageStream.get_next_frame()
#     #t2cam_pcd.to_legacy()
#         o3d.visualization.draw([t2cam_pcd_cuda])
#     end_cv2 = cv2.getTickCount()
#     time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     print("FPS:", time_sec)

# with open('cross_section.npy', 'rb') as f:
#     inner_arr = np.load(f)
#     outer_arr = np.load(f)
inner_arr_un = np.load('inner_arr_un.npy', allow_pickle=True)
outer_arr_un = np.load('outer_arr_un.npy', allow_pickle=True)
inner_lat_un = np.load('inner_lat_un.npy', allow_pickle=True)
outer_lat_un = np.load('outer_lat_un.npy', allow_pickle=True)

mask_left1 = (inner_arr_un[:,:,2] < 0.10) & (inner_arr_un[:,:,1] > 0.2142) & (inner_arr_un[:,:,0] < 0.1)
mask_mid1 = (inner_arr_un[:,:,2] > 0.10) & (inner_arr_un[:,:,2] < 0.16) & (inner_arr_un[:,:,0] < 0.1)
mask_right1 = (inner_arr_un[:,:,2] > 0.16) & (inner_arr_un[:,:,0] < 0.1)

mask_left_o1 = (outer_arr_un[:,:,2] < 0.10) & (outer_arr_un[:,:,1] > 0.2142) & (outer_arr_un[:,:,0] < 0.1)
mask_mid_o1 = (outer_arr_un[:,:,2] > 0.10) & (outer_arr_un[:,:,2] < 0.16) & (outer_arr_un[:,:,0] < 0.1)
mask_right_o1 = (outer_arr_un[:,:,2] > 0.16) & (outer_arr_un[:,:,0] < 0.1)

mask_fr1 = (inner_lat_un[:,:,0] < -0.01) & (inner_lat_un[:,:,1] > 0.25)
mask_cen1 = (inner_lat_un[:,:,0] > -0.01) & (inner_lat_un[:,:,0] < 0.01) & (inner_lat_un[:,:,1] > 0.25)
mask_ba1 = (inner_lat_un[:,:,0] > 0.01) & (inner_lat_un[:,:,1] > 0.25)

mask_fr_o1 = (outer_lat_un[:,:,0] < -0.01) & (outer_lat_un[:,:,1] > 0.25)
mask_cen_o1 = (outer_lat_un[:,:,0] > -0.01) & (outer_lat_un[:,:,0] < 0.01) & (outer_lat_un[:,:,1] > 0.25)
mask_ba_o1 = (outer_lat_un[:,:,0] > 0.01) & (outer_lat_un[:,:,1] > 0.25)

inner_arr_loaded = np.load('inner_arr.npy', allow_pickle=True)
outer_arr_loaded = np.load('outer_arr.npy', allow_pickle=True)

print(inner_arr_un[mask_left1][0].shape)



# print(inner_arr_loaded[0].shape,inner_arr_loaded[100].shape,inner_arr_loaded[2].shape)

mask_left = (inner_arr_loaded[:,:,2] < 0.10) & (inner_arr_loaded[:,:,1] > 0.2142) & (inner_arr_loaded[:,:,0] < 0.1)
mask_mid = (inner_arr_loaded[:,:,2] > 0.10) & (inner_arr_loaded[:,:,2] < 0.16) & (inner_arr_loaded[:,:,0] < 0.1)
mask_right = (inner_arr_loaded[:,:,2] > 0.16) & (inner_arr_loaded[:,:,0] < 0.1)

mask_left_o = (outer_arr_loaded[:,:,2] < 0.10) & (outer_arr_loaded[:,:,1] > 0.2142) & (outer_arr_loaded[:,:,0] < 0.1)
mask_mid_o = (outer_arr_loaded[:,:,2] > 0.10) & (outer_arr_loaded[:,:,2] < 0.16) & (outer_arr_loaded[:,:,0] < 0.1)
mask_right_o = (outer_arr_loaded[:,:,2] > 0.16) & (outer_arr_loaded[:,:,0] < 0.1)

mask_x_outer = (outer_arr_loaded[:,:,0] < 0.1329) & (outer_arr_loaded[:,:,0] > -0.2267) 
# mask_x_outer_long_mid = (outer_arr_loaded[:,:,0] < 0.1329) & (outer_arr_loaded[:,:,0] > -0.2267) 
# mask_x_outer_long_right = (outer_arr_loaded[:,:,0] < 0.1329) & (outer_arr_loaded[:,:,0] > -0.2267) 

# plt.figure()
# #plt.scatter(inner_arr_loaded[0,:,0],inner_arr_loaded[0,:,1])
# plt.scatter(outer_arr_loaded[0,:,0],outer_arr_loaded[0,:,1])
# plt.axis('equal')


x_data_inner_long_left = inner_arr_loaded[:,:,0]
y_data_inner_long_left = inner_arr_loaded[:,:,1]

x_data_inner_long_left = [x_data_inner_long_left[i][mask_left[i]] for i in range(x_data_inner_long_left.shape[0])]
y_data_inner_long_left = [y_data_inner_long_left[i][mask_left[i]] for i in range(y_data_inner_long_left.shape[0])]

x_data_inner_long_mid = inner_arr_loaded[:,:,0]
y_data_inner_long_mid = inner_arr_loaded[:,:,1]

x_data_inner_long_mid = [x_data_inner_long_mid[i][mask_mid[i]] for i in range(x_data_inner_long_mid.shape[0])]
y_data_inner_long_mid = [y_data_inner_long_mid[i][mask_mid[i]] for i in range(y_data_inner_long_mid.shape[0])]

x_data_inner_long_right = inner_arr_loaded[:,:,0]
y_data_inner_long_right = inner_arr_loaded[:,:,1]

x_data_inner_long_right = [x_data_inner_long_right[i][mask_right[i]] for i in range(x_data_inner_long_right.shape[0])]
y_data_inner_long_right = [y_data_inner_long_right[i][mask_right[i]] for i in range(y_data_inner_long_right.shape[0])]

x_data_outer_long_left = outer_arr_loaded[:,:,0]
y_data_outer_long_left = outer_arr_loaded[:,:,1]

x_data_outer_long_left = [x_data_outer_long_left[i][mask_left_o[i] ] for i in range(x_data_outer_long_left.shape[0])]
y_data_outer_long_left = [y_data_outer_long_left[i][mask_left_o[i] ] for i in range(y_data_outer_long_left.shape[0])]

x_data_outer_long_mid = outer_arr_loaded[:,:,0]
y_data_outer_long_mid = outer_arr_loaded[:,:,1]

x_data_outer_long_mid = [x_data_outer_long_mid[i][mask_mid_o[i] ] for i in range(x_data_outer_long_mid.shape[0])]
y_data_outer_long_mid = [y_data_outer_long_mid[i][mask_mid_o[i] ] for i in range(y_data_outer_long_mid.shape[0])]

x_data_outer_long_right = outer_arr_loaded[:,:,0]
y_data_outer_long_right = outer_arr_loaded[:,:,1]

x_data_outer_long_right = [x_data_outer_long_right[i][mask_right_o[i] ] for i in range(x_data_outer_long_right.shape[0])]
y_data_outer_long_right = [y_data_outer_long_right[i][mask_right_o[i]] for i in range(y_data_outer_long_right.shape[0])]

print(len(x_data_outer_long_left))


# print(mask_x_outer.shape)
# x_data_outer = x_data_outer[mask_x_outer]
# y_data_outer = y_data_outer[mask_x_outer]
# print(x_data_outer.shape)
# mask_x_outer_long_left = (x_data_outer_long_left < 0.1329) & (x_data_outer_long_left > -0.2267) 
# x_data_outer_long_left = [x_data_outer_long_left[i][mask_x_outer_long_left[i]] for i in range(x_data_outer_long_left.shape[0])]
# y_data_outer_long_left = [y_data_outer_long_left[i][mask_x_outer_long_left[i]] for i in range(y_data_outer_long_left.shape[0])]

# mask_x_outer_long_mid = (x_data_outer_long_mid < 0.1329) & (x_data_outer_long_mid > -0.2267) 
# x_data_outer_long_mid = [x_data_outer_long_mid[i][mask_x_outer_long_mid[i]] for i in range(x_data_outer_long_mid.shape[0])]
# y_data_outer_long_mid = [y_data_outer_long_mid[i][mask_x_outer_long_mid[i]] for i in range(y_data_outer_long_mid.shape[0])]

# mask_x_outer_long_right = (x_data_outer_long_right < 0.1329) & (x_data_outer_long_right > -0.2267) 
# x_data_outer_long_right = [x_data_outer_long_right[i][mask_x_outer_long_right[i]] for i in range(x_data_outer_long_right.shape[0])]
# y_data_outer_long_right = [y_data_outer_long_right[i][mask_x_outer_long_right[i]] for i in range(y_data_outer_long_right.shape[0])]

plt.figure(figsize=(10, 6))
plt.scatter(x_data_outer_long_left[0], -y_data_outer_long_left[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(x_data_outer_long_left[5], -y_data_outer_long_left[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_outer_long_left[10], -y_data_outer_long_left[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_outer_long_left[15], -y_data_outer_long_left[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_outer_long_left[20], -y_data_outer_long_left[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(x_data_inner_long_left[0], -y_data_inner_long_left[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(x_data_inner_long_left[5], -y_data_inner_long_left[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_inner_long_left[10], -y_data_inner_long_left[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_inner_long_left[15], -y_data_inner_long_left[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_inner_long_left[20], -y_data_inner_long_left[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_arr_un[mask_left1,0],-inner_arr_un[mask_left1,1], s=1, c='black', marker='^')
plt.scatter(outer_arr_un[mask_left_o1,0],-outer_arr_un[mask_left_o1,1], s=1, c='black', marker='^')
plt.scatter(x_data_outer_long_left[0][0], -y_data_outer_long_left[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(x_data_outer_long_left[5][0], -y_data_outer_long_left[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(x_data_outer_long_left[10][0], -y_data_outer_long_left[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(x_data_outer_long_left[15][0], -y_data_outer_long_left[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(x_data_outer_long_left[20][0], -y_data_outer_long_left[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_arr_un[mask_left1,0][0],-inner_arr_un[mask_left1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_arr_un[mask_left_o1,0][0],-outer_arr_un[mask_left_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('X [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Longitudinal Cross section of Inner and Outer \n Corresponding Deformation at z = 0.09 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)
#plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.scatter(x_data_outer_long_mid[0], -y_data_outer_long_mid[0], s=0.5, c='blue', alpha=0.7) 
plt.scatter(x_data_outer_long_mid[5], -y_data_outer_long_mid[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_outer_long_mid[10], -y_data_outer_long_mid[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_outer_long_mid[15], -y_data_outer_long_mid[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_outer_long_mid[20], -y_data_outer_long_mid[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(x_data_inner_long_mid[0], -y_data_inner_long_mid[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(x_data_inner_long_mid[5], -y_data_inner_long_mid[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_inner_long_mid[10], -y_data_inner_long_mid[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_inner_long_mid[15], -y_data_inner_long_mid[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_inner_long_mid[20], -y_data_inner_long_mid[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_arr_un[mask_mid1,0],-inner_arr_un[mask_mid1,1], s=1, c='black', marker='^')
plt.scatter(outer_arr_un[mask_mid_o1,0],-outer_arr_un[mask_mid_o1,1], s=1, c='black', marker='^')
plt.scatter(x_data_outer_long_mid[0][0], -y_data_outer_long_mid[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(x_data_outer_long_mid[5][0], -y_data_outer_long_mid[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(x_data_outer_long_mid[10][0], -y_data_outer_long_mid[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(x_data_outer_long_mid[15][0], -y_data_outer_long_mid[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(x_data_outer_long_mid[20][0], -y_data_outer_long_mid[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_arr_un[mask_mid1,0][0],-inner_arr_un[mask_mid1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_arr_un[mask_mid_o1,0][0],-outer_arr_un[mask_mid_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('X [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Longitudinal Cross section of Inner and Outer \n Corresponding Deformation at z = 0.13 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)


plt.figure(figsize=(10, 6))
plt.scatter(x_data_outer_long_right[0], -y_data_outer_long_right[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(x_data_outer_long_right[5], -y_data_outer_long_right[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_outer_long_right[10], -y_data_outer_long_right[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_outer_long_right[15], -y_data_outer_long_right[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_outer_long_right[20], -y_data_outer_long_right[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(x_data_inner_long_right[0], -y_data_inner_long_right[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(x_data_inner_long_right[5], -y_data_inner_long_right[5], s=0.5, c='green', alpha=0.7)
plt.scatter(x_data_inner_long_right[10], -y_data_inner_long_right[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(x_data_inner_long_right[15], -y_data_inner_long_right[15], s=0.5, c='red', alpha=0.7)
plt.scatter(x_data_inner_long_right[20], -y_data_inner_long_right[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_arr_un[mask_right1,0],-inner_arr_un[mask_right1,1], s=1, c='black', marker='^')
plt.scatter(outer_arr_un[mask_right_o1,0],-outer_arr_un[mask_right_o1,1], s=1, c='black', marker='^')
plt.scatter(x_data_outer_long_right[0][0], -y_data_outer_long_right[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(x_data_outer_long_right[5][0], -y_data_outer_long_right[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(x_data_outer_long_right[10][0], -y_data_outer_long_right[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(x_data_outer_long_right[15][0], -y_data_outer_long_right[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(x_data_outer_long_right[20][0], -y_data_outer_long_right[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_arr_un[mask_right1,0][0],-inner_arr_un[mask_right1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_arr_un[mask_right_o1,0][0],-outer_arr_un[mask_right_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('X [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Longitudinal Cross section of Inner and Outer \n Corresponding Deformation at z = 0.17 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)

inner_lat_loaded = np.load('inner_lat.npy', allow_pickle=True)
outer_lat_loaded = np.load('outer_lat.npy', allow_pickle=True)

mask_fr = (inner_lat_loaded[:,:,0] < -0.01) & (inner_lat_loaded[:,:,1] > 0.25)
mask_cen = (inner_lat_loaded[:,:,0] > -0.01) & (inner_lat_loaded[:,:,0] < 0.01) & (inner_lat_loaded[:,:,1] > 0.25)
mask_ba = (inner_lat_loaded[:,:,0] > 0.01) & (inner_lat_loaded[:,:,1] > 0.25)

mask_fr_o = (outer_lat_loaded[:,:,0] < -0.01) & (outer_lat_loaded[:,:,1] > 0.25)
mask_cen_o = (outer_lat_loaded[:,:,0] > -0.01) & (outer_lat_loaded[:,:,0] < 0.01) & (outer_lat_loaded[:,:,1] > 0.25)
mask_ba_o = (outer_lat_loaded[:,:,0] > 0.01) & (outer_lat_loaded[:,:,1] > 0.25)

y_data_inner_lat_fr = inner_lat_loaded[:,:,1]
z_data_inner_lat_fr = inner_lat_loaded[:,:,2]

y_data_inner_lat_fr = [y_data_inner_lat_fr[i][mask_fr[i]] for i in range(y_data_inner_lat_fr.shape[0])]
z_data_inner_lat_fr = [z_data_inner_lat_fr[i][mask_fr[i]] for i in range(z_data_inner_lat_fr.shape[0])]

y_data_inner_lat_cen = inner_lat_loaded[:,:,1]
z_data_inner_lat_cen = inner_lat_loaded[:,:,2]

y_data_inner_lat_cen = [y_data_inner_lat_cen[i][mask_cen[i]] for i in range(y_data_inner_lat_cen.shape[0])]
z_data_inner_lat_cen = [z_data_inner_lat_cen[i][mask_cen[i]] for i in range(z_data_inner_lat_cen.shape[0])]

y_data_inner_lat_ba = inner_lat_loaded[:,:,1]
z_data_inner_lat_ba = inner_lat_loaded[:,:,2]

y_data_inner_lat_ba = [y_data_inner_lat_ba[i][mask_ba[i]] for i in range(y_data_inner_lat_ba.shape[0])]
z_data_inner_lat_ba = [z_data_inner_lat_ba[i][mask_ba[i]] for i in range(z_data_inner_lat_ba.shape[0])]

y_data_outer_lat_fr = outer_lat_loaded[:,:,1]
z_data_outer_lat_fr = outer_lat_loaded[:,:,2]

y_data_outer_lat_fr = [y_data_outer_lat_fr[i][mask_fr_o[i] ] for i in range(y_data_outer_lat_fr.shape[0])]
z_data_outer_lat_fr = [z_data_outer_lat_fr[i][mask_fr_o[i] ] for i in range(z_data_outer_lat_fr.shape[0])]

y_data_outer_lat_cen = outer_lat_loaded[:,:,1]
z_data_outer_lat_cen = outer_lat_loaded[:,:,2]

y_data_outer_lat_cen = [y_data_outer_lat_cen[i][mask_cen_o[i] ] for i in range(y_data_outer_lat_cen.shape[0])]
z_data_outer_lat_cen = [z_data_outer_lat_cen[i][mask_cen_o[i] ] for i in range(z_data_outer_lat_cen.shape[0])]

y_data_outer_lat_ba = outer_lat_loaded[:,:,1]
z_data_outer_lat_ba = outer_lat_loaded[:,:,2]

y_data_outer_lat_ba = [y_data_outer_lat_ba[i][mask_ba_o[i] ] for i in range(y_data_outer_lat_ba.shape[0])]
z_data_outer_lat_ba = [z_data_outer_lat_ba[i][mask_ba_o[i] ] for i in range(z_data_outer_lat_ba.shape[0])]


plt.figure(figsize=(10, 6))
plt.scatter(z_data_outer_lat_fr[0], -y_data_outer_lat_fr[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_outer_lat_fr[5], -y_data_outer_lat_fr[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_outer_lat_fr[10], -y_data_outer_lat_fr[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_outer_lat_fr[15], -y_data_outer_lat_fr[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_outer_lat_fr[20], -y_data_outer_lat_fr[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(z_data_inner_lat_fr[0], -y_data_inner_lat_fr[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_inner_lat_fr[5], -y_data_inner_lat_fr[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_inner_lat_fr[10], -y_data_inner_lat_fr[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_inner_lat_fr[15], -y_data_inner_lat_fr[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_inner_lat_fr[20], -y_data_inner_lat_fr[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_lat_un[mask_fr1,2],-inner_lat_un[mask_fr1,1], s=1, c='black', marker='^')
plt.scatter(outer_lat_un[mask_fr_o1,2],-outer_lat_un[mask_fr_o1,1], s=1, c='black', marker='^')
plt.scatter(z_data_outer_lat_fr[0][0], -y_data_outer_lat_fr[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(z_data_outer_lat_fr[5][0], -y_data_outer_lat_fr[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(z_data_outer_lat_fr[10][0], -y_data_outer_lat_fr[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(z_data_outer_lat_fr[15][0], -y_data_outer_lat_fr[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(z_data_outer_lat_fr[20][0], -y_data_outer_lat_fr[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_lat_un[mask_fr1,2][0],-inner_lat_un[mask_fr1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_lat_un[mask_fr_o1,2][0],-outer_lat_un[mask_fr_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('Z [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Lateral Cross section of Inner and Outer \n Corresponding Deformation at x = -0.04 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)

plt.figure(figsize=(10, 6))
plt.scatter(z_data_outer_lat_cen[0], -y_data_outer_lat_cen[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_outer_lat_cen[5], -y_data_outer_lat_cen[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_outer_lat_cen[10], -y_data_outer_lat_cen[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_outer_lat_cen[15], -y_data_outer_lat_cen[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_outer_lat_cen[20], -y_data_outer_lat_cen[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(z_data_inner_lat_cen[0], -y_data_inner_lat_cen[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_inner_lat_cen[5], -y_data_inner_lat_cen[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_inner_lat_cen[10], -y_data_inner_lat_cen[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_inner_lat_cen[15], -y_data_inner_lat_cen[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_inner_lat_cen[20], -y_data_inner_lat_cen[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_lat_un[mask_cen1,2],-inner_lat_un[mask_cen1,1], s=1, c='black', marker='^')
plt.scatter(outer_lat_un[mask_cen_o1,2],-outer_lat_un[mask_cen_o1,1], s=1, c='black', marker='^')
plt.scatter(z_data_outer_lat_cen[0][0], -y_data_outer_lat_cen[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(z_data_outer_lat_cen[5][0], -y_data_outer_lat_cen[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(z_data_outer_lat_cen[10][0], -y_data_outer_lat_cen[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(z_data_outer_lat_cen[15][0], -y_data_outer_lat_cen[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(z_data_outer_lat_cen[20][0], -y_data_outer_lat_cen[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_lat_un[mask_cen1,2][0],-inner_lat_un[mask_cen1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_lat_un[mask_cen_o1,2][0],-outer_lat_un[mask_cen_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('Z [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Lateral Cross section of Inner and Outer \n Corresponding Deformation at x = 0 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)


plt.figure(figsize=(10, 6))
plt.scatter(z_data_outer_lat_ba[0], -y_data_outer_lat_ba[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_outer_lat_ba[5], -y_data_outer_lat_ba[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_outer_lat_ba[10], -y_data_outer_lat_ba[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_outer_lat_ba[15], -y_data_outer_lat_ba[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_outer_lat_ba[20], -y_data_outer_lat_ba[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(z_data_inner_lat_ba[0], -y_data_inner_lat_ba[0], s=0.5, c='blue', alpha=0.7)
plt.scatter(z_data_inner_lat_ba[5], -y_data_inner_lat_ba[5], s=0.5, c='green', alpha=0.7)
plt.scatter(z_data_inner_lat_ba[10], -y_data_inner_lat_ba[10], s=0.5, c='orange', alpha=0.7)
plt.scatter(z_data_inner_lat_ba[15], -y_data_inner_lat_ba[15], s=0.5, c='red', alpha=0.7)
plt.scatter(z_data_inner_lat_ba[20], -y_data_inner_lat_ba[20], s=0.5, c='yellow', alpha=0.7)
plt.scatter(inner_lat_un[mask_ba1,2],-inner_lat_un[mask_ba1,1], s=1, c='black', marker='^')
plt.scatter(outer_lat_un[mask_ba_o1,2],-outer_lat_un[mask_ba_o1,1], s=1, c='black', marker='^')
plt.scatter(z_data_outer_lat_ba[0][0], -y_data_outer_lat_ba[0][0], s=35, c='blue', alpha=0.7, label = 't=0.066s')
plt.scatter(z_data_outer_lat_ba[5][0], -y_data_outer_lat_ba[5][0], s=35, c='green', alpha=0.7, label = 't=0.33s')
plt.scatter(z_data_outer_lat_ba[10][0], -y_data_outer_lat_ba[10][0], s=35, c='orange', alpha=0.7, label = 't=0.66s')
plt.scatter(z_data_outer_lat_ba[15][0], -y_data_outer_lat_ba[15][0], s=35, c='red', alpha=0.7, label = 't=0.99s')
plt.scatter(z_data_outer_lat_ba[20][0], -y_data_outer_lat_ba[20][0], s=35, c='yellow', alpha=0.7, label = 't=1.32s')
plt.scatter(inner_lat_un[mask_ba1,2][0],-inner_lat_un[mask_ba1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nInner')
plt.scatter(outer_lat_un[mask_ba_o1,2][0],-outer_lat_un[mask_ba_o1,1][0], s=35, c='black', marker='^',label = 'Undeformed\nOuter')
plt.xlabel('Z [m]',fontsize=30)
plt.ylabel('Y [m]',fontsize=30)
plt.title('Lateral Cross section of Inner and Outer \n Corresponding Deformation at x = 0.04 [m]',fontsize=30)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axis('equal')  # Ensures aspect ratio is equal for x and y
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(.5, 0), fontsize=14,labelspacing=2, ncol = 7)


plt.show()