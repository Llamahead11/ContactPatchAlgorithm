import cv2
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d
import os
import cProfile

plt.ioff() 

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x,y,x-fx,y-fy]).T.reshape(-1,2,2)
    lines = np.int32(lines+0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.polylines(img_bgr,lines, 0, (0,255,0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1,y1),1, (0,255,0),-1)

    return img_bgr

def createDispPlot(label ,fig, ax, clipped, prev_3D_points, vmin=-0.003, vmax=0.003):
    sc = ax.scatter(prev_3D_points[:, 0], prev_3D_points[:, 2], prev_3D_points[:, 1], c=clipped, cmap='plasma', s=2,vmin=vmin, vmax=vmax)
    ax.set_title("{} Displacement".format(label))
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label='X Displacement [m]')
    cbar.ax.yaxis.tick_left()   
    #cbar.set_ticklabels([])
    cbar_pos = list(cbar.ax.get_position().bounds)
    ax_hist = fig.add_axes([cbar_pos[i] + offset for i, offset in enumerate([0.02, 0, 0, 0])])
    bins = np.linspace(vmin, vmax, 100 + 1)
    hist, bins, patches = ax_hist.hist(clipped, bins=bins, orientation='horizontal', color = 'gray', alpha=1)
    for patch, bin_edge in zip(patches, bins[:-1]):
        norm_val = (bin_edge - np.min(bins)) / (np.max(bins) - np.min(bins))
        color = plt.cm.plasma(norm_val)  # Map bin value to colormap
        patch.set_facecolor(color)
    ax_hist.set_xticks(np.linspace(0,np.max(hist),2))
    ax_hist.set_yticks(np.linspace(vmin, vmax, num=7))
    ax_hist.set_ylim(vmin, vmax)

def createOuterDeformationPlot(label, points, dist, vmin=-0.003, vmax=0.003):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 2], -points[:, 0], points[:, 1], c=dist, cmap='plasma', s=2,vmin=vmin, vmax=vmax)
    ax.set_title("{} Deformation".format(label))
    ax.azim = 90
    ax.elev = 70
    plt.ylim(-0.3,0.3)
    plt.xlim(0,0.3)
    ax.set_zlim(0,0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label='Displacement [m]')
    cbar.ax.yaxis.tick_left()   
    #cbar.set_ticklabels([])
    cbar_pos = list(cbar.ax.get_position().bounds)
    ax_hist = fig.add_axes([cbar_pos[i] + offset for i, offset in enumerate([0.02, 0, 0, 0])])
    bins = np.linspace(vmin, vmax, 100 + 1)
    hist, bins, patches = ax_hist.hist(dist, bins=bins, orientation='horizontal', color = 'gray', alpha=1)
    for patch, bin_edge in zip(patches, bins[:-1]):
        norm_val = (bin_edge - np.min(bins)) / (np.max(bins) - np.min(bins))
        color = plt.cm.plasma(norm_val)  # Map bin value to colormap
        patch.set_facecolor(color)
    ax_hist.set_xticks(np.linspace(0,np.max(hist),2))
    ax_hist.set_yticks(np.linspace(vmin, vmax, num=7))
    ax_hist.set_yticklabels([])
    ax_hist.set_ylim(vmin, vmax)
    return fig, sc

def updateScatterPlot(sc, points, dist, vmin=-0.003, vmax=0.003):
    # Update the data for the scatter plot
    sc._offsets3d = (points[:, 2], -points[:, 0], points[:, 1])
    sc.set_array(dist)  # Update the color data
    sc.set_clim(vmin, vmax)  # Update the color limits
    
    # Redraw the plot
    plt.draw()

def visContactEdge(label, points,vmin,vmax,dist):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 2], -points[:, 0], points[:, 1], c=dist, cmap='plasma', s=2,vmin=vmin, vmax=vmax)
    ax.set_title("{} Contact Patch Edge".format(label))
    ax.azim = 90
    ax.elev = 70
    plt.ylim(-0.3,0.3)
    plt.xlim(0,0.3)
    ax.set_zlim(0,0.4)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    return fig

# def createDeformationPlot(ax,fig,label, points, dist,vmin,vmax,azim,elev):
#     sc = ax.scatter(points[:, 2], -points[:, 0], points[:, 1], c=dist, cmap='plasma', s=2,vmin=vmin, vmax=vmax)
#     ax.set_title("{} Deformation".format(label))
#     ax.azim = azim
#     ax.elev = elev
#     plt.ylim(-0.3,0.3)
#     plt.xlim(0,0.3)
#     ax.set_zlim(0,0.4)
#     ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
#     cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label='Displacement [m]')
#     cbar.ax.yaxis.tick_left()   
#     #cbar.set_ticklabels([])
#     cbar_pos = list(cbar.ax.get_position().bounds)
#     ax_hist = fig.add_axes([cbar_pos[i] + offset for i, offset in enumerate([0.02, 0, 0, 0])])
#     bins = np.linspace(vmin, vmax, 100 + 1)
#     hist, bins, patches = ax_hist.hist(dist, bins=bins, orientation='horizontal', color = 'gray', alpha=1)
#     for patch, bin_edge in zip(patches, bins[:-1]):
#         norm_val = (bin_edge - np.min(bins)) / (np.max(bins) - np.min(bins))
#         color = plt.cm.plasma(norm_val)  # Map bin value to colormap
#         patch.set_facecolor(color)
#     ax_hist.set_xticks(np.linspace(0,np.max(hist),2))
#     ax_hist.set_yticks(np.linspace(vmin, vmax, num=7))
#     ax_hist.set_ylim(vmin, vmax)

# def makeOrthoDeformationPlot(label, points, dist, vmin=-0.003, vmax=0.003):
#     fig = plt.figure(figsize=(54, 12))
#     ax1 = fig.add_subplot(131, projection='3d')
#     createDeformationPlot(ax1,fig,label, points, dist,vmin,vmax,90,70)

#     # Y Displacement Plot
#     ax2 = fig.add_subplot(132, projection='3d')
#     createDeformationPlot(ax2,fig,label, points, dist,vmin,vmax,0,30)

#     # Z Displacement Plot
#     ax3 = fig.add_subplot(133, projection='3d')
#     createDeformationPlot(ax3,fig,label, points, dist,vmin,vmax,30,60)
#     return fig
    
def createDeformationPlot(ax, fig, label, points, dist, vmin, vmax, azim, elev, bins, norm):
    """Creates a 3D deformation scatter plot with a histogram color bar."""
    
    # Scatter Plot
    sc = ax.scatter(points[:, 2], -points[:, 0], points[:, 1], c=dist, cmap='plasma', s=2, vmin=vmin, vmax=vmax)
    ax.set_title(f"{label} Deformation")
    ax.azim, ax.elev = azim, elev
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.4)

    # Maintain aspect ratio
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label='Displacement [m]')
    cbar.ax.yaxis.tick_left()

    # Histogram Plot
    cbar_pos = np.array(cbar.ax.get_position().bounds)  # Use NumPy array for direct modification
    ax_hist = fig.add_axes(cbar_pos + [0.02, 0, 0, 0])  # Shift colorbar position
    hist, _, patches = ax_hist.hist(dist, bins=bins, orientation='horizontal', color='gray', alpha=1)

    # Apply colormap to histogram bins
    for patch, bin_edge in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.plasma(norm(bin_edge)))

    # Set histogram axis properties
    ax_hist.set_xticks(np.linspace(0, np.max(hist), 2))
    ax_hist.set_yticks(np.linspace(vmin, vmax, num=7))
    ax_hist.set_ylim(vmin, vmax)

def makeOrthoDeformationPlot(label, points, dist, vmin=-0.003, vmax=0.003):
    """Creates three orthogonal deformation plots."""
    
    fig = plt.figure(figsize=(54, 12))
    bins = np.linspace(vmin, vmax, 101)  # Precompute bins for histogram
    norm = plt.Normalize(vmin, vmax)     # Normalize color mapping once

    angles = [(90, 70), (0, 30), (30, 60)]
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

    for ax, (azim, elev) in zip(axes, angles):
        createDeformationPlot(ax, fig, label, points, dist, vmin, vmax, azim, elev, bins, norm)

    return fig

def plotAndSavePlt(local_point_displacements, prev_3D_points, vmin=-0.003, vmax=0.003):
    clipped_x = np.clip(local_point_displacements[:, 0], vmin, vmax)
    clipped_y = np.clip(local_point_displacements[:, 1], vmin, vmax)
    clipped_z = np.clip(local_point_displacements[:, 2], vmin, vmax)

    x_colors = plt.get_cmap('plasma')((clipped_x - clipped_x.min()) / (clipped_x.max() - clipped_x.min()))
    y_colors = plt.get_cmap('plasma')((clipped_y - clipped_y.min()) / (clipped_y.max() - clipped_y.min()))
    z_colors = plt.get_cmap('plasma')((clipped_z - clipped_z.min()) / (clipped_z.max() - clipped_z.min()))
    x_colors = x_colors[:, :3]
    y_colors = y_colors[:, :3]
    z_colors = z_colors[:, :3]

    fig = plt.figure(figsize=(15, 5))

    # X Displacement Plot
    ax1 = fig.add_subplot(131, projection='3d')
    createDispPlot("X",fig,ax1,clipped_x,prev_3D_points,vmin,vmax)

    # Y Displacement Plot
    ax2 = fig.add_subplot(132, projection='3d')
    createDispPlot("Y",fig,ax2,clipped_y,prev_3D_points,vmin,vmax)

    # Z Displacement Plot
    ax3 = fig.add_subplot(133, projection='3d')
    createDispPlot("Z",fig,ax3,clipped_z,prev_3D_points,vmin,vmax)

    #plt.close()

def saveImages():
    pass

def makeVidfromImage(image_folder):
    #video_name = 'Inner_Deformation_updated_ICP_downsampled.avi'
    video_name = 'hist_of_norm.avi' #'Outer_def_plane_max_100.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    #print("Images:", images)

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")


def main():
    # makeVidfromImage("./Outer_Deformation/")
    makeVidfromImage("./histnorm/")

if __name__ == "__main__":
    main()