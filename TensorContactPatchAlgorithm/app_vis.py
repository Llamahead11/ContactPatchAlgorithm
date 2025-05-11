import time
import numpy
import open3d as o3d


class Viewer3D(object):
    def __init__(self, title):
        self.CLOUD_NAME = 'T2Cam_Deformation'
        self.LINE_NAME = 'Contact Patch Tag 0 and 49'
        self.first_cloud = True
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(title)
        self.main_vis.enable_raw_mode(enable=True)
        self.main_vis.show_settings = True
        self.main_vis.show_axes = False
        self.main_vis.size_to_fit()
        app.add_window(self.main_vis)

    def tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    def update_cloud(self, geometries,lines):
        if self.first_cloud:
            def add_first_cloud():
                self.main_vis.add_geometry(self.CLOUD_NAME, geometries)
                self.main_vis.add_geometry(self.LINE_NAME, lines)
                self.main_vis.reset_camera_to_default()
                self.main_vis.setup_camera(60,
                                           [0.07, 0.07, 0],
                                           [-0.1, -0.1, 0],
                                           [-1, 0, 0])

            add_first_cloud()
            self.main_vis.remove_geometry(self.CLOUD_NAME)
            self.main_vis.remove_geometry(self.LINE_NAME)
            self.first_cloud = False
        else:
            def update_with_cloud():
                self.main_vis.remove_geometry(self.CLOUD_NAME)
                self.main_vis.remove_geometry(self.LINE_NAME)
                self.main_vis.add_geometry(self.CLOUD_NAME, geometries)
                self.main_vis.add_geometry(self.LINE_NAME, lines)
                
            update_with_cloud()

    # def update_cloud(self, g1, g2, g3, g4, g5):
    #     if self.first_cloud:
    #         def add_first_cloud():
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("g3", g3)
    #             self.main_vis.add_geometry("g4", g4)
    #             self.main_vis.add_geometry("g5", g5)
    #             # self.main_vis.reset_camera_to_default()
    #             # self.main_vis.setup_camera(60,
    #             #                            [0.07, 0.07, 0],
    #             #                            [-0.1, -0.1, 0],
    #             #                            [-1, 0, 0])

    #         add_first_cloud()
    #         self.main_vis.remove_geometry('g1')
    #         self.main_vis.remove_geometry('g2')
    #         self.main_vis.remove_geometry('g3')
    #         self.main_vis.remove_geometry('g4')
    #         self.main_vis.remove_geometry('g5')
    #         self.first_cloud = False
    #     else:
    #         def update_with_cloud():
    #             self.main_vis.remove_geometry('g1')
    #             self.main_vis.remove_geometry('g2')
    #             self.main_vis.remove_geometry('g3')
    #             self.main_vis.remove_geometry('g4')
    #             self.main_vis.remove_geometry('g5')
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("g3", g3)
    #             self.main_vis.add_geometry("g4", g4)
    #             self.main_vis.add_geometry("g5", g5)
                
    #         update_with_cloud()
    #         # self.main_vis.update_geometry('g1', g1,0)
    #         # self.main_vis.update_geometry('g2', g2,0)
    #         # self.main_vis.update_geometry('g3', g3,0)
    #         # self.main_vis.update_geometry('g4', g4,0)
    #         # self.main_vis.update_geometry('g5', g5,0)

    # def update_cloud(self, g1, g2, d1,d2):
    #     if self.first_cloud:
    #         def add_first_cloud():
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
    #             # self.main_vis.reset_camera_to_default()
    #             # self.main_vis.setup_camera(60,
    #             #                            [0.07, 0.07, 0],
    #             #                            [-0.1, -0.1, 0],
    #             #                            [-1, 0, 0])

    #         add_first_cloud()
    #         self.main_vis.remove_geometry('g1')
    #         self.main_vis.remove_geometry('g2')
    #         self.main_vis.remove_geometry('d1')
    #         self.main_vis.remove_geometry('d2')
    #         self.first_cloud = False
    #     else:
    #         def update_with_cloud():
    #             self.main_vis.remove_geometry('g1')
    #             self.main_vis.remove_geometry('g2')
    #             self.main_vis.remove_geometry('d1')
    #             self.main_vis.remove_geometry('d2')
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
                
    #         update_with_cloud()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_cloud(self, g1, g2, d1,d2, frame, vel_arrow):
        if self.first_cloud:
            def add_first_cloud():
                self.main_vis.add_geometry("g1", g1)
                self.main_vis.add_geometry("g2", g2)
                self.main_vis.add_geometry("d1", d1)
                self.main_vis.add_geometry("d2", d2)
                self.main_vis.add_geometry("frame", frame)
                self.main_vis.add_geometry("vel_arrow", vel_arrow)
                # self.main_vis.reset_camera_to_default()
                # self.main_vis.setup_camera(60,
                #                            [0.07, 0.07, 0],
                #                            [-0.1, -0.1, 0],
                #                            [-1, 0, 0])

            add_first_cloud()
            self.main_vis.remove_geometry('g1')
            self.main_vis.remove_geometry('g2')
            self.main_vis.remove_geometry('d1')
            self.main_vis.remove_geometry('d2')
            self.main_vis.remove_geometry('frame')
            self.main_vis.remove_geometry('vel_arrow')
            self.first_cloud = False
        else:
            def update_with_cloud():
                self.main_vis.remove_geometry('g1')
                self.main_vis.remove_geometry('g2')
                self.main_vis.remove_geometry('d1')
                self.main_vis.remove_geometry('d2')
                self.main_vis.remove_geometry('frame')
                self.main_vis.remove_geometry('vel_arrow')
                self.main_vis.add_geometry("g1", g1)
                self.main_vis.add_geometry("g2", g2)
                self.main_vis.add_geometry("d1", d1)
                self.main_vis.add_geometry("d2", d2)
                self.main_vis.add_geometry("frame", frame)
                self.main_vis.add_geometry("vel_arrow", vel_arrow)
                
            update_with_cloud()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # def update_cloud(self, g1, g2, d1,d2, d3,d4,frame, vel_arrow):
    #     if self.first_cloud:
    #         def add_first_cloud():
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
    #             self.main_vis.add_geometry("d3", d3)
    #             self.main_vis.add_geometry("d4", d4)
    #             self.main_vis.add_geometry("frame", frame)
    #             self.main_vis.add_geometry("vel_arrow", vel_arrow)
    #             # self.main_vis.reset_camera_to_default()
    #             # self.main_vis.setup_camera(60,
    #             #                            [0.07, 0.07, 0],
    #             #                            [-0.1, -0.1, 0],
    #             #                            [-1, 0, 0])

    #         add_first_cloud()
    #         self.main_vis.remove_geometry('g1')
    #         self.main_vis.remove_geometry('g2')
    #         self.main_vis.remove_geometry('d1')
    #         self.main_vis.remove_geometry('d2')
    #         self.main_vis.remove_geometry('d3')
    #         self.main_vis.remove_geometry('d4')
    #         self.main_vis.remove_geometry('frame')
    #         self.main_vis.remove_geometry('vel_arrow')
    #         self.first_cloud = False
    #     else:
    #         def update_with_cloud():
    #             self.main_vis.remove_geometry('g1')
    #             self.main_vis.remove_geometry('g2')
    #             self.main_vis.remove_geometry('d1')
    #             self.main_vis.remove_geometry('d2')
    #             self.main_vis.remove_geometry('d3')
    #             self.main_vis.remove_geometry('d4')
    #             self.main_vis.remove_geometry('frame')
    #             self.main_vis.remove_geometry('vel_arrow')
    #             self.main_vis.add_geometry("g1", g1)
    #             self.main_vis.add_geometry("g2", g2)
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
    #             self.main_vis.add_geometry("d3", d3)
    #             self.main_vis.add_geometry("d4", d4)
    #             self.main_vis.add_geometry("frame", frame)
    #             self.main_vis.add_geometry("vel_arrow", vel_arrow)
                
    #         update_with_cloud()


    # def update_cloud(self, d1, d2, d3, d4, g5):
    #     if self.first_cloud:
    #         def add_first_cloud():
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
    #             self.main_vis.add_geometry("d3", d3)
    #             self.main_vis.add_geometry("d4", d4)
    #             self.main_vis.add_geometry("g5", g5)
    #             # self.main_vis.reset_camera_to_default()
    #             # self.main_vis.setup_camera(60,
    #             #                            [0.07, 0.07, 0],
    #             #                            [-0.1, -0.1, 0],
    #             #                            [-1, 0, 0])

    #         add_first_cloud()
    #         self.main_vis.remove_geometry('d1')
    #         self.main_vis.remove_geometry('d2')
    #         self.main_vis.remove_geometry('d3')
    #         self.main_vis.remove_geometry('d4')
    #         self.main_vis.remove_geometry('g5')
    #         self.first_cloud = False
    #     else:
    #         def update_with_cloud():
    #             self.main_vis.remove_geometry('d1')
    #             self.main_vis.remove_geometry('d2')
    #             self.main_vis.remove_geometry('d3')
    #             self.main_vis.remove_geometry('d4')
    #             self.main_vis.remove_geometry('g5')
    #             self.main_vis.add_geometry("d1", d1)
    #             self.main_vis.add_geometry("d2", d2)
    #             self.main_vis.add_geometry("d3", d3)
    #             self.main_vis.add_geometry("d4", d4)
    #             self.main_vis.add_geometry("g5", g5)
                
    #         update_with_cloud()


    def stop(self):
        self.main_vis.stop()


# viewer3d = Viewer3D("mytitle")
# pcd_data = o3d.data.DemoICPPointClouds()
# cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
# bounds = cloud.get_axis_aligned_bounding_box()
# extent = bounds.get_extent()

# while True:
#     # Step 1) Perturb the cloud with a random walk to simulate an actual read
#     # (based on https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/multiple_windows.py)
#     pts = numpy.asarray(cloud.points)
#     magnitude = 0.005 * extent
#     displacement = magnitude * (numpy.random.random_sample(pts.shape) -
#                                 0.5)
#     new_pts = pts + displacement
#     cloud.points = o3d.utility.Vector3dVector(new_pts)

#     # Step 2) Update the cloud and tick the GUI application
#     viewer3d.update_cloud(cloud)
#     viewer3d.tick()
#     time.sleep(0.1)