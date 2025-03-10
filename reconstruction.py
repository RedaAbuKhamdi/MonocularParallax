import os
import numpy as np
from matplotlib import animation, pyplot as plt
from os import listdir
from os.path import isfile, join
import re
from skimage import io
from scipy import ndimage
from numba import jit, prange
from tqdm import tqdm
from ellipsemodel import EllipseModel
import traceback

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class InverseCylinder:
    def __init__(self, inverse : bool = True):
        self.inverse = inverse
        self.images = []
        folder = "./inputFrames"
        for image in listdir(folder):
            if isfile(join(folder, image)):
                print("./inputFrames/" + image)
                slice = np.transpose(io.imread(folder + "/" + image))
                slice = ndimage.binary_erosion(slice.copy() > 0, ndimage.generate_binary_structure(2, 2))
                self.images.append(slice)
        self.images = np.array(self.images)
        print(self.images.shape)
        self.extract_points()
        self.chosen_points = None
    
    def extract_points(self):
        self.points = []
        images = self.images.copy()
        for i in range(images.shape[0]):
            print(i)
            cylinder_points = np.argwhere(images[i] > 0).astype(np.float32)
            print(cylinder_points.shape)
            self.points.append(cylinder_points)
        

    def calculate_angle_wurf(self, p1 : np.ndarray, p2 : np.ndarray, p3 : np.ndarray, p4 : np.ndarray, center : np.ndarray):
        sin_alpha = np.sin(self.calculate_angle(p1, p2, center))
        sin_beta = np.sin(self.calculate_angle(p2, p3, center))
        sin_gamma = np.sin(self.calculate_angle(p3, p4, center))
        return 0 if ((sin_alpha + sin_beta + sin_gamma) * sin_beta) == 0 else float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))

    
    def calculate_min_max_dimension(self, dim : int):
        if self.points is None:
            self.extract_points()
        max_dim = 0
        min_dim = np.inf
        
        for i in range(len(self.points)):
            frame = self.points[i]
            max_dim = max(np.max(frame[:,dim]), max_dim)
            min_dim = min(np.min(frame[:,dim]), min_dim)
        return min_dim, max_dim
    def get_width(self):
        min_x, max_x = self.calculate_min_max_dimension(0)
        return max_x - min_x
    
    def get_center_axis(self):
        min_x, max_x = self.calculate_min_max_dimension(0)
        return min_x + (max_x - min_x) / 2

    def calculate_ellipse(self, track : np.ndarray):
        x_c = self.get_center_axis()
        a = self.get_width() / 2
        min_y, max_y = self.calculate_min_max_dimension(1)
        ellipse_model = EllipseModel(track)
        params = ellipse_model.fit_ellipse(x_c, a, min_y, max_y)
        return ellipse_model, params
    def find_point_on_circle(self, point : np.ndarray, radius : float):
        delta = np.sqrt((point[1]**2) / (point[0]**2) + 1)
        circle_pts = np.array([
            [radius / delta, point[1] * radius / (point[0] * delta)],
            [radius / (-delta), point[1] * radius / (-point[0] * delta)]
        ])
        return circle_pts[0] if np.linalg.norm(point - circle_pts[0]) < np.linalg.norm(point - circle_pts[1]) else circle_pts[1]
    def calculate_angle_ellipse(self, ellipse_model : EllipseModel):
        x_c, y_c, a, b = ellipse_model.params
        track = ellipse_model.track
        track = track -  np.tile(np.array([x_c, y_c]), (track.shape[0], 1)) 
        ellipse_model.calculate_angle_original()    
        # dir = track[1] - track[0]
        # conjugate = np.array([((b/a) ** 2) * dir[0] , dir[1]])
        # ts = np.linspace(- 1 / (((dir[0] / a) ** 2 + (dir[1] / b) ** 2) ** 0.5 ), 1 / (((dir[0] / a) ** 2 + (dir[1] / b) ** 2) ** 0.5 ), 100)
        # pts = np.zeros((ts.shape[0], 2))
        # pts_c = np.zeros((ts.shape[0], 2))
        # for i in range(ts.shape[0]):
        #     pts[i] = dir * ts[i]
        #     pts_c[i] = conjugate * ts[i]
        # plt.scatter(pts[:,0], pts[:,1], color = "red")
        # plt.scatter(pts_c[:,0], pts_c[:,1], color = "red")
        # # for i in range(track.shape[0]):
        # #     track[i] = self.find_point_on_circle(track[i], 1)
        # plt.scatter(track[:,0], track[:,1], color = "green")
        # plt.scatter(0, 0)
        # angles = np.linspace(0, 2 * np.pi, 100)
        # xs =  np.cos(angles) * a
        # ys =  np.sin(angles) * b
        # plt.scatter(xs, ys, color = "blue")
        # plt.show()
        # for i in range(2, track.shape[0]):
        #     pt0 = track[i - 2]
        #     pt1 = track[i - 1]
        #     pt2 = track[i]
        #     p1 = pt1 - pt0
        #     p2 = pt2 - pt1
        #     print("angle = {}".format(
        #         np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
        #     ))
    def calculate_angle(self, point1, point2, center):
        vector1 = point1 - center
        vector2 = point2 - center
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    def visualize(self, chosen_points = []):
        fig, ax = plt.subplots()
        print(chosen_points)
        def update(n):
            ax.clear()
            ax.set_title("Frame {}".format(n))
            slice_image = ndimage.binary_dilation(self.images[n].copy().astype(np.uint16), ndimage.generate_binary_structure(2, 3))
            slice_image = slice_image.astype(np.uint16)
            slice_image *= 60
            contrasting_slice_image = np.zeros((self.images[n].shape[0], self.images[n].shape[1]), dtype=np.uint16)
            for pt_index in range(len(chosen_points)):
                contrasting_slice_image[chosen_points[pt_index][n][0], chosen_points[pt_index][n][1]] = 1
            contrasting_slice_image = ndimage.binary_dilation(contrasting_slice_image, ndimage.generate_binary_structure(2, 3))
            contrasting_slice_image = contrasting_slice_image.astype(np.uint16)
            contrasting_slice_image *= 254
            ax.imshow(slice_image + contrasting_slice_image, cmap='gray', vmin=0, vmax=255) 
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.images.shape[0], interval=1500)
        plt.show()

    def visualize_ellipse(self, params : tuple, track : np.ndarray):
        fig, ax = plt.subplots()
        xs = np.linspace(params[0] - params[2], params[0] + params[2], 100)
        ellipse_pts = []
        x_c = params[0]
        a = params[2]
        b = params[3]
        y_c = params[1]
        for x in xs:
            ellipse_pts.append(np.array([
                x,
                (np.sqrt(1 - ((x - x_c)**2) / (a**2)) * b + y_c)
            ]))
            ellipse_pts.append(np.array([
                x,
                (-np.sqrt(1 - ((x - x_c)**2) / (a**2)) * b + y_c)
            ]))
        ellipse_pts = np.array(ellipse_pts)
        def update(n):
            ax.clear()
            # ax.set_xlim(400, 1500)
            # ax.set_ylim(0, 1000)
            ax.set_title("Frame {}".format(n))
            cylinder_pts = self.points[n]
            ax.scatter(ellipse_pts[:,0], ellipse_pts[:, 1], color = "red", s=16)
            ax.scatter(cylinder_pts[:,0], cylinder_pts[:, 1], color = "blue", s=16)   
            ax.scatter(track[n,0], track[n, 1], color = "green", s=16)   
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=track.shape[0], interval=1500)
        plt.show()
    def visualize_track(self, track):
        fig, ax = plt.subplots()
        def update(n):
            ax.clear()
            ax.set_xlim(400, 1500)
            ax.set_ylim(0, 1000)
            ax.set_title("Frame {}".format(n))
            cylinder_pts = self.points[n]
            ax.scatter(track[:,0], track[:, 1], color = "red", s=16)
            ax.scatter(cylinder_pts[:,0], cylinder_pts[:, 1], color = "blue", s=16)   
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.images.shape[0], interval=1500)
        plt.show()
    def get_tracks(self):
        for file in os.listdir("./trajectories"):
            track = np.load("./trajectories/{}".format(file))
            yield track
    def visualize_tracks(self):
        for track in self.get_tracks():
            self.visualize_track(track)
    def clear_trajectories(self):
        for file in os.listdir("./trajectories"):
            os.remove("./trajectories/{}".format(file))
    def wurf(self, pt1, pt2, pt3, pt4):
        ab = np.linalg.norm(pt2 - pt1)
        bc = np.linalg.norm(pt3 - pt2)
        cd = np.linalg.norm(pt4 - pt3)
        return ab*cd / ((ab + bc + cd) * bc)
    def track_by_wurf(self):
        threshold = 0.00001
        self.clear_trajectories()
        for i in range(len(self.points[0])):
            track = None
            best = np.inf
            pt1 = self.points[0][i]
            for j in np.argsort(np.sum(((self.points[1] - pt1)**2), axis=1))[:15]:
                pt2 = self.points[1][j]
                for k in np.argsort(np.sum(((self.points[2] - pt2)**2), axis=1))[:15]:
                    pt3 = self.points[2][k]
                    for l in np.argsort(np.sum(((self.points[3] - pt3)**2), axis=1))[:15]:
                        pt4 = self.points[3][l]
                        for m in np.argsort(np.sum(((self.points[4] - pt4)**2), axis=1))[:15]:
                            pt5 = self.points[4][m]
                            wurf = max(self.wurf(pt1, pt2, pt3, pt4), self.wurf(pt2, pt3, pt4, pt5))
                            if np.abs(wurf - 1/3) < min(threshold, best):
                                best = np.abs(wurf - 1/3)
                                track = np.array([pt1, pt2, pt3, pt4, pt5])
                                print(best, track, i)
            if track is not None:
                np.save("./trajectories/trajectory{}.npy".format(i), track)
    def track_by_proximity(self):
        self.clear_trajectories()
        input()
        indecies = np.argsort(self.points[0][:,1])
        track = np.zeros((len(self.points), 2))
        for index in indecies:
            track[0] = self.points[0][index]
            for i in range(1, len(self.points)):
                pt_distance = np.sum(((self.points[i] - track[i - 1])**2), axis=1) 
                pt_index = np.argsort(pt_distance)[0]
                track[i] = self.points[i][pt_index]
            success = True
            for i in range(3, track.shape[0]):
                wurf = self.wurf(track[i - 3], track[i - 2], track[i - 1], track[i])
                if np.abs(wurf - 1/3) > 0.001:
                    success = False
                    break
            if success:
                np.save("./trajectories/trajectory{}.npy".format(index), track)
