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
from skimage.measure import EllipseModel
import traceback

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class InverseCylinder:
    def __init__(self, inverse : bool = True):
        self.inverse = inverse
        self.images = []
        for image in listdir("./inputFrames"):
            if isfile(join("./inputFrames", image)):
                print("./inputFrames/" + image)
                slice = np.transpose(io.imread("./inputFrames/" + image))
                slice = ndimage.binary_erosion(slice.copy(), ndimage.generate_binary_structure(2, 3))
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

    def get_width(self):
        if self.points is None:
            self.extract_points()

        max_x = 0
        min_x = np.inf
        
        for i in range(len(self.points)):
            frame = self.points[i]
            max_x = max(np.max(frame[:,0]), max_x)
            min_x = min(np.min(frame[:,0]), min_x)
        
        return max_x - min_x

    def fit_ellipse(self):
        

    def calculate_ellipse_equation(seld, track : np.ndarray):
        A = np.zeros((5, 5))
        B = np.ones((5))
        for i in range(5):
            pt = track[i]
            A[i, 0] = pt[0]**2
            A[i, 1] = pt[1]**2
            A[i, 2] = pt[0]*pt[1]
            A[i, 3] = pt[0]
            A[i, 4] = pt[1]
        result = np.linalg.solve(A, B)
        print("Ellipse check")
        for i in range(track.shape[0]):
            pt = track[i]
            print(result[0] * pt[0]**2 
                  + result[1] * pt[1]**2
                  + result[2] * pt[0]*pt[1]
                  + result[3] * pt[0]
                  + result[4] * pt[1])
        return result
    def calculate_ellipse_center(self, coeffs : np.ndarray):
        A = np.array([
            [2* coeffs[0], coeffs[2]],
            [coeffs[2], 2 * coeffs[1]]
        ])
        B = np.array([-coeffs[3], -coeffs[4]])
        return np.linalg.solve(A, B)
    def calculate_ellipse(self, track : np.ndarray):
        result = self.calculate_ellipse_equation(track)
        center = self.calculate_ellipse_center(result)
        valid = False
        model = None
        x_c = center[0]
        y_c = center[1]
        print("Ellipse center {}".format(center))
        print("{} x*x + {} y*y + {} x*y + {} x + {} y = 1".format(*tuple([str(el) for el in result])))
        if result[2]**2 - 4*result[0]*result[1] <= 0:

            model = EllipseModel()
            model.estimate(track)
            b = model.params[3]
            scaled_track = track
            scaled_track[:,1] = scaled_track[:,1] * b
            scaled_center = np.array([x_c, y_c * b])

            print("Ellipse: xc = {}, yc = {}, a = {}, b = {}, theta = {}".format(*model.params))
            print("Ellipse center {}".format(center))

            for i in range(1, track.shape[0]):
                print(self.calculate_angle(scaled_track[i-1], scaled_track[i], scaled_center))
            valid = abs(model.params[4]) < 0.1
        else:
            print("Not ellipse")
        return result, valid, model
        

    def calculate_angle(self, point1, point2, center):
        vector1 = point1 - center
        vector2 = point2 - center
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    def calculate_wurf(self, index : int, start : int = 0):
        center = np.array([self.images[index].shape[0] / 2, self.images[index].shape[1] / 2])
        sin_alpha = np.sin(self.calculate_angle(self.chosen_points[start][index], self.chosen_points[start - 1][index], center))
        sin_beta = np.sin(self.calculate_angle(self.chosen_points[start - 1][index], self.chosen_points[start - 2][index], center))
        sin_gamma = np.sin(self.calculate_angle(self.chosen_points[start - 2][index], self.chosen_points[start - 3][index], center))
        return float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))
    
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

    def visualize_track(self, track, result, model):
        fig, ax = plt.subplots()
        x = np.linspace(400, 1500, 1000)
        y = np.linspace(0, 1000, 1000)
        ellipse_pts = []
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                value = result[0] * x[i]**2 + result[1] * y[j]**2 + result[2] * x[i]*y[j] + result[3] * x[i]+ result[4] * y[j] - 1
                if(abs(value) < 10e-3):
                    ellipse_pts.append(np.array([x[i], y[j]]))

        ellipse_pts = np.array(ellipse_pts)
        def update(n):
            ax.clear()
            ax.set_xlim(400, 1500)
            ax.set_ylim(0, 1000)
            ax.set_title("Frame {}".format(n))
            cylinder_pts = self.points[n]
            ax.scatter(track[:,0], track[:, 1], color = "red")
            ax.scatter(cylinder_pts[:,0], cylinder_pts[:, 1], color = "blue")   
            preds = model.predict_xy(np.linspace(0, 2 * np.pi, 100)) 
            if (len(ellipse_pts) > 0):
                ax.scatter(ellipse_pts[:,0], ellipse_pts[:, 1], color = "green")   
            ax.scatter(preds[:,0], preds[:, 1], color = "purple")  


        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.images.shape[0], interval=1500)
        plt.show()
    def visualize_tracks(self):
        for file in os.listdir("./trajectories"):
            track = np.load("./trajectories/{}".format(file))
            valid = False
            model = None
            try:
                result, valid, model = self.calculate_ellipse(track.copy())
            except: 
                print(traceback.format_exc())
            if valid:
                self.visualize_track(track, result, model)

    def track_by_proximity(self):
        indecies = np.argsort(self.points[0][:,1])
        track = np.zeros((len(self.points), 2))
        for index in indecies:
            track[0] = self.points[0][index]
            for i in range(1, len(self.points)):
                pt_distance = np.sum(((self.points[i] - track[i - 1])**2), axis=1) 
                pt_index = np.argsort(pt_distance)[0]
                track[i] = self.points[i][pt_index]
            np.save("./trajectories/trajectory{}.npy".format(index), track)
