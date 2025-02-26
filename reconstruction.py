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

SEARCH_LENGTH = 100
EPSILON = 1e-3

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
@jit(fastmath = True)
def calculate_angle(point1, point2, center):
    vector1 = point1 - center
    vector2 = point2 - center
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
@jit(fastmath = True)
def calculate_angle_wurf(p1 : np.ndarray, p2 : np.ndarray, p3 : np.ndarray, p4 : np.ndarray, center : np.ndarray):
    sin_alpha = np.sin(calculate_angle(p1, p2, center))
    sin_beta = np.sin(calculate_angle(p2, p3, center))
    sin_gamma = np.sin(calculate_angle(p3, p4, center))
    return 0 if ((sin_alpha + sin_beta + sin_gamma) * sin_beta) == 0 else float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))
@jit(nopython = True, parallel = True)
def search_support(i : int, j : int, k : int, frames : np.ndarray, center : np.ndarray, indicies : np.ndarray):
    vals = np.zeros(len(frames[3]))
    for l in prange(indicies.shape[0]):
        vals[indicies[l]] = calculate_angle_wurf(frames[0][i], frames[1][j], frames[2][k], frames[3][indicies[l]], center)
    return vals
def find_points_for_given_point(center : np.ndarray, frames : np.ndarray, i : int):
        best_points = np.zeros((4, 2))
        best_val = np.inf
        indicies = ModuleNotFoundError
        for j in np.nonzero(
                np.sum(((frames[1] - frames[0][i])**2), axis=1) < (SEARCH_LENGTH * SEARCH_LENGTH)
                )[0]:
            for k in np.nonzero(
                np.sum(((frames[2] - frames[1][j])**2), axis=1) < (SEARCH_LENGTH * SEARCH_LENGTH)
                )[0]:
                vals = search_support(i, j, k, frames, center, np.nonzero(
                np.sum(((frames[3] - frames[2][k])**2), axis=1) < (SEARCH_LENGTH * SEARCH_LENGTH)
                )[0])
                differences = np.abs(vals - 1/3)
                min_val = np.argmin(differences)
                min_wurf = vals[min_val]
                if differences[min_val] < min(EPSILON, best_val):
                    best_points[0] = frames[0][i]
                    best_points[1] = frames[1][j]
                    best_points[2] = frames[2][k]
                    best_points[3] = frames[3][min_val]
                    indicies = (i, j, k, min_val)
                    best_val = differences[min_val]
        return best_points, indicies
class InverseCylinder:
    def __init__(self, inverse : bool = True):
        self.inverse = inverse
        self.images = []
        for image in listdir("./inputFrames"):
            if isfile(join("./inputFrames", image)):
                print("./inputFrames/" + image)
                slice = np.transpose(io.imread("./inputFrames/" + image))
                slice = ndimage.binary_erosion(slice.copy() > 0, ndimage.generate_binary_structure(2, 3))
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

    def calculate_ellipse_equation(seld, track : np.ndarray):
        A = np.zeros((5, 5))
        B = np.ones((5))
        for i in range(5):
            pt = track[i * 2]
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
    def track_point(self, center : np.ndarray, index : int = 0):
        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        if os.path.isfile("./trajectories/trajectory{}.npy".format(index)):
            return np.load("./trajectories/trajectory{}.npy".format(index))
        trajectory = np.zeros((self.images.shape[0], 2), dtype=np.float32)
        first_points, indicies = find_points_for_given_point(center, self.points, index)
        indicies = list(indicies)
        if first_points is None:
            raise Exception("No points found!")

        trajectory[0] = first_points[0]
        trajectory[1] = first_points[1]
        trajectory[2] = first_points[2]
        trajectory[3] = first_points[3]
        
        for i in range(4, self.images.shape[0]):
            found_point = False
            min_val = np.inf
            indicies.append(None)
            for j in np.nonzero(
                    np.sum(((self.points[i] - trajectory[i-1])**2), axis=1) < (SEARCH_LENGTH * SEARCH_LENGTH)
                )[0]:
                wurf = calculate_angle_wurf(trajectory[i-3], trajectory[i-2], trajectory[i-1], self.points[i][j], center)
                if np.abs(wurf - 1/3) < min(EPSILON, min_val):
                    trajectory[i] = self.points[i][j]
                    found_point = True
                    min_val = np.abs(wurf - 1/3)
                    indicies[i] = j
            if not found_point:
                raise Exception("No point was found for trajectory!")
            
        np.save("./trajectories/trajectory{}.npy".format(index), trajectory)
        for i, index in enumerate(indicies):
            self.points[i][index][0] = 0
            self.points[i][index][1] = 0
        return trajectory

    def calculate_contrasting_points(self):
        chosen_points = []

        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        x = self.images[0].shape[0] // 2 
        y = self.images[0].shape[1] // 2 
        print("x = {}, y = {}".format(x, y))

        for i in tqdm(range(self.points[0].shape[0])):
            if self.points[0][i][0] == 0 and self.points[0][i][1] == 0:
                #print("already have {}".format(i))
                continue
            try:
                chosen_points.append(
                        self.track_point(np.array([x, y]), i)
                    )
                
                print("Found track for {}".format(i))
            except Exception as e:  
                #print(traceback.format_exc())
                pass
        return chosen_points
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
        x = np.linspace(400, 1500, 1500)
        y = np.linspace(0, 1000, 1500)
        ellipse_pts = []
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                value = result[0] * x[i]**2 + result[1] * y[j]**2 + result[2] * x[i]*y[j] + result[3] * x[i]+ result[4] * y[j] - 1
                if(abs(value) < 10e-6):
                    ellipse_pts.append(np.array([x[i], y[j]]))

        ellipse_pts = np.array(ellipse_pts)
        def update(n):
            ax.clear()
            ax.set_xlim(400, 1500)
            ax.set_ylim(0, 1000)
            ax.set_title("Frame {}".format(n))
            cylinder_pts = self.points[n]
            ax.scatter(track[:,0], track[:, 1], color = "red", s=16)
            ax.scatter(cylinder_pts[:,0], cylinder_pts[:, 1], color = "blue", s=16)   
            if (len(ellipse_pts) > 0):
                ax.scatter(ellipse_pts[:,0], ellipse_pts[:, 1], color = "green", s=16)   
            # if model is not None:  
            #     preds = model.predict_xy(np.linspace(0, 2 * np.pi, 100)) 
            #     ax.scatter(preds[:,0], preds[:, 1], color = "purple")  


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
            if (valid):
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
