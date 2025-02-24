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

SEARCH_LENGTH = 25
SEARCH_LENGTH_MIN = 5      
EPSILON = 1e-4

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
                self.images.append(io.imread("./inputFrames/" + image))
        self.images = np.array(self.images)
        self.extract_points()
        self.chosen_points = None
        self.points = None
    
    def extract_points(self):
        self.points = []
        images = self.images.copy()
        for i in range(self.images.shape[0]):
            cylinder_points = np.argwhere(images[i] > 0 if self.inverse else images[i] == 0).astype(np.float32)
            self.points.append(cylinder_points)

    def track_point(self, center : np.ndarray, index : int = 0):
        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        if os.path.isfile("./trajectories/trajectory{}.npy".format(index)):
            return np.load("./trajectories/trajectory{}.npy".format(index))
        trajectory = np.zeros((self.images.shape[0], 2), dtype=np.float32)
        first_points, indicies = find_points_for_given_point(center, self.points, index)
        indicies = list(indicies)
        if first_points is None:
            print("No points found for index {}".format(index))
            raise Exception("No points found!")

        trajectory[0] = first_points[0]
        trajectory[1] = first_points[1]
        trajectory[2] = first_points[2]
        trajectory[3] = first_points[3]
        min_val = np.inf
        for i in range(4, self.images.shape[0]):
            found_point = False
            indicies.append(None)
            for j in np.nonzero(
                    np.sum(((self.points[i] - trajectory[i-1])**2), axis=1) < (SEARCH_LENGTH * SEARCH_LENGTH)
                )[0]:
                wurf = self.calculate_angle_wurf(trajectory[i-3], trajectory[i-2], trajectory[i-1], self.points[i][j], center)
                if np.abs(wurf - 1/3) < min(EPSILON, min_val):
                    print(np.abs(wurf - 1/3))
                    trajectory[i] = self.points[i][j]
                    found_point = True
                    min_val = np.abs(wurf - 1/3)
                    indicies[i] = j
            if not found_point:
                raise Exception("No point was found for trajectory!")
            
        np.save("./trajectories/trajectory{}.npy".format(index), trajectory)
        np.save("./trajectories/indicies{}.npy".format(index), np.array(indicies))
        for i, index in enumerate(indicies):
            self.points[i][index][0] = 0
            self.points[i][index][1] = 0
        return trajectory

    def calculate_angle_wurf(self, p1 : np.ndarray, p2 : np.ndarray, p3 : np.ndarray, p4 : np.ndarray, center : np.ndarray):
        sin_alpha = np.sin(calculate_angle(p1, p2, center))
        sin_beta = np.sin(calculate_angle(p2, p3, center))
        sin_gamma = np.sin(calculate_angle(p3, p4, center))
        return 0 if ((sin_alpha + sin_beta + sin_gamma) * sin_beta) == 0 else float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))
    def calculate_contrasting_points(self):
        chosen_points = []

        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        x = self.images[0].shape[0] // 2 
        y = self.images[0].shape[1] // 2 
        print("x = {}, y = {}".format(x, y))

        for file in os.listdir("./trajectories"):
            if "indicies" in file:
                indicies = np.load("./trajectories/{}".format(file))
                for i, index in enumerate(indicies):
                    self.points[i][index][0] = 0
                    self.points[i][index][1] = 0

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
                #print("No point found for {}".format(self.points[0][i]))
                pass
        return chosen_points

    def calculate_ellise_tracks(self, track : np.ndarray, index : int):
        point = track[index - 1]
        best_model = None
        center_distance = np.inf
        center_x = self.images[0].shape[0] // 2 
        if index >= 5:
            model = EllipseModel()
            try:
                success = model.estimate(track)
                if success and np.sum(np.abs(model.residuals(track))) < 1:
                    return model
                else:
                    return None
            except:
                # print(track)
                # print(traceback.format_exc())
                return None
        distances = np.sum(((self.points[index] - point)**2), axis=1)
        for i in np.nonzero(
                    np.logical_and(distances > (SEARCH_LENGTH_MIN * SEARCH_LENGTH_MIN), distances < (SEARCH_LENGTH * SEARCH_LENGTH))
                )[0]:
            track[index] = self.points[index][i]
            model = self.calculate_ellise_tracks(track, index + 1)
            if model:
                xc = model.params[0]
                if np.abs(xc - center_x) < min(50, center_distance) and 0 <= model.params[4] < 0.2:
                    center_distance = np.abs(xc - center_x)
                    print(center_distance)
                    best_model = model
        return best_model


    def calculate_contrasting_points_ellipse(self):
        chosen_points = []

        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")

        for i in tqdm(range(self.points[0].shape[0])):
            track = np.zeros((5, 2))
            track[0] = self.points[0][i]
            model = self.calculate_ellise_tracks(track, 1)
            if model:
                chosen_points.append(track)
                print(model.params)
                preds = model.predict_xy(np.linspace(0, 2 * np.pi, 15))
                plt.scatter(preds[:, 0], preds[:, 1], color="red")
                plt.scatter(track[:, 0], track[:, 1], color="blue")
                plt.show()

        return chosen_points

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
    
    def visualize_by_input(self):
        available_points = [
            re.findall(r'\d+', f)[0] 
            for f in os.listdir("./trajectories") 
            if "trajectory" in f
        ] 
        print("Available points: {}".format(available_points))
        while True:
            choice = input("Type one of the following:\nA point number to show it's track\n\'all\' to show all tracks\n \'calculate\' to run point choice algorithm\nanything else to quit: ")
            if choice in available_points:
                self.visualize(
                    [np.load("./trajectories/trajectory{}.npy".format(choice)).astype(np.int32)]
                )
            elif choice == "all":
                chosen_points = []
                for file in os.listdir("./trajectories"):
                    if "trajectory" in file:
                        chosen_points.append(np.load("./trajectories/{}".format(file)))
                self.visualize(chosen_points)
                break
            elif choice == "calculate":
                self.chosen_points = self.calculate_contrasting_points()
                self.visualize(self.chosen_points)
                break
            else:
                break
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

    def ellipse_try(self):
        distance = 0
        index = -1
        track = np.zeros((len(self.points), 2))
        for i in range(self.points[0].shape[0]):
            pt_distance = np.sum(((self.points[0] - self.points[0][i])**2), axis=1) 
            pt_index = np.argmax(pt_distance)
            if pt_distance[pt_index] > distance:
                distance = pt_distance[pt_index]
                index = i
        track[0] = self.points[0][index]
        for i in range(1, len(self.points)):
            pt_distance = np.sum(((self.points[i] - track[i - 1])**2), axis=1) 
            pt_index = np.argmin(pt_distance)
            track[i] = self.points[i][pt_index]
        np.save("./trajectories/trajectory{}.npy".format(index), track)
