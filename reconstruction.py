import numpy as np
from matplotlib import animation, pyplot as plt
from os import listdir
from os.path import isfile, join
import re
from skimage import io
from skimage.measure import EllipseModel
from scipy import ndimage

EPSILON = 1e-2
CENTER_SEARCH_EPSILON = 1e-6
CONTRASTING_POINTS = 3

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class InverseCylinder:
    def __init__(self, inverse : bool = True):
        self.inverse = inverse
        self.images = io.imread("cylinder.tiff").astype(np.float16)
        self.extract_points()
        self.chosen_points = None
        self.points = None
    def extract_points(self):
        self.points = []
        print(self.images.shape)
        images = self.images.copy()
        for i in range(self.images.shape[0]):
            cylinder_points = np.argwhere(images[i] > 0 if self.inverse else images[i] == 0)
            self.points.append(cylinder_points)
        print("Number of points extracted: {}".format(len(self.points)))

    def find_points_for_center(self, center : np.ndarray, frames : np.ndarray):
        best_points = None
        for i in InverseCylinder.sort_by_distance(center, frames[0]):
            for j in InverseCylinder.sort_by_distance(center, frames[1]):
                for k in InverseCylinder.sort_by_distance(center, frames[2]):
                    for l in InverseCylinder.sort_by_distance(center, frames[3]):
                        wurf = self.calculate_angle_wurf(frames[0][i], frames[1][j], frames[2][k], frames[3][l], center)
                        if np.abs(wurf - 1/3) < CENTER_SEARCH_EPSILON:
                            best_points = np.array([frames[0][i], frames[1][j], frames[2][k], frames[3][l]])
                            return best_points
        return best_points

    def sort_by_distance(center : np.ndarray, points : np.ndarray):
        distances = np.full(points.shape[0], np.inf)
        for i, point in enumerate(points):
            distances[i] = np.linalg.norm(point - center)
        return np.argsort(distances)
        

    def track_point(self, center : np.ndarray):
        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        trajectory = np.zeros((self.images.shape[0], 2), dtype=np.int32)
        first_points = self.find_points_for_center(center, self.points)

        if first_points is None:
            raise Exception("No points found!")

        trajectory[0] = first_points[0]
        trajectory[1] = first_points[1]
        trajectory[2] = first_points[2]
        trajectory[3] = first_points[3]
        
        for i in range(4, self.images.shape[0]):
            found_point = False
            for j in range(self.points[i].shape[0]):
                wurf = self.calculate_angle_wurf(trajectory[i-3], trajectory[i-2], trajectory[i-1], self.points[i][j], center)
                print(wurf, j)
                if np.abs(wurf - 1/3) < EPSILON:
                    print(i)
                    trajectory[i] = self.points[i][j]
                    found_point = True
                    break
            if not found_point:
                raise Exception("No point was found for trajectory!")
        return trajectory

    def calculate_contrasting_points(self, n: int):
        chosen_points = []
        if self.points is None:
            raise Exception("No extracted points! Run extract_points() first.")
        step = 1
        x = self.images[0].shape[0] // 2 
        y = step
        print("x = {}, y = {}".format(x, y))
        for i in range(n):
            point_found = False
            while not point_found:
                try:
                    chosen_points.append(
                        self.track_point(np.array([x, y]))
                    )
                    point_found = True
                except: 
                    pass
                finally:
                    y += step
                if x >= self.images[0].shape[0] or y >= self.images[0].shape[1]:
                    raise Exception("No points found!")
                print("x = {}, y = {}".format(x, y))
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
    
    def calculate_angle_wurf(self, point1, point2, point3, point4, center):
        sin_alpha = np.sin(self.calculate_angle(point1, point2, center))
        sin_beta = np.sin(self.calculate_angle(point2, point3, center))
        sin_gamma = np.sin(self.calculate_angle(point3, point4, center))
        return float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))

    def avg_wurf(self):
        wurf_mean = np.zeros(len(self.points))
        for i in range(len(self.points)):
            for j in range(3):
                wurf_mean[i] += self.calculate_wurf(j, i)
            wurf_mean[i] /= 3
        return np.mean(wurf_mean)

    def calculate_speed(self):
        wurf_mean = 0
        angle_mean = 0
        n = 0
        ellipse_params = []
        ellipse_centeres = []
        for i in range(3):
            ellipse = EllipseModel()
            ellipse.estimate(np.array(self.chosen_points)[:, i])
            xc, yc, a, b, _ = ellipse.params
            ellipse_params.append((a, b))
            ellipse_centeres.append((xc, yc))
        for i in range(len(self.points)):
            for j in range(3):
                wurf = self.calculate_wurf(j, i)
                if (abs(wurf - 1/3) < EPSILON):
                    wurf_mean += wurf
                    a, b = ellipse_params[j]
                    print(a, b)
                    transformation = lambda point : np.array([point[0], point[1] * b / a])
                    # print(self.calculate_angle(transformation(self.chosen_points[i][j]), transformation(self.chosen_points[i-1][j]), transformation(np.array([self.images[i].shape[0] / 2, self.images[i].shape[1] / 2]))))
                    # print(self.calculate_angle(self.chosen_points[i][j],self.chosen_points[i-1][j], np.array([self.images[i].shape[0] / 2, self.images[i].shape[1] / 2])))
                    x_current = transformation(self.chosen_points[i][j])[0]
                    y_current = transformation(self.chosen_points[i][j])[1]
                    x_c_current = transformation(np.array([ellipse_centeres[j][0], ellipse_centeres[j][1]]))[0]
                    y_c_current = transformation(np.array([ellipse_centeres[j][0], ellipse_centeres[j][1]]))[1]
                    print((x_current - x_c_current)**2 + (y_current - y_c_current)**2)
                    print(max(a, b) ** 2)
                    angle_mean += self.calculate_angle(transformation(self.chosen_points[i][j]), transformation(self.chosen_points[i-1][j]), transformation(np.array([self.images[i].shape[0] / 2, self.images[i].shape[1] / 2])))
                    # angle_mean += self.calculate_angle(self.chosen_points[i][j], self.chosen_points[i-1][j], np.array([self.images[i].shape[0] / 2, self.images[i].shape[1] / 2]))
                    n += 1
        print(wurf_mean, angle_mean, n)
        return (wurf_mean / n, angle_mean / n)
    def visualize(self):
        if (self.chosen_points is None):
            chosen_points = self.calculate_contrasting_points(CONTRASTING_POINTS)
            self.chosen_points = chosen_points
        else:
            chosen_points = self.chosen_points
        print(chosen_points)
        fig, ax = plt.subplots()
        def update(n):
            ax.clear()
            slice_image = ndimage.binary_dilation(self.images[n].copy().astype(np.uint16), ndimage.generate_binary_structure(2, 3))
            slice_image = slice_image.astype(np.uint16)
            slice_image *= 60
            contrasting_slice_image = np.zeros((self.images[n].shape[0], self.images[n].shape[1]), dtype=np.uint16)
            for pt_index in range(CONTRASTING_POINTS):
                contrasting_slice_image[chosen_points[pt_index][n][0], chosen_points[pt_index][n][1]] = 1
            contrasting_slice_image = ndimage.binary_dilation(contrasting_slice_image, ndimage.generate_binary_structure(2, 3))
            contrasting_slice_image = contrasting_slice_image.astype(np.uint16)
            contrasting_slice_image *= 254
            ax.imshow(slice_image + contrasting_slice_image, cmap='gray', vmin=0, vmax=255) 
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.images.shape[0] - 1, interval=150)
        plt.show()

        