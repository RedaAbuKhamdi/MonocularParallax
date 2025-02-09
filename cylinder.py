import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from skimage import io

FRAME_WIDTH = 1000
FRAME_HEIGHT = 1000

class Cylinder:
    def __init__(
            self, 
            radius : float, 
            height : float, 
            frames_per_revolution : int = 60, 
            revolutions : int = 15,
            number_of_pts : int = 60,
            invert : bool = False):
        self.radius = radius
        self.height = height
        self.frames_per_revolution = frames_per_revolution
        self.revolutions = revolutions
        self.number_of_pts = number_of_pts
        self.invert = invert
        self.calculate_points()
    
    def calculate_points(self) -> None:
        """
        Sets self.points to a numpy array of shape (number_of_pts, 3) with random points on the cylinder.
        """
        self.points: np.ndarray = np.zeros((self.number_of_pts, 3))
        step = 4 * np.pi / self.number_of_pts
        for i in range(self.number_of_pts):
            x: float = self.radius * np.sin(i * step)
            z: float = np.random.uniform(-self.height / 2 , self.height / 2 )
            y: float = self.radius * np.cos(i * step)
            self.points[i, 0] = x
            self.points[i, 1] = y
            self.points[i, 2] = z

    def calculate_central_projection(self, 
                                     focal_length: float,
                                     point : np.ndarray) -> np.ndarray:
        projection = np.zeros(2)
        coefficient = focal_length / -point[0]
        projection[0] = coefficient * point[1]
        projection[1] = coefficient * point[2]
        return projection
    
    def shift_coordinate_system(self, center : np.ndarray, points : np.ndarray) -> np.ndarray:
        return points - np.tile(center, (points.shape[0], 1))  
    
    def create_frame(self, center : np.ndarray, focal_length : float) -> np.ndarray:
        points = self.shift_coordinate_system(center, self.points)
        frame = np.zeros((self.number_of_pts, 2))
        for i in range(self.number_of_pts):
            frame[i] = self.calculate_central_projection(focal_length, points[i])
        return frame

    def rotate(self):
        angle = 2 * np.pi / self.frames_per_revolution
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        self.points = np.dot(self.points, rotation_matrix)

    def create_image(self, focal_length : float, distance_from_camera : float) -> np.ndarray:

        center_point = np.array([focal_length + distance_from_camera, 0, 0])
        frames = []
        for i in range(self.revolutions * self.frames_per_revolution):
            frame = self.create_frame(center_point, focal_length)
            self.rotate()
            frames.append(frame)
        return np.array(frames)
    def save_image(self, images : np.ndarray):
        sides_size = np.ceil((np.max(images) - np.min(images)) * 1.2)
        factor = 100
        result = np.zeros((images.shape[0], FRAME_WIDTH, FRAME_HEIGHT), dtype=np.uint8)
        for i in range(images.shape[0]):
            adjustet_points = self.shift_coordinate_system(np.array([-sides_size / 2, sides_size / 2]), images[i])
            for j in range(adjustet_points.shape[0]):
                result[i, np.round(adjustet_points[j, 1] * factor).astype(np.int16), np.round(adjustet_points[j, 0] * factor).astype(np.int16)] = 254
        io.imsave("cylinder.tiff", result)
        for i in range(images.shape[0]):
            io.imsave(f"./frames/cylinder{i}.png", result[i])
    def visualize(self, images : np.ndarray):
        fig, ax = plt.subplots()
        fig.set_label("Cylinder")
        x_limit = (self.radius) * 1.2
        y_limit = (self.height / 2) * 1.2
        ax.set_facecolor('black' if self.invert else 'white')
        def update(n):
            ax.cla()
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)
            im =  sns.scatterplot(
                x = images[n, :, 0], 
                y = images[n, :, 1],
                legend = False,
                s = 0.2,
                color = "white" if self.invert else "black")
            im.set(xticklabels = [], yticklabels = [])
            im.tick_params(bottom=False, left = False)
            return im
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=images.shape[0] - 1, interval=40)
        plt.show()

    def calculate_angle(self, point1, point2, center):
        vector1 = point1 - center
        vector2 = point2 - center
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    def calculate_wurf(self, points, index : int, start : int = 0):
        center = np.array([0, 0])
        sin_alpha = np.sin(self.calculate_angle(points[start][index], points[start - 1][index], center))
        sin_beta = np.sin(self.calculate_angle(points[start - 1][index], points[start - 2][index], center))
        sin_gamma = np.sin(self.calculate_angle(points[start - 2][index], points[start - 3][index], center))
        return float(sin_alpha * sin_gamma / ((sin_alpha + sin_beta + sin_gamma) * sin_beta))

    def avg_wurf(self, points):
        wurf_mean = np.zeros(len(points))
        for i in range(len(points)):
            for j in range(3):
                wurf_mean[i] += self.calculate_wurf(points, j, i)
            wurf_mean[i] /= 3
        return np.mean(wurf_mean)
        


    