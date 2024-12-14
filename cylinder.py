import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation

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
        step = 2 * self.radius / (self.number_of_pts + 1)
        for i in range(self.number_of_pts):
            x: float = np.random.uniform(-self.radius + step * i, -self.radius + step * (i + 1))
            z: float = np.random.uniform(-self.height / 2 , self.height / 2 )
            y: float = np.random.choice(np.array([-1, 1])) * np.sqrt(self.radius ** 2 - x ** 2)
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
    def visualize(self, images : np.ndarray, save = False):
        fig, ax = plt.subplots()
        fig.set_label("Cylinder")
        x_limit = (self.radius) * 1.2
        y_limit = (self.height / 2) * 1.2
        ax.set_facecolor('black' if self.invert else 'white')
        def update(n):
            ax.clear()
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)
            plt.margins(0,0)
            return ax.scatter(images[n, :, 0], images[n, :, 1], color='white' if self.invert else 'black', s = 0.02)
        update(0)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=images.shape[0] - 1, interval=150)
        plt.show()
        if save:
            pass
        


    