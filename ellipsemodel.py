import numpy as np
import cv2
from matplotlib import pyplot as plt
from numba import jit
@jit
def fit_ellipse(track, x_c, a, min_y, max_y):
    best_b = None
    best_y_c = None
    best_metric = np.inf
    for y_c in np.linspace(min_y, max_y, 2000):
        for b in np.linspace(0.1, 100, 2000):
            max_metric = 0
            for pt in track:
                x = pt[0]
                y = pt[1]
                res = ((x - x_c) / a )**2 + ((y - y_c) / b)**2
                if abs(res - 1) > max_metric:
                    max_metric = abs(res - 1)
            if  max_metric < min(10e-3, best_metric):
                best_metric = max_metric
                best_b = b
                best_y_c = y_c
    return best_b, best_y_c

class EllipseModel():
    def __init__(self, track : np.ndarray):
        self.track = track
        self.params = None
    def fit_ellipse(self, x_c, a, min_y, max_y):
        best_b, best_y_c = fit_ellipse(self.track, x_c, a, min_y, max_y)
        if best_b is not None:
            self.params = (x_c, best_y_c, a, best_b)
            return x_c, best_y_c, a, best_b
        return None
    def parametric_ellipse(self, t: np.array):
        result = np.zeros((t.shape[0], 2))
        result[:,0] = self.params[0] + self.params[2] * np.cos(t)
        result[:,1] = self.params[1] + self.params[3] * np.sin(t)
        return result
    def get_homography_matrix(self):
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ellipse_pts = self.parametric_ellipse(angles)
        circle_pts = np.zeros((angles.shape[0], 2))
        circle_pts[:,0] = np.cos(angles)
        circle_pts[:,1] = np.sin(angles)
        H, _ = cv2.findHomography(ellipse_pts, circle_pts)
        return H

    def calculate_angle_homography(self):
        if (self.params is None):
            print("Fit ellipse first")
            return None
        H = self.get_homography_matrix()
        x_c, y_c, a, b = self.params
        angles = np.linspace(0, 2 * np.pi, 50)
        ellipse_pts = self.parametric_ellipse(angles)
        ellipse_pts = cv2.perspectiveTransform(ellipse_pts.reshape(-1, 1, 2), H)
        track = cv2.perspectiveTransform(self.track.reshape(-1, 1, 2), H)
        plt.scatter(track[:,0, 0], track[:,0, 1], color = "red", s=16)
        plt.scatter(ellipse_pts[:,0, 0], ellipse_pts[:, 0, 1], color = "blue", s=16)
        plt.show()
        track = track.reshape(-1, 2)
        ellipse_pts = ellipse_pts.reshape(-1, 2)
        for i in range(1, track.shape[0]):
            pt0 = track[i - 1]
            pt1 = track[i]
            print("Angle = {} radians".format(
                np.arccos(
                    np.dot(pt0, pt1) / (np.linalg.norm(pt0) * np.linalg.norm(pt1))
                ) * (a / b)
            ))
    def find_true_center(self):
        """Estimate the true center using conjugate diameters."""
        if self.params is None:
            print("Fit ellipse first")
            return None

        x_c, y_c, a, b = self.params

        # Define two conjugate directions
        p, q = 1, 0  # Major axis direction
        p_prime = - (b**2 / a**2) * p
        q_prime = q  # Conjugate diameter direction

        # Choose points along these directions
        y1, z1 = x_c + a, y_c  # Along major axis
        y2, z2 = x_c - a, y_c
        y3, z3 = x_c, y_c + b  # Along minor axis
        y4, z4 = x_c, y_c - b

        # Compute midpoints of conjugate chords
        M1 = np.array([(y1 + y2) / 2, (z1 + z2) / 2])
        M2 = np.array([(y3 + y4) / 2, (z3 + z4) / 2])

        # Solve for the intersection of the midpoints' line
        A = np.array([[M2[1] - M1[1], M1[0] - M2[0]]])
        B = np.array([[M1[1] * M2[0] - M2[1] * M1[0]]])

        try:
            true_center = np.linalg.lstsq(A, B, rcond=None)[0].flatten()
        except np.linalg.LinAlgError:
            true_center = np.array([x_c, y_c])  # If numerical issue, return fitted center

        return true_center

    def calculate_angle_original(self):
        if self.params is None:
            print("Fit ellipse first")
            return None
        H = self.get_homography_matrix()
        
        true_center = self.find_true_center()
        if true_center is None:
            print("Could not determine the true center")
            return None
        
        angles = np.linspace(0, 2 * np.pi, 50)
        ellipse_pts = self.parametric_ellipse(angles)
        track = self.track

        for i in range(1, track.shape[0]):
            # Compute vectors relative to the corrected center
            pt0 = track[i - 1] - true_center
            pt1 = track[i] - true_center

            print("Angle = {} radians".format(
                np.arccos(
                    np.dot(pt0, pt1) / (np.linalg.norm(pt0) * np.linalg.norm(pt1))
                )
            ))
