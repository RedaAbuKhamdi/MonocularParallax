import sys
from cylinder import Cylinder
from reconstruction import InverseCylinder
import estimator
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel


if __name__ == "__main__":
    args = sys.argv[1:]
    if "direct" in args:
        cylinder = Cylinder(4, 10, number_of_pts=60, frames_per_revolution=500, revolutions=2, invert=True)
        images = cylinder.create_image(5, 5)
        cylinder.visualize(images)
        cylinder.save_image(images) 
    elif "inverse" in args:
        inverse = InverseCylinder()
        inverse.extract_points()
        inverse.ellipse_try()
        inverse.visualize_by_input()
    elif "ellipse" in args:
        inverse = InverseCylinder()
        inverse.extract_points()
        tracks = np.load("./trajectories/trajectory90.npy")
        model = EllipseModel()
        model.estimate(tracks)
        plt.scatter(tracks[:, 0], tracks[:, 1], color="blue")
        preds = model.predict_xy(np.linspace(0, 2 * np.pi, 15))
        plt.scatter(preds[:, 0], preds[:, 1], color="red")
        plt.show()
    else: 
        print("Usage: python main.py [direct|inverse]")
