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
        inverse.calculate_contrasting_points()
    elif "vis" in args:
        inverse = InverseCylinder()
        inverse.visualize_tracks()
    else: 
        inverse = InverseCylinder()
        inverse.track_by_proximity()
        inverse.visualize_tracks()
        
        print("Usage: python main.py [direct|inverse]")
