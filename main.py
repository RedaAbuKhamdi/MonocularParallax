import sys
from cylinder import Cylinder
from reconstruction import InverseCylinder
import estimator
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from tqdm import tqdm


if __name__ == "__main__":
    args = sys.argv[1:]
    if "direct" in args:
        cylinder = Cylinder(5, 10, number_of_pts=300, frames_per_revolution=800, revolutions=1, invert=True)
        images = cylinder.create_image(5, 5)
        cylinder.visualize(images)
        cylinder.save_image(images) 
    elif "inverse" in args:
        inverse = InverseCylinder()
        command = input("Type a command: [visualize tracks|search ellipse]")
        if command == "visualize tracks":
            inverse.visualize_tracks()
        elif command == "search ellipse":
            for track in tqdm(inverse.get_tracks()):
                ellipse_model, params = inverse.calculate_ellipse(track)
                if params is not None:
                    print("For track {} found ellipse with params {}".format(track, params))
                    inverse.calculate_angle_ellipse(ellipse_model)
                    inverse.visualize_ellipse(params, track)
        else: 
            if (input("Track by proximity? [y/n] ") == "y"): inverse.track_by_proximity()
            else: inverse.track_by_wurf()
    else: 
        print("Usage: python main.py [direct|inverse]")
