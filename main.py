import numpy as np
from cylinder import Cylinder


if __name__ == "__main__":
    cylinder = Cylinder(4, 10, number_of_pts=250, invert=True)
    images = cylinder.create_image(5, 5)
    cylinder.visualize(images, True)
    print(images[0])