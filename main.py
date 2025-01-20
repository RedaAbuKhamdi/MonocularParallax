import numpy as np
from cylinder import Cylinder
from reconstruction import InverseCylinder



if __name__ == "__main__":
    cylinder = Cylinder(4, 10, number_of_pts=40, frames_per_revolution=300, revolutions=2, invert=True)
    images = cylinder.create_image(5, 5)
    cylinder.visualize(images)
    cylinder.save_image(images)
    print(cylinder.avg_wurf(images))  
    inverse = InverseCylinder()
    inverse.extract_points()
    inverse.visualize()
    print(inverse.avg_wurf())
