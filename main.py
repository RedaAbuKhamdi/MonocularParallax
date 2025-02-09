import sys
from cylinder import Cylinder
from reconstruction import InverseCylinder



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
        inverse.visualize()
        print(inverse.avg_wurf())
    else: 
        print("Usage: python main.py [direct|inverse]")
