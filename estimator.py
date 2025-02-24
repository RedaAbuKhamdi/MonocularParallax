from skimage.measure import EllipseModel
import numpy as np
import os
import re

def fit_ellipse(points : np.ndarray):
    model = EllipseModel()
    print("estimation success: {}".format(model.estimate(points)))
    return model

def choose_trajectory():
    trajectories = [filename for filename in os.listdir("./trajectories") if "trajectory" in filename]
    if len(trajectories) == 0:
        raise Exception("No trajectories found!")
    indicies = [re.findall(r'\d+', f)[0] for f in trajectories]
    print("Available trajectories: {}".format(
        "\n".join(indicies)
    ))
    choice = input("Type a trajectory number to choose it: ")
    if choice not in indicies:
        raise Exception("Invalid trajectory number!")
    trajectory = np.load("./trajectories/trajectory{}.npy".format(choice))
    print(trajectory)
    return trajectory

def try_all():
    trajectories = [filename for filename in os.listdir("./trajectories") if "trajectory" in filename]
    for filename in trajectories:
        trajectory = np.load("./trajectories/{}".format(filename))
        model = EllipseModel()
        if model.estimate(trajectory):
            print(filename)
            return model, trajectory
    return None