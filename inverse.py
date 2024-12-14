import numpy as np
from matplotlib import pyplot as plt

def create_images(images : np.ndarray, inverse : bool = True):
    
    x_limit = image * 1.2
    y_limit = (self.height / 2) * 1.2
    for i in range(images.shape[0]):
        fig, ax = plt.subplots()
        ax.set_facecolor('black' if self.invert else 'white')
        def update(n):
            ax.clear()
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)
            ax.set_xlabel('y')
            ax.set_ylabel('z')
            return ax.scatter(images[n, :, 0], images[n, :, 1], color='white' if self.invert else 'black', s = 0.02)
        update(0)