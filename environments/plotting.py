import numpy as np
import matplotlib.pyplot as plt

def normalize_0_1(x):
    return (x - x.min()) / (x.max() - x.min())

def plotR(R, h, w, title="", grid=None):
    
    R = np.asarray(R[:-1])
    
    if grid:
        R2 = []
        i = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                R2.append(R[i])
                i += 1
        R = np.asarray(R2)
    
    plt.imshow(R.reshape(h,w))
    plt.title(title)
    
def compare_grid_data(R1, R2, h, w, title1="Original", title2="Recovered", suffix="Reward", grid=None):
    
    plt.subplot(1,2,1)
    plotR(R1, h, w, title1+" "+suffix, grid)
    plt.subplot(1,2,2)
    plotR(R2, h, w, title2+" "+suffix, grid)