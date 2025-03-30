import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#---------------------Initalising Parameters-----------------------------------

# This is number of particles in the experiment
N = 100

# Dimensions of the box will be a 10x10 box.     
W, H = 10, 10           

# Position of the hole  
hole_y_center = H / 2     
hole_height = 1.0         
hole_x = W                


# Setting some parameters to unity
T = 1.0
m = 1.0                  
k = 1.0                  
sigma = np.sqrt(k * T / m)
# Radius of each particle
radius = 0.1 

#--------------------Defining Functions----------------------------------------     

def total_kinetic_energy():
    return 0.5 * m * np.sum(vx[~escaped]**2 + vy[~escaped]**2)
def handle_collisions():

def update(frame):





