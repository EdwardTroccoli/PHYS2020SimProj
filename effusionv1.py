#!/usr/bin/env python3
# Written by Edward Troccoli, 2025
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#---------------------Initalising Parameters-----------------------------------

N = 100 # particles in the simulation

W, H = 10, 10 # dimensions of the box, width x height

# position of the hole/apperture 
y_pos = H / 2     
hole_height = 1.0         
x_pos = W                

# general constants are set to be zero
T = 10.0  # temperature
m = 1.0  # mass                
k = 1.0  # boltzmann constant              
sigma = np.sqrt(k * T / m) # max velocity

# radius of each particle
radius = 0.1 

# set particle positions such that they can only just touch the walls,
# and velocities to be random between the max velocity we define above
x = np.random.uniform(radius, W - radius, N)
y = np.random.uniform(radius, H - radius, N)
vx = np.random.normal(0, sigma, N)
vy = np.random.normal(0, sigma, N)

# set net momentum to be zero, equivalent to avg velocity being zero as
# the mass is the same for all of them
vx -= np.mean(vx)
vy -= np.mean(vy)

# track the escaped particles, including the time at which they escape and
# the total number of escaped particles
escaped = np.zeros(N, dtype=bool)
escape_times = []
escape_count = 0

# track the time, timestep and total steps taken
current_time = 0.0
dt = 0.005
steps = 400

#construction of the axis
fig, ax = plt.subplots()
scat = ax.scatter(x, y, s=10)
time_text = ax.text(0.02, 1.02, 'Time: '+str(current_time), transform=ax.transAxes)
escape_text = ax.text(0.02, 1.10, 'Number of escaped particles: '+str(escape_count), transform=ax.transAxes)
ax.set_xlim(0, W)
ax.set_ylim(0, H)

# visualisation of the hole, placed in the middle of the right barrier
ax.plot([W, W], [y_pos - hole_height/2, y_pos + hole_height/2], color='red', linewidth=10)


#--------------------Defining Functions----------------------------------------     

# function to update each frame in the animation. Here I will update the position at each
# timestep and then it will check for collisions and if the particles have escaped and 
# correct accordingly
def update(frame):
    global x, y, vx, vy, escaped, escape_count, current_time

    current_time += dt

    # update position of the particles
    x += vx * dt
    y += vy * dt






