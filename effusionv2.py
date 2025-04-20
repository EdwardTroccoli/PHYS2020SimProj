#!/usr/bin/env python3
# Written by Edward Troccoli, 2025
# Inspirtation from Louis Sharma, 2023
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from itertools import product

#---------------------Initalising Parameters-----------------------------------

N = 100 # particles in the simulation

W, H = 10, 10 # dimensions of the box, width x height

# position of the hole/apperture 
y_pos = H / 2     
hole_height = 1.0         
x_pos = W                

# general constants are set to be zero
T = 100.0  # temperature
m = 1.0  # mass                
k = 1.0  # boltzmann constant              
v0 = np.sqrt(k * T / m) # max velocity

# radius of each particle
radius = 0.1 

# track the time, timestep and total steps taken
current_time = 0.0
dt = 0.0001
steps = 400

# track the escaped particles, including the time at which they escape and
# the total number of escaped particles
escaped = np.zeros(N, dtype=bool)
escape_times = []
escape_count = 0

#---------------------Defining Ideal Gas Class-------------------------------

class IdealGas:

    def __init__(self, N, m, radius, W, H, v0, dt, steps):

        self.N = N 
        self.m = m 
        self.radius = radius 
        self.H = H 
        self.W = W 
        self.dt = dt 
        self.steps = steps 
        self.time = dt*steps 
        self.v0 = v0 

        #intialise random positions
        self.r = np.stack((
        np.random.uniform(radius, W - radius, N),
        np.random.uniform(radius, H - radius, N)
        ), axis=1)

        #intialise random positions
        vx = np.random.normal(0, v0, N)
        vy = np.random.normal(0, v0, N)
        self.v = np.stack((vx,vy), axis=1)
    
    def lennard_jones_potential(self):
        """Calculate the force due to a Lennard Jones potential of the form V = 4\varepsilon((\sigma/r)^12-(\sigma/r)^6)"""
        return -4*((1/self.r)**6-(1/self.r)**12)

    def check_collisions(self):
        """Checks for particle-particle collisions and particle-wall collisions.
        If collisions are found, update the velocities of the particles that are colliding, accounting for 
        the conservation of energy and momentum."""
        force = self.lennard_jones_potential()
        # advance the position through the Verlet algorithm and Lennard Jones force/potential
        r_next = self.r + self.v*self.dt + (force/2*self.m)*(self.dt)**2
        #check for collisions with the walls
        # Left wall
        self.v[self.r[:, 0] < self.radius, 0] *= -1
        # Bottom wall
        self.v[self.r[:, 1] < self.radius, 1] *= -1
        # Top wall
        self.v[self.r[:, 1] > self.H - self.radius, 1] *= -1
        # Right wall — reflect only if outside the slit
        right_wall_hit = (self.r[:, 0] > self.W - self.radius) & (
        (self.r[:, 1] < y_pos - hole_height / 2) | (self.r[:, 1] > y_pos + hole_height / 2)
        )
        self.v[right_wall_hit, 0] *= -1


        #check for collisions between particles
        n_particles = len(self.r)
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                if np.linalg.norm(r_next[i] - r_next[j]) < 2*self.radius:

                    rdiff = self.r[i] - self.r[j] #vector between particle 1 and particle 2
                    vdiff = self.v[i] - self.v[j]
                    self.v[i] = self.v[i] - rdiff.dot(vdiff)/rdiff.dot(rdiff)*rdiff #update velocity of particle i
                    self.v[j] = self.v[j] + rdiff.dot(vdiff)/rdiff.dot(rdiff)*rdiff #update velocity of particle j

    def check_escaped(self, escaped_mask, escape_times, escape_count, current_time):
        # Check for escape condition (particles past the right wall in the hole range and not yet escaped)
        escape_condition = (
            (self.r[:, 0] > self.W) &  # beyond right wall
            (self.r[:, 1] > y_pos - hole_height / 2) &
            (self.r[:, 1] < y_pos + hole_height / 2) &
            (~escaped_mask)
        )

        # Determine which particles escaped
        newly_escaped = np.sum(escape_condition)
        escape_count += newly_escaped
        escape_times.extend([current_time] * newly_escaped)

        # Keep only unescaped particles
        keep_mask = ~escape_condition
        self.r = self.r[keep_mask]
        self.v = self.v[keep_mask]

        # Update the escaped mask (remove escaped entries to stay aligned with r and v)
        escaped_mask = escaped_mask[keep_mask]

        return escaped_mask, escape_times, escape_count

    def step(self):
        """Computes the positions at the next timestep."""
        self.check_collisions()
        force = self.lennard_jones_potential()
        self.r += self.v*self.dt + (force/2*self.m)*(self.dt)**2
        updated_force = self.lennard_jones_potential()
        self.v += ((updated_force+force)/(2*self.m))*self.dt

#---------------------Creating the animation------------------------------------------

gas = IdealGas(N, m, radius, W, H, v0, dt, steps)    

fig, ax = plt.subplots()
scat = ax.scatter(gas.r[:, 0], gas.r[:, 1], s=10)
time_text = ax.text(0.02, 1.02, 'Time: '+str(current_time), transform=ax.transAxes)
escape_text = ax.text(0.02, 1.10, 'Number of escaped particles: '+str(escape_count), transform=ax.transAxes)
ax.set_xlim(0, gas.W)
ax.set_ylim(0, gas.H)

# visualisation of the hole, placed in the middle of the right barrier
ax.plot([W, W], [y_pos - hole_height/2, y_pos + hole_height/2], color='red', linewidth=10)

def update(frame):
    global current_time, escape_count, escaped, escape_times

    # Step the simulation
    gas.step()
    current_time += dt

    # Check for escaped particles
    escaped, escape_times, escape_count = gas.check_escaped(escaped, escape_times, escape_count, current_time)

    # Update the scatter plot with active particles
    scat.set_offsets(gas.r)

    # Update the time and count display
    time_text.set_text(f"Time: {current_time:.2f}")
    escape_text.set_text(f"Number of escaped particles: {escape_count}")

    return scat, time_text, escape_text

# run the animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=1, blit=False)
plt.show()

    