""" 
2D SPH - 21 January 2021
========================

This is a proof-of-concept for testing collision-less Smoothed Particle
Hydrodynamics in 2D before being brought into a more robust programming
language in 3D. This is also a practice of visualization and handling data
as Python will how the final simulation software communicates with the
visualization methods (e.g. MatPlotLib or Blender).

Dependencies
------------
 - SciPy libraries : for Python numerical methods and data visualization
 - ffmpeg : for compiling frames into videos for final viewing

References
----------
.. [1] Giordano, N. J., &amp; Nakanishi, H. (2006). Ordinary Differential 
       Equations with Initial Values. In Computational Physics (pp. 456-463). 
       New Delhi: Dorling Kindersley.
.. [2] Monaghan, J. (1992). Smoothed Particle Hydrodynamics. Annu. Rev. Astron. 
       Astrophys, 30: 543-74.
.. [3] Rosswog, S. (2009). Astrophysical Smoothed Particle Hydrodynamics. 
       Elsevier.
"""
# Author: N Nguyen

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Global control functions ====================================================

WRITE_TO_FILES = False

# Domain Parameters ===========================================================
#   The simulation is run at 24fps, so the timesteps equate to 
#   (24 * STEPS_PER_FRAME) updates per second

DOMAIN_X = 4.
DOMAIN_Y = 4.
TIMESTEPS = 1440
STEPS_PER_FRAME = 10

# Simulation Parameters =======================================================

PARTICLE_COUNT = 128
PARTICLE_SIZE = 0.1
PARTICLE_MASS = 0.1
SMOOTHING_LENGTH = 0.1
SOFTENING_FACTOR = 0.1
G = 0.01

# Initialization ==============================================================

# Initialize the particle positions to be evenly spread rotationally
part_pos = np.zeros((PARTICLE_COUNT, 2))
rand = 2. * np.pi * np.random.rand(PARTICLE_COUNT)
part_pos[:,0] = np.cos(rand)
part_pos[:,1] = np.sin(rand)
part_pos *= (0.4 * DOMAIN_X) * np.random.rand(PARTICLE_COUNT, 1)

# Initialize the velocities to move tangentially to the center point
part_vel = np.zeros_like(part_pos)
part_vel[:,0] = np.sin(rand)
part_vel[:,1] = -np.cos(rand)
part_vel = 0.25 * np.multiply(np.random.rand(PARTICLE_COUNT, 1), part_vel)

# Initialize an acceleration array (This is likely unnecessary)
part_acc = np.zeros((PARTICLE_COUNT, 2))

# Functions and Operation =====================================================

# Create a unique directory for storing the frames
stamp_time = time.localtime()
timestamp = time.strftime('%m-%d-%Y_%H%M',stamp_time)
directory = "./SPH_2D_{}/".format(timestamp)
if(WRITE_TO_FILES) : os.makedirs(directory,exist_ok=True)

# Initialize the plot and save the first frame
fig, ax = plt.subplots()
ax.set_xlim(-DOMAIN_X / 2., DOMAIN_X / 2.)
ax.set_ylim(-DOMAIN_Y / 2., DOMAIN_Y / 2.)
ax.set_aspect('equal')
ax.grid()
scatter = ax.scatter(part_pos[:,0], part_pos[:,1], s=2)
if(WRITE_TO_FILES) : plt.savefig('{}SPH_2D_{:04d}.png'.format(directory,0))

def weight_func(r, h) :
    """
    Interpolates the smoothing of particle values into the domain

    The interpolation function is responsible for the smoothing of the 
    particles' interactions. In this case, the interpolation function is a 
    normalized Gaussian with the h parameter representing the standard 
    deviation, σ. One potential downside of this choice is that the Gaussian is
    non-zero even at far distances so there is no culling in large systems.

    Parameters
    ----------
    r : float
        The distance from a point to evaluate the weight from
    h : float
        The smoothing length of the function, in this case the std. dev.

    Returns
    -------
    W : float
        The weight at the given distance
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (r ** 2) / (h ** 2)) / h

def density(r, particle_positions, mass) :
    """
    Calculates the density at the given point using the SPH weighting method

    Parameters
    ----------
    r : array_like
        The coordinate to evaluate the density at
    particle_positions : array_like
        The array of particle positions contributing to density
    mass : float
        The mass of each particle
    
    Returns
    -------
    rho : float
        The density at the given point
    """
    return np.multiply(mass * \
        weight_func(particle_positions, SMOOTHING_LENGTH))

# Energy Functions
def kinetic_energy(particle_vel, mass) :
    """
    Calculates the total kinetic energy of the system

    Parameters
    ----------
    particle_vel : array_like
        The array of velocity vectors for each particle
    mass : float
        The mass per particle

    Returns
    -------
    K : float
        The kinetic energy from the given particles
    """
    return 0.5 * mass * np.sum(particle_vel ** 2)

def grav_energy(particle_positions, mass) :
    """
    This calculates the total gravitational energy of the system
    
    Parameters
    ----------
    particle_positions : array_like
        The array of particle positions contributing to the potential
    mass : float
        The mass per particle
    
    Returns
    -------
    P : float
        The total gravitational potential energy of the system

    Notes
    -----
    This function does not use the weighting function; rather, it treats each
    particle as a point mass.
    """
    energy = 0.
    for partID in range(0, np.shape(particle_positions)[0]) :
        r = particle_positions[partID] - particle_positions[partID:]
        r = np.linalg.norm(r, axis=1)
        nonzero_inds = np.nonzero(r)
        r[nonzero_inds] = np.reciprocal(r[nonzero_inds])
        energy += np.sum(r)
    return -energy * G * (mass ** 2)

def grav_potential(r, particle_positions, mass) :
    """
    This calculates the graviational potential 
    
    Parameters
    ----------
    r : array_like
        The position vector at which to evaluate the potential.
    particle_positions : array_like
        The array of particle positions contributing to the potential
    mass : float
        The mass per particle
    
    Returns
    -------
    P : float
        The gravitational potential energy at the given point

    Notes
    -----
    This function does not use the weighting function; rather, it treats each
    particle as a point mass.
    """
    invDist = r - particle_positions
    invDist = np.linalg.norm(invDist, axis=1)
    nonzero_inds = np.nonzero(invDist)
    invDist[nonzero_inds] = np.reciprocal(invDist[nonzero_inds])
    return -G * mass * np.sum(invDist)

# Dynamics Functions
def grav_acc(r, particle_positions, mass) :
    """
    Calculates the gravitational acceleration at a point

    Parameters
    ----------
    r : array_like
        The position vector at which to evaluate the potential.
    particle_positions : array_like
        The array of particle positions contributing to the potential
    mass : float
        The mass per particle
    
    Returns
    -------
    P : array_like
        The total gravitational acceleration (vector)

    Notes
    -----
    This function does not use the weighting function; rather, it treats each
    particle as a point mass.
    """
    rDel = r - particle_positions
    invCube = np.linalg.norm(rDel, axis=1)
    invCube = (invCube ** 2) + (SMOOTHING_LENGTH ** 2)
    invCube = np.power(invCube, 1.5)
    nonzero_inds = np.nonzero(invCube)
    invCube[nonzero_inds] = np.reciprocal(invCube[nonzero_inds])
    return -G * mass * np.sum(np.multiply(invCube.reshape(invCube.size, 1), \
        rDel), axis=0)

def ode_FE(part_pos, part_vel) :
    """
    Perform the simulation using the forward Euler method

    Parameters
    ----------
    part_pos : array_like
        The initial particle positions
    part_vel : array_like
        The initial velocity positions
    
    Notes
    -----
    If WRITE_TO_FILES is set to true, this will save the visualization at each
    frame and compile it to mp4 at the end
    """
    # Iterate over each frame
    for t in range(1,TIMESTEPS):
        # Iterate over each substep
        for _ in range(0, STEPS_PER_FRAME) :
            # Update the positions
            part_pos += part_vel / (24. * STEPS_PER_FRAME)
            
            # Clear the acceleration of each particle then recalculate
            part_acc = np.zeros_like(part_vel)
            for partID in range(0, np.shape(part_pos)[0]) :
                part_set = np.delete(part_pos, partID, 0)
                part_acc[partID] += grav_acc(part_pos[partID], \
                    part_set, PARTICLE_MASS)
            
            # Update the velocities
            part_vel += part_acc / (24. * STEPS_PER_FRAME)
            
        # Save the frame to file if applicable
        scatter.set_offsets(part_pos)
        if(WRITE_TO_FILES): 
            plt.savefig('{}SPH_2D_{:04d}.png'.format(directory, t))

    # Compile the frames to mp4 using ffmpeg
    if(WRITE_TO_FILES) : 
        os.system("ffmpeg -r 24 -f image2 -i {}SPH_2D_%04d.png -vcodec \
            libx264 -crf 25 -pix_fmt yuv420p SPH_2D_FE_{}.mp4".format( \
                directory, timestamp))

def ode_LF(part_pos, part_vel) :
    """
    Perform the simulation using the leap-frog (Kick-Drift-Kick) method

    Parameters
    ----------
    part_pos : array_like
        The initial particle positions
    part_vel : array_like
        The initial velocity positions
    
    Notes
    -----
    If WRITE_TO_FILES is set to true, this will save the visualization at each
    frame and compile it to mp4 at the end
    """
    # Iterate through all the frames
    for t in range(1,TIMESTEPS):
        # Iterate over each substep of the frame
        for _ in range(0, STEPS_PER_FRAME) :
            # Clear the acceleration of each particle then recalculate
            part_acc = np.zeros_like(part_vel)
            for partID in range(0, np.shape(part_pos)[0]) :
                part_set = np.delete(part_pos, partID, 0)
                part_acc[partID] = grav_acc(part_pos[partID], part_set, \
                    PARTICLE_MASS)

            # Kick - Perform the leap on the velocity
            part_vel += part_acc / (48. * STEPS_PER_FRAME)

            # Drift - Move the particle
            part_pos += part_vel / (24. * STEPS_PER_FRAME)
            
            # Kick - Finish the leap-frog
            part_vel += part_acc / (48. * STEPS_PER_FRAME)

        # Save the frame to file if applicable
        if(WRITE_TO_FILES): 
            scatter.set_offsets(part_pos)
            plt.savefig('{}SPH_2D_{:04d}.png'.format(directory, t))

    # Compile the frames to mp4 using ffmpeg
    if(WRITE_TO_FILES) : 
        os.system("ffmpeg -r 24 -f image2 -i {}SPH_2D_%04d.png -vcodec \
            libx264 -crf 25 -pix_fmt yuv420p SPH_2D_LF_{}.mp4".format( \
                directory, timestamp))

# TODO: Modify the steps to use simultaneous updates so that each particle is
#   interacting with particles on the same half-step and whole step for
#   more accurate integration
def ode_RK4(part_pos, part_vel) :
    """
    Perform the simulation using a 4th-order explicit Runge-Kutta method

    WARNING: 
        This method has large errors for long-term energy conservation and
        is only meant for testing Runge-Kutta implementation

    Parameters
    ----------
    part_pos : array_like
        The initial particle positions
    part_vel : array_like
        The initial velocity positions
    
    Notes
    -----
    If WRITE_TO_FILES is set to true, this will save the visualization at each
    frame and compile it to mp4 at the end.
    """
    # Iterate through all the frames
    for t in range(1, TIMESTEPS):
        # Iterate through each substep of the frame
        for _ in range(0, STEPS_PER_FRAME) :
            # Copy the old positions so all the calculations are performed on 
            #   the same data set
            part_pos_old = part_pos.copy()

            # Iterate through all the particles to update
            for partID in range(0, np.shape(part_pos)[0]) :
                part_set = np.delete(part_pos_old, partID, 0)

                # Calculate k1: f(t, y)
                k1_pos = part_vel[partID]
                k1_vel = grav_acc(part_pos_old[partID], \
                    part_set, PARTICLE_MASS)
    
                # Calculate k2: f(t + h/2, y + k1 * h/2)
                k2_pos = part_vel[partID] + k1_vel / (48. * STEPS_PER_FRAME)
                k2_vel = grav_acc(part_pos_old[partID] + \
                    k1_pos / (48. * STEPS_PER_FRAME), part_set, PARTICLE_MASS)

                # Calculate k3: f(t + h/2, y + k2 * h/2)
                k3_pos = part_vel[partID] + k2_vel / (48. * STEPS_PER_FRAME)
                k3_vel = grav_acc(part_pos_old[partID] + \
                    k2_pos / (48. * STEPS_PER_FRAME), part_set, PARTICLE_MASS)

                # Calculate k4: f(t + h, y + k3 * h)
                k4_pos = part_vel[partID] + k3_vel / (24. * STEPS_PER_FRAME)
                k4_vel = grav_acc(part_pos_old[partID] + \
                    k4_pos / (24. * STEPS_PER_FRAME), part_set, PARTICLE_MASS)

                # Update the values with these RK coefficients
                part_pos[partID] += (k1_pos + (2. * k2_pos) + \
                    (2. * k3_pos) + k4_pos) / (144. * STEPS_PER_FRAME)
                part_vel[partID] += (k1_vel + (2. * k2_vel) + \
                    (2. * k3_vel) + k4_vel) / (144. * STEPS_PER_FRAME)

        # Save the frame to file if applicable
        if(WRITE_TO_FILES): 
            scatter.set_offsets(part_pos)
            plt.savefig('{}SPH_2D_{:04d}.png'.format(directory,t))   

    # Compile the frames to mp4 using ffmpeg
    if(WRITE_TO_FILES) : 
        os.system("ffmpeg -r 24 -f image2 -i {}SPH_2D_%04d.png -vcodec \
            libx264 -crf 25 -pix_fmt yuv420p SPH_2D_RK4_{}.mp4".format( \
                directory, timestamp))     

# Perform the actual simulation by calling the relevant solver
energy1 = grav_energy(part_pos, PARTICLE_MASS) + kinetic_energy(part_vel, \
    PARTICLE_MASS)
ode_LF(part_pos, part_vel)
energy2 = grav_energy(part_pos, PARTICLE_MASS) + kinetic_energy(part_vel, \
    PARTICLE_MASS)

# Print the energy results of the simulation
print("Initial Energy: {}".format(energy1))
print("Final Energy: {}".format(energy2))
print("Error: {0:+.3f}%".format(-100. * (energy2 - energy1) / energy1))

# END =========================================================================