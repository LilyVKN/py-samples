# py-samples
This is a collection some of my miscellaneous python samples/tools I've
developed for easy access and reference. I'll try to add some details on each
as I upload them as well as some sample outputs.

#### Quad- and Octree
These programs <sup>[[1]](./Quad-Tree.py), [[2]](./Oct-Tree.py)</sup> create a
random distribution of particles in two- and three-dimensional domains
respectively then begin to partition them according to the octree method first
described by Barnes & Hut (1986) for N-body astrophysical simulations. The
algorithm subdivides the domain repeatedly until there is only one particle per
subdivision. Empty subdivisions are discarded. This algorithm allows for a much
faster O(N log N) approach to simulating gravitational N-body scenarios due to
the clustering of particles leaving large swaths of the domain empty. I have
compiled some of the outputs using a visualization of the subdivisions at each
iteration [here (YouTube)](https://youtu.be/HUZ363rNHh0)

#### Smoothed Particle Hydrodynamics (SPH)
These programs <sup>[[1]](./SPH_2D.py), [[2]](./SPH_3D.py)</sup> calculates the
trajectories of gravitation, N-body simulations in 2- and 3-dimensional domains
using basic integrators. Available are 3 different integrators: forward Euler,
leap-frog, and 4th-order Runge-Kutta. Due to the nature of the calculations for
gravitional simulations, the leap-frog integrator maintains the most accurate
energy long-term. I have compiled some of the outputs displaying the particles'
motion [here](https://youtu.be/CHpxrZNSei4) and
[here](https://youtu.be/GoBorlSVHuc). These were developed as proofs-of-concept
and tests before shifting to Fortran using MPI parallelization.

#### Lattice Boltzmann Method (LBM)
This program <sup>[[1]](./LBM_D2Q9_main.py)</sup> uses the Lattice Boltzmann
method to calculate fluid flow, essentially solving for the Boltzmann
statistical equations for fluids to reproduce the Navier-Stokes equations. It
is for single-phase flow in a very basic wind-tunnel style domain. It has some
artifacts which make the simulation unstable at high velocities due to the
boundary condition implementations. A sample output of a radial shockwave can
be seen [here](https://youtu.be/4gwGsLY80SA). This was developed to be brought
into a more robust, 3D package in C++/Fortran