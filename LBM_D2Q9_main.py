from math import sqrt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import time

start_time = time.perf_counter()

#LBM ALGORITHM GLOBALS
D = 2   #D2
Q = 9   #Q9

#Domain Constants
DOMAIN_X = 480
DOMAIN_Y = 120
timesteps = 1200
nu = 1./144.        #Kinematic Viscosity    m^2/s
tau = 1.            #Relaxation Time        s
del_t = 1./24.      #Lattice Time Step      s
del_x = 1./24.      #Lattice Spacing        m
c = 1.              #Lattice Speed          m/s
Re = 1000.          #Reynold's Number 
#Constant velocity at top of domain      
Vd = nu*Re/(DOMAIN_Y*del_x)

RADIUS = 12         #Radius of the wave     pix
pos_x = 64          #X Coordinate of the sphere
pos_y = DOMAIN_Y//2 #Y Coordinate of the sphere

#Set the inlet and outlet densities
del_rho_ratio = 0.3
rho_avg = 0.5
rho_in = rho_avg * (1 + del_rho_ratio / 2.)
rho_out = rho_avg * (1 - del_rho_ratio / 2.)
#Inverse of these values
invrho_in = 1./rho_in
invrho_out = 1./rho_out

#External forces per unit mass: gravity -9.81m/s
g = np.array([0.0,-9.81])

#The weights and discrete velocities for D2Q9
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
e = np.zeros((Q,D),dtype=int)
e[1,:] = np.array([1,0])    #Right
e[2,:] = np.array([0,1])    #Up
e[3,:] = -e[1,:]            #Left
e[4,:] = -e[2,:]            #Down
e[5,:] = np.array([1,1])    #Up Right
e[6,:] = np.array([-1,1])   #Up Left
e[7,:] = -e[5,:]            #Down Left
e[8,:] = -e[6,:]            #Down Right

#Define collision barriers
bCol = np.full((DOMAIN_X,DOMAIN_Y,2),False)
#bCol[0,:,:] = np.full((DOMAIN_Y,2),True)
#bCol[DOMAIN_X-1,:,:] = bCol[0,:,:]
bCol[:,0,:] = np.full((DOMAIN_X,2),True)
bCol[:,DOMAIN_Y-1,:] = bCol[:,0,:]
bCol[0,:,1] = np.full(DOMAIN_Y,True)
bCol[DOMAIN_X-1,:,1] = np.full(DOMAIN_Y,True)

for i in range(-RADIUS,RADIUS+1):
    for j in range(-RADIUS,RADIUS+1):
        if((i**2 + j**2)<=RADIUS**2):
            bCol[i+pos_x,j+pos_y,0] = True

#Initialize f, feq, and rho
f = np.zeros((DOMAIN_X, DOMAIN_Y, Q))
f[:,:,0] = np.full((DOMAIN_X,DOMAIN_Y),w[0])
f[:,:,1:5] = np.full((DOMAIN_X,DOMAIN_Y,4),w[1])
f[:,:,5:9] = np.full((DOMAIN_X,DOMAIN_Y,4),w[5])
f *= rho_avg
#for i in range(-RADIUS,RADIUS+1):
#    for j in range(-RADIUS,RADIUS+1):
#        if((i**2 + j**2)<=RADIUS*2):
#            f[i+DOMAIN_X//2,j+DOMAIN_Y//2,:] *= 4.

feq = np.zeros((DOMAIN_X, DOMAIN_Y, Q))
rho = np.zeros((DOMAIN_X, DOMAIN_Y))
vel = np.zeros((DOMAIN_X, DOMAIN_Y, 2))
vor = np.zeros((DOMAIN_X, DOMAIN_Y, 2))

#Update Functions
def UpdateDensity(f):
    return np.sum(f,axis=-1)
def UpdateVelocity(f):
    result = c * np.tensordot(f,e,axes=1)
    div = ReverseTranspose(np.tile(rho,(D,1,1)))
    result = np.divide(result,div,np.zeros_like(result),where=div!=0)
    
    return result

#Dimensional Transpose
transpose_key = np.zeros(D+1,dtype=int)
transpose_key[0] = D
transpose_key[1:D+1] = np.arange(D)
transpose_key_rev = np.zeros(D+1,dtype=int)
transpose_key_rev[0:D] = np.arange(1,D+1)
transpose_key_rev[D] = 0
def Transpose(m):
    return np.transpose(m,transpose_key)
def ReverseTranspose(m):
    return np.transpose(m,transpose_key_rev)

#Define the color map data for writing frames to files
col_map = cm.get_cmap('jet')
vel_norm = colors.Normalize(vmin=0.0,vmax=0.25)
vor_norm = colors.Normalize(vmin=-0.1,vmax=0.1)
#Create a unique directory for storing the frames
stamp_time = time.localtime()
timestamp = time.strftime('%m-%d-%Y_%H%M',stamp_time)
directory = "./LBMD2Q9_{}/".format(timestamp)
os.makedirs(directory,exist_ok=True)

for t in range(0,timesteps):
    f_next = np.zeros((DOMAIN_X,DOMAIN_Y,Q))
    for i in range(0, DOMAIN_X):
        for j in range(0, DOMAIN_Y):
            #Since we're using on-grid bounce-back, the colliding points should
            #be ignored
            if(bCol[i,j,0]==False):
                #Create an array of offset data labeled with its direction
                offset = e[:,:] + np.array([i,j])
                offset = np.concatenate((offset,np.array([np.arange(0,Q)]).T),
                    axis=1)

                #Create a mask to prevent accessing invalid data
                mask = offset[:,0:D] >= np.array([0,0])
                mask = mask & (offset[:,0:D] < np.array([DOMAIN_X,DOMAIN_Y]))
                mask = mask[:,0] & mask[:,1]
                offset = offset[mask,:]
                
                #Update mask to determine collision directions
                mask = mask[mask] & (bCol[offset[:,0].T,offset[:,1].T,0]==False)

                #Stream to non-colliding neighbors
                osStream = offset[mask==True,:]
                f_next[osStream[:,0],osStream[:,1],osStream[:,2]] += \
                    f[i,j,osStream[:,2]]
                
                if(bCol[i,j,1]==False):
                    #Bounce-back the colliding velocities on-grid
                    bbRemap = np.array([0,3,4,1,2,7,8,5,6])
                    osCollide = offset[mask == False,:]
                    f_next[i,j,bbRemap[osCollide[:,2]]] += \
                        f[i,j,osCollide[:,2]]

    #Define the Zou-He velocity conditions for the top
    #rho[0:DOMAIN_X,DOMAIN_Y-2] = f_next[0:DOMAIN_X,DOMAIN_Y-2,0] + \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,1] + f_next[0:DOMAIN_X,DOMAIN_Y-2,3] + \
    #    2. * (f_next[0:DOMAIN_X,DOMAIN_Y-2,2] + f_next[0:DOMAIN_X,DOMAIN_Y-2,5] + \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,6])
    #f_next[0:DOMAIN_X,DOMAIN_Y-2,4] = f_next[0:DOMAIN_X,DOMAIN_Y-2,2]
    #f_next[0:DOMAIN_X,DOMAIN_Y-2,7] = (rho[0:DOMAIN_X,DOMAIN_Y-2] * (1. - Vd) - \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,0]) * 0.5 - f_next[0:DOMAIN_X,DOMAIN_Y-2,2] - \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,3] - f_next[0:DOMAIN_X,DOMAIN_Y-2,6]
    #f_next[0:DOMAIN_X,DOMAIN_Y-2,8] = (rho[0:DOMAIN_X,DOMAIN_Y-2] * (1. + Vd) - \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,0]) * 0.5 - f_next[0:DOMAIN_X,DOMAIN_Y-2,1] - \
    #    f_next[0:DOMAIN_X,DOMAIN_Y-2,2] - f_next[0:DOMAIN_X,DOMAIN_Y-2,5]
    
    #Set periodic bounds on the left and right
    #Inlet Wall
    u_x = 1. - invrho_in * (np.sum(f_next[0,2:DOMAIN_Y-2,[0,2,4]],axis=0) + \
        2.*np.sum(f_next[0,2:DOMAIN_Y-2,[3,6,7]],axis=0))
    f_next[0,2:DOMAIN_Y-2,1] = f_next[0,2:DOMAIN_Y-2,3]+(2./3.)*rho_in*u_x
    pre_fact = rho_in * u_x * (1./6.)
    f_next[0,2:DOMAIN_Y-2,5] = pre_fact + 0.5 * \
        (-f_next[0,2:DOMAIN_Y-2,2] + f_next[0,2:DOMAIN_Y-2,4]) + \
        f_next[0,2:DOMAIN_Y-2,7]
    f_next[0,2:DOMAIN_Y-2,8] = pre_fact + 0.5 * \
        (f_next[0,2:DOMAIN_Y-2,2] - f_next[0,2:DOMAIN_Y-2,4] + \
        f_next[0,2:DOMAIN_Y-2,6])
    #Inlet Bottom Corner
    f_next[0,1,[1,2,5]] = f_next[0,1,[3,4,7]]
    f_next[0,1,6] = (-0.5 * f_next[0,1,0]) - \
        np.sum(f_next[0,1,[3,4,7]],axis=0)
    f_next[0,1,6] += 0.5 * rho_in
    f_next[0,1,8] = f_next[0,1,6]
    #Inlet Top Corner
    f_next[0,DOMAIN_Y-2,[1,4,8]] = f_next[0,DOMAIN_Y-2,[3,2,6]]
    f_next[0,DOMAIN_Y-2,5] = (-0.5 * f_next[0,DOMAIN_Y-2,0]) - \
        np.sum(f_next[0,DOMAIN_Y-2,[2,3,6]],axis=0)
    f_next[0,DOMAIN_Y-2,5] += 0.5 * rho_in
    f_next[0,DOMAIN_Y-2,7] = f_next[0,DOMAIN_Y-2,5]
    #Outlet Wall
    u_x = invrho_out * (np.sum(f[DOMAIN_X-1,2:DOMAIN_Y-2,[0,2,4]],axis=0) + \
        2. * np.sum(f_next[DOMAIN_X-1,2:DOMAIN_Y-2,[1,5,8]],axis=0)) - 1.
    f_next[DOMAIN_X-1,2:DOMAIN_Y-2,3] = f_next[DOMAIN_X-1,2:DOMAIN_Y-2,1] - \
        (2./3.) * rho_out * u_x
    pre_fact = rho_out * u_x * (-1./6.)
    f_next[DOMAIN_X-1,2:DOMAIN_Y-2,6] = pre_fact + 0.5 * \
        (-f_next[DOMAIN_X-1,2:DOMAIN_Y-2,2] + f_next[DOMAIN_X-1,2:DOMAIN_Y-2,4]) + \
        f_next[DOMAIN_X-1,2:DOMAIN_Y-2,8]
    f_next[DOMAIN_X-1,2:DOMAIN_Y-2,7] = pre_fact + 0.5 * \
        (f_next[DOMAIN_X-1,2:DOMAIN_Y-2,2] - f_next[DOMAIN_X-1,2:DOMAIN_Y-2,4]) + \
        f_next[DOMAIN_X-1,2:DOMAIN_Y-2,5]
    #Outlet Bottom Corner
    f_next[DOMAIN_X-1,1,[2,3,6]] = f_next[DOMAIN_X-1,1,[4,1,8]]
    f_next[DOMAIN_X-1,1,5] = (-0.5 * f_next[DOMAIN_X-1,1,0]) - \
        np.sum(f_next[DOMAIN_X-1,1,[1,4,8]],axis=0)
    f_next[DOMAIN_X-1,1,5] += 0.5 * rho_out
    f_next[DOMAIN_X-1,1,7] = f_next[DOMAIN_X-1,1,5]
    #Outlet Top Corner
    f_next[DOMAIN_X-1,DOMAIN_Y-2,[3,4,7]] = f_next[DOMAIN_X-1,DOMAIN_Y-2,[1,2,5]]
    f_next[DOMAIN_X-1,DOMAIN_Y-2,6] = (-0.5 * f_next[DOMAIN_X-1,DOMAIN_Y-2,0]) - \
        np.sum(f_next[DOMAIN_X-1,DOMAIN_Y-2,[1,2,5]],axis=0)
    f_next[DOMAIN_X-1,DOMAIN_Y-2,6] += 0.5 * rho_out
    f_next[DOMAIN_X-1,DOMAIN_Y-2,8] = f_next[DOMAIN_X-1,DOMAIN_Y-2,6]

    #Update the densities according to the streaming step
    rho = UpdateDensity(f_next)

    #Update velocities according to the new distribution
    vel = UpdateVelocity(f_next)

    #Calculate the equilibrium distribution
    s = np.tensordot(e,Transpose(vel),axes=1)
    s = s  + (3. * s**2 - np.sum(vel**2,axis=-1)) / (2. * c)
    s = 3. * w * ReverseTranspose(s) / c
    f_eq = (w + s) * ReverseTranspose(np.tile(rho,(Q,1,1)))

    #Calculate the forces
    #F_i = ReverseTranspose(np.tile(rho.T,(2,1,1))) * g
    #F_i = w * np.tensordot(F_i,e.T,axes=1) / c
    
    f[:,:,:] = f_next[:,:,:] - \
         ((f_next[:,:,:] - f_eq[:,:,:]) / tau)
         # + (F_i * del_t)
    #f[1:DOMAIN_X-1,DOMAIN_Y-1,:] = f_next[1:DOMAIN_X-1,DOMAIN_Y-1,:]

    vel = UpdateVelocity(f)
    img = plt.imshow(np.sqrt(np.sum(vel**2,axis=-1)).T,col_map,
       norm=vel_norm,origin='lower')
    img.write_png('{}/LBMD2Q9_Tunnel_Velocity_{:05d}.png'.format(directory,t))

    #Calculate the vorticity using central difference derivatives
    vor[1:DOMAIN_X-1,:,0] = 0.5*(vel[2:DOMAIN_X,:,1]-vel[0:DOMAIN_X-2,:,1])
    vor[0,:,0] = vel[1,:,1] - vel[0,:,1]
    vor[DOMAIN_X-1,:,0] = vel[DOMAIN_X-1,:,1] - vel[DOMAIN_X-2,:,1]
    vor[:,1:DOMAIN_Y-1,1] = 0.5*(-vel[:,2:DOMAIN_Y,0]+vel[:,0:DOMAIN_Y-2,0])
    vor[:,0,1] = -vel[:,1,0] + vel[:,0,0]
    vor[:,DOMAIN_Y-1,1] = -vel[:,DOMAIN_Y-1,0] + vel[:,DOMAIN_Y-2,0]
    img = plt.imshow(np.sum(vor,axis=-1).T,col_map,norm=vor_norm,origin='lower')
    img.write_png('{}/LBMD2Q9_Tunnel_Vorticity_{:05d}.png'.format(directory,t))

end_time = time.perf_counter()
time_total = end_time-start_time
sim_time = "{}:{:02d}:{:09.6f}".format(int(time_total//3600),
    int((time_total//60)%60),time_total%60.)
print("\nSimulation Time for ",DOMAIN_X,"X",DOMAIN_Y," Grid",
    " with ",timesteps," steps: ",sim_time)
tpf = (end_time-start_time)/timesteps
print("\nAvg time per frame: ",tpf,"s\n")

plt.show()

img = plt.imshow(np.sqrt(np.sum(vel**2,axis=-1)).T,col_map,norm=vel_norm,origin='lower')
plt.show()