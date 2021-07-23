""" 
Oct-Tree - 26 January 2021
========================

This is a proof-of-concept for testing a quad-tree algorithm to be used for
O(N log N) particle updates for SPH in 3D or 2D.

Dependencies
------------
 - SciPy libraries : for Python numerical methods and data visualization
 - ffmpeg : for compiling frames into videos for final viewing

References
----------
.. [1] Pfalzner, S., & Gibbon, P. (2005). Basic Principles of the 
       Hierarchical Tree Method. In Many-body tree methods in physics 
       (pp. 9-18). Cambridge: Cambridge University Press.
"""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os
import time

# Global control functions ====================================================

WRITE_TO_FILES = True
CURRENT_FRAME = 0

# Domain Parameters ===========================================================

DOMAIN_X = 4.
DOMAIN_Y = 4.
DOMAIN_Z = 4.
NDIM = 3                    # <-- Switch between 2- and 3-D
FIXED_ROOT_SIZE = True      # Determines if the root should match the domain

# Simulation Parameters =======================================================

PARTICLE_COUNT = 64
PARTICLE_MASS = 0.1

# Initialize the particle positions to be evenly spread rotationally
part_pos = np.zeros((PARTICLE_COUNT, NDIM))
randTheta = 2. * np.pi * np.random.rand(PARTICLE_COUNT)
randPhi = np.full(PARTICLE_COUNT, 0.)
if(NDIM == 3) : randPhi = np.pi * (np.random.rand(PARTICLE_COUNT) - 0.5)
part_pos[:,0] = np.cos(randTheta) * np.cos(randPhi)
part_pos[:,1] = np.sin(randTheta) * np.cos(randPhi)
if(NDIM == 3) : part_pos[:,2] = np.sin(randPhi)
part_pos *= (0.4 * DOMAIN_X) * np.random.rand(PARTICLE_COUNT, 1)

# Functions and Operation =====================================================

# Create the fig and axes for rendering the points and tree
fig = plt.figure()
if(NDIM == 3) : ax = fig.add_subplot(projection="3d")
else : ax = fig.add_subplot()

# Set the axes properties
ax.set_xlim(-DOMAIN_X / 2., DOMAIN_X / 2.)
ax.set_ylim(-DOMAIN_Y / 2., DOMAIN_Y / 2.)
if(NDIM == 3) : ax.set_zlim(-DOMAIN_Z / 2., DOMAIN_Z / 2.)
else : ax.set_aspect('equal')

# Render the scatter plot of points on the top layer of the figure
if(NDIM == 3) : 
    scatter = ax.scatter3D(part_pos[:,0], part_pos[:,1], part_pos[:,2], s=2)
else : scatter = ax.scatter(part_pos[:,0], part_pos[:,1], s=2)

# Create a unique directory for storing the frames
stamp_time = time.localtime()
timestamp = time.strftime('%m-%d-%Y_%H%M',stamp_time)
if(NDIM == 3):
    directory = "./OctTree_{}/".format(timestamp)
else:
    directory = "./QuadTree_{}/".format(timestamp)
if(WRITE_TO_FILES) : os.makedirs(directory,exist_ok=True)
if(NDIM == 3):
    prefix = "{}OctTree_".format(directory)
else:
    prefix = "{}QuadTree_".format(directory)

# Save the first frame to file
if(WRITE_TO_FILES) : 
    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
    CURRENT_FRAME += 1

class LLNode:
    """
    (L)inked (L)ist Node - contains the data and the "pointer" to the next
    element of the list

    Attributes
    ----------
    data 
        The data of arbitrary type stored in this node
    next : LLNode
        Points to the next element of the array

    """
    def __init__(self, data):
        self.next = None
        self.data = data

class Stack:
    """
    A basic stack (FIFO) data structure implemented using a linked list with 
    the top entry as the head

    Attributes
    ----------
    head : LLNode
        The first entry of the stack and the start of the internal linked list
    """
    def __init__(self):
        self.head = None

    def insert(self, data):
        """ Inserts the data to the top of the stack stored in a new node """
        if(self.head == None):
            self.head = LLNode(data)
        else:
            newNode = LLNode(data)
            newNode.next = self.head
            self.head = newNode

    def is_empty(self):
        return (self.head == None)

    def peek(self):
        """ Return the top entry of the stack """
        return self.head.data

    def pop(self):
        """ Pop the top entry off the stack and return it """
        if(self.head != None):
            node = self.head
            self.head = node.next
            return node
        else: return None

class QuadNode:
    """
    The basic unit that makes up the quadtree

    Attributes
    ----------
    x_min : float
    x_max : float
    y_min : float
    y_max : float

    cell_0 : QuadNode
    cell_1 : QuadNode
    cell_2 : QuadNode
    cell_3 : QuadNode

    level : int
    """
    def __init__(self, x_min, y_min, x_max, y_max):
        self.data = None

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.cell_0 = None
        self.cell_1 = None
        self.cell_2 = None
        self.cell_3 = None

        self.level = 0

    def is_point_inside(self, x, y):
        if(x < self.x_min): return False
        if(x >= self.x_max): return False
        if(y < self.y_min): return False
        if(y >= self.y_max): return False

    def split(self):
        x_mid = ((self.x_max - self.x_min) / 2.) + self.x_min
        y_mid = ((self.y_max - self.y_min) / 2.) + self.y_min
        self.cell_0 = QuadNode(self.x_min, self.y_min, x_mid, y_mid)
        self.cell_1 = QuadNode(x_mid, self.y_min, self.x_max, y_mid)
        self.cell_2 = QuadNode(x_mid, y_mid, self.x_max, self.y_max)
        self.cell_3 = QuadNode(self.x_min, y_mid, x_mid, self.y_max)

    def render(self, draw_recursive=True):
        """ 
        Add the square boundary of this node to the plot 
        
        Arguments
        ---------
        draw_recursive : boolean
            Indicates whether to draw the children too
        """
        rect = plt.Rectangle((self.x_min, self.y_min), 
            self.x_max - self.x_min, self.y_max - self.y_min, 
            ec = 'grey', fill = False)
        plt.gca().add_patch(rect)

        # If indicated not to, leave the function before rendering the children
        if not draw_recursive: return

        # Recursively call the daughter nodes so they are also displayed
        if self.cell_0 != None: self.cell_0.render(True)
        if self.cell_1 != None: self.cell_1.render(True)
        if self.cell_2 != None: self.cell_2.render(True)
        if self.cell_3 != None: self.cell_3.render(True)

class QuadTree:
    """
    The top-level structure responsible for generating and storing the entire
    quad-tree

    Attributes
    ----------
    root : QuadNode
    """
    def __init__(self, points):
        # Set the root node large enough to encapsulate all points
        x_min = -DOMAIN_X / 2.
        x_max = DOMAIN_X / 2.
        y_min = -DOMAIN_Y / 2.
        y_max = DOMAIN_Y / 2.

        numpoints = np.shape(points)[0]
        if not FIXED_ROOT_SIZE:
            # Expand the root to fit all points even outside the domain 
            for pointID in range(1, numpoints):
                if points[pointID][0] < x_min: x_min = points[pointID][0]
                elif points[pointID][0] > x_max: x_max = points[pointID][0]
                if points[pointID][1] < y_min: y_min = points[pointID][1]
                elif points[pointID][1] > y_max: y_max = points[pointID][1]

        # Create the root node
        self.root = QuadNode(x_min, y_min, x_max, y_max)

        # Create the stack and add the root node to be processed as a tuple
        #   with the indices of the points inside it (all; 0 to numpoints)
        stack = Stack()
        stack.insert((self.root, np.arange(0, numpoints)))

        # Keep processing/subdividing the nodes until the stack is empty
        while not stack.is_empty():
            nodeData = stack.pop().data            
            self.__subdivide__(nodeData[0], points, nodeData[1], stack)

    def render(self): self.root.render()

    def __subdivide__(self, node, points, indices, subdivide_stack):
        """
        Subdivide the given node into daughter nodes based on the universal set and
        the indices

        Arguments
        ---------
        node : QuadNode
            The node to be subdivided
        points : array_like
            The universal set of all nodes in the system
        indices : array_like
            The indices of the points inside the given node
        subdivide_stack : Stack
            The stack that stores all nodes to be processed (subdivided) further

        Notes
        -----
        The subdivide stack should have nodes with a tuple of the type 
        (QuadNode, array_like) that stores the node that needs to be processed and
        the array of indices of the points inside that node
        """
        # Render this cell to frame
        global CURRENT_FRAME
        if(WRITE_TO_FILES):
            node.render(False)
            plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
            CURRENT_FRAME += 1

        # Iterate the level since it's being broken down to the next stage
        level = node.level + 1

        # Initialize a set of indices for each array
        inds_0 = np.array([], dtype=np.dtype(int))
        inds_1 = np.array([], dtype=np.dtype(int))
        inds_2 = np.array([], dtype=np.dtype(int))
        inds_3 = np.array([], dtype=np.dtype(int))

        # Get the mid-points for partitioning the particles
        x_mid = ((node.x_max - node.x_min) / 2.) + node.x_min
        y_mid = ((node.y_max - node.y_min) / 2.) + node.y_min

        # Iterate over the indices in the given node
        for i in np.nditer(indices):
            # Partition the points by the x-axis then the y-axis to determine
            #   which node they're in
            if(points[int(i),0] <= x_mid):
                if(points[int(i),1] <= y_mid): inds_0 = np.append(inds_0, int(i))
                else: inds_3 = np.append(inds_3, int(i))
            else:
                if(points[int(i),1] <= y_mid): inds_1 = np.append(inds_1, int(i))
                else: inds_2 = np.append(inds_2, int(i))

        # Only process the nodes if they are non-empty
        if inds_0.size != 0: 
            # Create a new node
            node.cell_0 = QuadNode(node.x_min, node.y_min, x_mid, y_mid)
            node.cell_0.parent = node
            node.cell_0.level = level

            # If there are multiple points, this node must be subdivided
            if inds_0.size == 1:
                node.cell_0.plabel = inds_0[0]
                node.cell_0.mass = PARTICLE_MASS
                node.cell_0.center_of_gravity = points[inds_0[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_0.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_0, inds_0))

        if inds_1.size != 0: 
            node.cell_1 = QuadNode(x_mid, node.y_min, node.x_max, y_mid)
            node.cell_1.parent = node
            node.cell_1.level = level
            
            # If there are multiple points, this node must be subdivided
            if inds_1.size == 1:
                node.cell_1.plabel = inds_1[0]
                node.cell_1.mass = PARTICLE_MASS
                node.cell_1.center_of_gravity = points[inds_1[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_1.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_1, inds_1))

        if inds_2.size != 0: 
            node.cell_2 = QuadNode(x_mid, y_mid, node.x_max, node.y_max)
            node.cell_2.parent = node
            node.cell_2.level = level

            # If there are multiple points, this node must be subdivided
            if inds_2.size == 1:
                node.cell_2.plabel = inds_2[0]
                node.cell_2.mass = PARTICLE_MASS
                node.cell_2.center_of_gravity = points[inds_2[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_2.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_2, inds_2))

        if inds_3.size != 0: 
            node.cell_3 = QuadNode(node.x_min, y_mid, x_mid, node.y_max)
            node.cell_3.parent = node
            node.cell_3.level = level

            # If there are multiple points, this node must be subdivided
            if inds_3.size == 1:
                node.cell_3.plabel = inds_3[0]
                node.cell_3.mass = PARTICLE_MASS
                node.cell_3.center_of_gravity = points[inds_3[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_3.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_3, inds_3))

class OctNode:
    """
    The basic unit that makes up the oct-tree

    Attributes
    ----------
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    z_min : float
    z_max : float

    cell_0 : OctNode
    cell_1 : OctNode
    cell_2 : OctNode
    cell_3 : OctNode
    cell_4 : OctNode
    cell_5 : OctNode
    cell_6 : OctNode
    cell_7 : OctNode

    level : int
    """
    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max):
        self.data = None

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.cell_0 = None
        self.cell_1 = None
        self.cell_2 = None
        self.cell_3 = None
        self.cell_4 = None
        self.cell_5 = None
        self.cell_6 = None
        self.cell_7 = None

        self.level = 0

    def is_point_inside(self, x, y, z):
        if(x <= self.x_min): return False
        if(x > self.x_max): return False
        if(y <= self.y_min): return False
        if(y > self.y_max): return False
        if(z <= self.z_min): return False
        if(z > self.z_max): return False

    def split(self):
        x_mid = ((self.x_max - self.x_min) / 2.) + self.x_min
        y_mid = ((self.y_max - self.y_min) / 2.) + self.y_min
        z_mid = ((self.z_max - self.z_min) / 2.) + self.z_min
        self.cell_0 = OctNode(self.x_min, self.y_min, self.z_min,
            x_mid, y_mid, z_mid)
        self.cell_1 = OctNode(x_mid, self.y_min, self.z_min, 
            self.x_max, y_mid, z_mid)
        self.cell_2 = OctNode(x_mid, y_mid, self.z_min, 
            self.x_max, self.y_max, z_mid)
        self.cell_3 = OctNode(self.x_min, y_mid, self.z_min,
            x_mid, self.y_max, z_mid)
        self.cell_4 = OctNode(self.x_min, self.y_min, z_mid,
            x_mid, y_mid, self.z_max)
        self.cell_5 = OctNode(x_mid, self.y_min, z_mid, 
            self.x_max, y_mid, self.z_max)
        self.cell_6 = OctNode(x_mid, y_mid, z_mid, 
            self.x_max, self.y_max, z_mid)
        self.cell_7 = OctNode(self.x_min, y_mid, z_mid,
            x_mid, self.y_max, self.z_max)

    def render(self, draw_recursive=True):
        """ 
        Add the square boundary of this node to the plot 
        
        Arguments
        ---------
        draw_recursive : boolean
            Indicates whether to draw the children too
        """
        ax.plot(np.array([self.x_min, self.x_max]), 
                np.array([self.y_min, self.y_min]),
                np.array([self.z_min, self.z_min]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_max]), 
                np.array([self.y_min, self.y_min]),
                np.array([self.z_max, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_max]), 
                np.array([self.y_max, self.y_max]),
                np.array([self.z_max, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_max]), 
                np.array([self.y_max, self.y_max]),
                np.array([self.z_min, self.z_min]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_min]), 
                np.array([self.y_min, self.y_max]),
                np.array([self.z_min, self.z_min]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_min]), 
                np.array([self.y_min, self.y_max]),
                np.array([self.z_max, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_max, self.x_max]), 
                np.array([self.y_min, self.y_max]),
                np.array([self.z_max, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_max, self.x_max]), 
                np.array([self.y_min, self.y_max]),
                np.array([self.z_min, self.z_min]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_min]), 
                np.array([self.y_min, self.y_min]),
                np.array([self.z_min, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_min, self.x_min]), 
                np.array([self.y_max, self.y_max]),
                np.array([self.z_min, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_max, self.x_max]), 
                np.array([self.y_max, self.y_max]),
                np.array([self.z_min, self.z_max]), c='grey', lw=0.5)
        ax.plot(np.array([self.x_max, self.x_max]), 
                np.array([self.y_min, self.y_min]),
                np.array([self.z_min, self.z_max]), c='grey', lw=0.5)      

        # If indicated not to, leave the function before rendering the children
        if not draw_recursive: return

        # Recursively call the daughter nodes so they are also displayed
        if self.cell_0 != None: self.cell_0.render(True)
        if self.cell_1 != None: self.cell_1.render(True)
        if self.cell_2 != None: self.cell_2.render(True)
        if self.cell_3 != None: self.cell_3.render(True)
        if self.cell_4 != None: self.cell_4.render(True)
        if self.cell_5 != None: self.cell_5.render(True)
        if self.cell_6 != None: self.cell_6.render(True)
        if self.cell_7 != None: self.cell_7.render(True)

class OctTree:
    """
    The top-level structure responsible for generating and storing the entire
    oct-tree

    Attributes
    ----------
    root : OctNode
    """
    def __init__(self, points):
        # Set the root node large enough to encapsulate all points
        x_min = -DOMAIN_X / 2.
        x_max = DOMAIN_X / 2.
        y_min = -DOMAIN_Y / 2.
        y_max = DOMAIN_Y / 2.
        z_min = -DOMAIN_Z / 2.
        z_max = DOMAIN_Z / 2.

        numpoints = np.shape(points)[0]
        if not FIXED_ROOT_SIZE:
            # Expand the root to fit all points even outside the domain 
            for pointID in range(1, numpoints):
                if points[pointID][0] < x_min: x_min = points[pointID][0]
                elif points[pointID][0] > x_max: x_max = points[pointID][0]
                if points[pointID][1] < y_min: y_min = points[pointID][1]
                elif points[pointID][1] > y_max: y_max = points[pointID][1]
                if points[pointID][2] < z_min: z_min = points[pointID][2]
                elif points[pointID][2] > z_max: z_max = points[pointID][2]

        # Create the root node
        self.root = OctNode(x_min, y_min, z_min, x_max, y_max, z_max)

        # Create the stack and add the root node to be processed as a tuple
        #   with the indices of the points inside it (all; 0 to numpoints)
        stack = Stack()
        stack.insert((self.root, np.arange(0, numpoints)))

        # Keep processing/subdividing the nodes until the stack is empty
        while not stack.is_empty():
            nodeData = stack.pop().data            
            self.__subdivide__(nodeData[0], points, nodeData[1], stack)

    def render(self): self.root.render()

    def __subdivide__(self, node, points, indices, subdivide_stack):
        """
        Subdivide the given node into daughter nodes based on the universal set 
        and the indices

        Arguments
        ---------
        node : OctNode
            The node to be subdivided
        points : array_like
            The universal set of all nodes in the system
        indices : array_like
            The indices of the points inside the given node
        subdivide_stack : Stack
            The stack that stores all nodes to be processed (subdivided)
            further

        Notes
        -----
        The subdivide stack should have nodes with a tuple of the type 
        (OctNode, array_like) that stores the node that needs to be processed 
        and the array of indices of the points inside that node
        """
        # Render this cell to frame
        global CURRENT_FRAME
        if(WRITE_TO_FILES):
            node.render(False)
            plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
            CURRENT_FRAME += 1

        # Iterate the level since it's being broken down to the next stage
        level = node.level + 1

        # Initialize a set of indices for each array
        inds_0 = np.array([], dtype=np.dtype(int))
        inds_1 = np.array([], dtype=np.dtype(int))
        inds_2 = np.array([], dtype=np.dtype(int))
        inds_3 = np.array([], dtype=np.dtype(int))
        inds_4 = np.array([], dtype=np.dtype(int))
        inds_5 = np.array([], dtype=np.dtype(int))
        inds_6 = np.array([], dtype=np.dtype(int))
        inds_7 = np.array([], dtype=np.dtype(int))

        # Get the mid-points for partitioning the particles
        x_mid = ((node.x_max - node.x_min) / 2.) + node.x_min
        y_mid = ((node.y_max - node.y_min) / 2.) + node.y_min
        z_mid = ((node.z_max - node.z_min) / 2.) + node.z_min

        # Iterate over the indices in the given node
        for i in np.nditer(indices):
            # Partition the points by the x-axis then the y-axis to determine
            #   which node they're in
            if(points[int(i),0] <= x_mid):
                if(points[int(i),1] <= y_mid): 
                    if(points[int(i),2] <= z_mid):
                        inds_0 = np.append(inds_0, int(i))
                    else: inds_4 = np.append(inds_4, int(i))
                else: 
                    if(points[int(i),2] <= z_mid):
                        inds_3 = np.append(inds_3, int(i))
                    else: inds_7 = np.append(inds_7, int(i))
            else:
                if(points[int(i),1] <= y_mid): 
                    if(points[int(i),2] <= z_mid):
                        inds_1 = np.append(inds_1, int(i))
                    else: inds_5 = np.append(inds_5, int(i))
                else: 
                    if(points[int(i),2] <= z_mid):
                        inds_2 = np.append(inds_2, int(i))
                    else: inds_6 = np.append(inds_6, int(i))

        # Only process the nodes if they are non-empty
        if inds_0.size != 0: 
            # Create a new node
            node.cell_0 = OctNode(node.x_min, node.y_min, node.z_min,
                x_mid, y_mid, z_mid)
            node.cell_0.parent = node
            node.cell_0.level = level

            # If there are multiple points, this node must be subdivided
            if inds_0.size == 1:
                node.cell_0.plabel = inds_0[0]
                node.cell_0.mass = PARTICLE_MASS
                node.cell_0.center_of_gravity = points[inds_0[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_0.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_0, inds_0))

        if inds_1.size != 0: 
            node.cell_1 = OctNode(x_mid, node.y_min, node.z_min,
                node.x_max, y_mid, z_mid)
            node.cell_1.parent = node
            node.cell_1.level = level
            
            # If there are multiple points, this node must be subdivided
            if inds_1.size == 1:
                node.cell_1.plabel = inds_1[0]
                node.cell_1.mass = PARTICLE_MASS
                node.cell_1.center_of_gravity = points[inds_1[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_1.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_1, inds_1))

        if inds_2.size != 0: 
            node.cell_2 = OctNode(x_mid, y_mid, node.z_min,
                node.x_max, node.y_max, z_mid)
            node.cell_2.parent = node
            node.cell_2.level = level

            # If there are multiple points, this node must be subdivided
            if inds_2.size == 1:
                node.cell_2.plabel = inds_2[0]
                node.cell_2.mass = PARTICLE_MASS
                node.cell_2.center_of_gravity = points[inds_2[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_2.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_2, inds_2))

        if inds_3.size != 0: 
            node.cell_3 = OctNode(node.x_min, y_mid, node.z_min,
                x_mid, node.y_max, z_mid)
            node.cell_3.parent = node
            node.cell_3.level = level

            # If there are multiple points, this node must be further subdivided
            if inds_3.size == 1:
                node.cell_3.plabel = inds_3[0]
                node.cell_3.mass = PARTICLE_MASS
                node.cell_3.center_of_gravity = points[inds_3[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_3.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_3, inds_3))

        # Only process the nodes if they are non-empty
        if inds_4.size != 0: 
            # Create a new node
            node.cell_4 = OctNode(node.x_min, node.y_min, z_mid,
                x_mid, y_mid, node.z_max)
            node.cell_4.parent = node
            node.cell_4.level = level

            # If there are multiple points, this node must be subdivided
            if inds_4.size == 1:
                node.cell_4.plabel = inds_4[0]
                node.cell_4.mass = PARTICLE_MASS
                node.cell_4.center_of_gravity = points[inds_4[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_4.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_4, inds_4))

        if inds_5.size != 0: 
            node.cell_5 = OctNode(x_mid, node.y_min, z_mid,
                node.x_max, y_mid, node.z_max)
            node.cell_5.parent = node
            node.cell_5.level = level
            
            # If there are multiple points, this node must be subdivided
            if inds_5.size == 1:
                node.cell_5.plabel = inds_5[0]
                node.cell_5.mass = PARTICLE_MASS
                node.cell_5.center_of_gravity = points[inds_5[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_5.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_5, inds_5))

        if inds_6.size != 0: 
            node.cell_6 = OctNode(x_mid, y_mid, z_mid,
                node.x_max, node.y_max, node.z_max)
            node.cell_6.parent = node
            node.cell_6.level = level

            # If there are multiple points, this node must be subdivided
            if inds_6.size == 1:
                node.cell_6.plabel = inds_6[0]
                node.cell_6.mass = PARTICLE_MASS
                node.cell_6.center_of_gravity = points[inds_6[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_6.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_6, inds_6))

        if inds_7.size != 0: 
            node.cell_7 = OctNode(node.x_min, y_mid, z_mid,
                x_mid, node.y_max, node.z_max)
            node.cell_7.parent = node
            node.cell_7.level = level

            # If there are multiple points, this node must be further subdivided
            if inds_7.size == 1:
                node.cell_7.plabel = inds_7[0]
                node.cell_7.mass = PARTICLE_MASS
                node.cell_7.center_of_gravity = points[inds_7[0],:].copy()

                if(WRITE_TO_FILES):
                    node.cell_7.render(False)
                    plt.savefig('{}{:04d}.png'.format(prefix, CURRENT_FRAME))
                    CURRENT_FRAME += 1
            else:
                # Add this node to the stack of nodes to further subdivide
                subdivide_stack.insert((node.cell_7, inds_7))

# Generate a tree
if(NDIM == 3): tree = OctTree(part_pos)
else: tree = QuadTree(part_pos)

# Compile the frames to mp4 using ffmpeg
if(WRITE_TO_FILES) : 
    os.system("ffmpeg -r 12 -f image2 -i {}%04d.png -vcodec libx264 \
        -crf 25 -pix_fmt yuv420p Tree_{}.mp4".format( \
            prefix, timestamp)) 
else: 
    tree.render()
    plt.show()