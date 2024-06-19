import fenics
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Plot functions
################
def plot_vertex_onto_mesh3D(mesh, vertex_to_highlight_coords, title):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # all mesh nodes
    coords = mesh.coordinates()
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='b', s=0.02)

    #Localize specific vertex
    x, y, z = vertex_to_highlight_coords
    ax.scatter(x, y, z, color='r', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()

    return


def plot_vertices_onto_mesh3D(mesh, dofs_coords, title): # color='r'; 'g'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # all mesh nodes
    coords = mesh.coordinates()
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='b', s=0.02)

    #Localize the selected COGs of the colliding paired-tets
    for coords in dofs_coords:
        x, y, z = coords
        ax.scatter(x, y, z, color='r', s=5) 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()

    return

def plot_colliding_nodes_onto_mesh3D(mesh, nodes_coords_1, color1, nodes_coords_2, color2, title):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # all mesh nodes
    coords = mesh.coordinates()
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='grey', s=0.02)

    #Localize all colliding nodes 
    for i, (coords_1, coords_2) in enumerate(zip(nodes_coords_1, nodes_coords_2)):
        x1, y1, z1 = coords_1 
        ax.scatter(x1, y1, z1, color=color1, s=5) 

        #Localize all paired-colliding nodes 
        x2, y2, z2 = coords_2
        ax.scatter(x2, y2, z2, color=color2, s=5) 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()

    return


def plot_couple_of_nodes_onto_mesh3D(mesh, node1_coords, color1, node2_coords, color2, title):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # all mesh nodes
    coords = mesh.coordinates()
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='grey', s=0.02)

    # Localize node1 
    x1, y1, z1 = node1_coords 
    ax.scatter(x1, y1, z1, color=color1, s=5) 

    # Localize node2 
    x2, y2, z2 = node2_coords
    ax.scatter(x2, y2, z2, color=color2, s=5) 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()

    return
