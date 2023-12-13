import matplotlib.pyplot as plt
import numpy as np

from simulation_braingrowth.MRI_driven_parameters.FA import FA_2_growth_laws

# Tangential growth from FA
###########################

def normalize_FA(fa, normalized_fa, vertex2dofs_S, mesh_meshio_with_FA):
    """
    Args:
    fa: fenics.Function(S)
    normalized_fa: fenics.Function(S)
    t_simu: 0.
    mesh_meshio_with_FA : .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    FA = mesh_meshio_with_FA.point_data["FA"] # nodal array 
    max_FA = np.max(FA)
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        fa.vector()[scalarDOF] = FA[vertex]
        normalized_fa.vector()[scalarDOF] = FA[vertex]/max_FA
        
    return fa, normalized_fa # FEniCS scalar functions


def tangential_growth_coef_from_FA(linear_coef, alphaTAN, vertex2dofs_S, normalized_fa):    
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa : fenics.Function(S) 
    """   

    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        alphaTAN.vector()[scalarDOF] = FA_2_growth_laws.FA_to_tangential_growth_law_linearrelationship(linear_coef, normalized_fa.vector()[scalarDOF] ) # FA should be a float between 0. and 1. => normalized_fa = fa/np.max(fa)  + growth is a linear function of FA. (here we test alphaTAN = 5*FA)
        #alphaTAN.vector()[scalarDOF] = FA_2_growth_laws.FA_to_tangential_growth_law_gompertz(linear_coef, normalized_fa.vector()[scalarDOF])
        #alphaTAN.vector()[scalarDOF] = FA_2_growth_laws.FA_to_tangential_growth_law_exponential(linear_coef, normalized_fa.vector()[scalarDOF])
        
    return alphaTAN


# Radial growth (no impact of the FA for the moment)
###############
def radial_growth_coef(): 

    alphaRAD = 0
    """
    grRAD = 1.
    alphaRAD = 0. # --> no radial growth
    dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) ) # = 0.0
    """
    
    return alphaRAD