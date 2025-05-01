#import matplotlib.pyplot as plt
import numpy as np
import os, sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))) 


# Tangential growth from FA
###########################

def normalize_FA(fa, normalized_fa, vertex2dofs_S, mesh_loaded_with_FA):
    """
    Args:
    fa: fenics.Function(S)
    normalized_fa: fenics.Function(S)
    tGW: "21GW"
    t_simu: 0.
    path_inputmeshloaded_with_SegLabels_VTK: .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    #mesh_loaded_with_FA = meshio.read(path_inputmeshloaded_with_FA_VTK)
    FA = mesh_loaded_with_FA.point_data["FA"] # nodal array 
    max_FA = np.max(FA)
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        fa.vector()[scalarDOF] = FA[vertex]
        normalized_fa.vector()[scalarDOF] = FA[vertex]/max_FA
        
    return fa, normalized_fa # FEniCS scalar functions


#####
# FA to alphaTAN laws
#####

def FA_to_alphaTAN_law(alphaTAN,  
                       vertex2dofs_S, 
                       normalized_fa,  
                       law="1over1plusFA_expNeg",
                       **kwargs): 
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa: fenics.Function(S) 
    - law: "linear", "linear_expNeg", "1over1plusFA", "1over1plusFA_expNeg"
    """ 

    if law == "linear":
        for vertex, scalarDOF in enumerate(vertex2dofs_S):
            alphaTAN.vector()[scalarDOF] = normalized_fa.vector()[scalarDOF]

    elif law == "linear_expNeg":
        """
        additional_params = {'t_in_GW': t_in_GW, 'T0_in_GW': T0_in_GW, 'tau':tau}
        FA_to_alphaTAN_law(alphaTAN, vertex2dofs_S, normalized_fa, law="linear_expNeg",**additional_params)
        """
    
        # FERRET & HUMAN
        ################
        # [22] C.D. Kroenke et al. “Regional patterns of cerebral cortical differentiation determined by diffusion tensor MRI”. In: Cerebral Cortex 19.12 (2009), pp. 2916–2929.
        # [26] Kroenke C.D. “Using diffusion anisotropy to study cerebral cortical gray matter development”. In: Journal of Magnetic Resonance 292 (2018), pp. 106–116.
        # --> FA(t) = (FAmax(x, T0) - FAmin(x, 40GW)) * math.exp(-(t_in_GW - T0_in_GW)/tau) + FAmin(x, 40GW))
        # --> See [22] : FAmin(x, 40GW) ~ 3/8* FAmax(x, T0)for ferret

        # HUMAN
        #######
        # TRIVEDI, Richa, GUPTA, Rakesh K., HUSAIN, Nuzhat, et al. Region-specific maturation of cerebral cortex in human fetal brain: diffusion tensor imaging and histology. Neuroradiology, 2009, vol. 51, p. 567-576.
        # --> FA decay in parietal lobe. at 21GW, FA ~ 0.3; at 36GW, FA ~ 0.1 => --> FA(t) = (FAmax(x, 21GW) - FAmin(x, 36GW)) * math.exp(-(t_in_GW - T0_in_GW=21GW)/tau) + FAmin(x, 36GW))
        
        # TODO: Rigorously, should be: (normalized_FA(x) - FA_min(x)) * math.exp(-(t_in_GW - T0_in_GW)/tau) + FA_min(x)

        t_in_GW = kwargs.get('t_in_GW')
        T0_in_GW = kwargs.get('T0_in_GW')
        tau = kwargs.get('tau')
        
        for vertex, scalarDOF in enumerate(vertex2dofs_S):
            alphaTAN.vector()[scalarDOF] = (normalized_fa.vector()[scalarDOF] - 1/3*normalized_fa.vector()[scalarDOF]) * math.exp(-(t_in_GW - T0_in_GW)/tau) + 1/3*normalized_fa.vector()[scalarDOF]

    elif law == "1over1plusFA" :
        for vertex, scalarDOF in enumerate(vertex2dofs_S):
            alphaTAN.vector()[scalarDOF] = (1 / (1 + normalized_fa.vector()[scalarDOF]) )

    elif law == "1over1plusFA_expNeg":
        """
        additional_params = {'t_in_GW': t_in_GW, 'T0_in_GW': T0_in_GW, 'tau':tau}
        FA_to_alphaTAN_law(alphaTAN, vertex2dofs_S, normalized_fa, law="1over1plusFA_expNeg",**additional_params)
        """
    
        # FERRET & HUMAN
        ################
        # [22] C.D. Kroenke et al. “Regional patterns of cerebral cortical differentiation determined by diffusion tensor MRI”. In: Cerebral Cortex 19.12 (2009), pp. 2916–2929.
        # [26] Kroenke C.D. “Using diffusion anisotropy to study cerebral cortical gray matter development”. In: Journal of Magnetic Resonance 292 (2018), pp. 106–116.
        # --> FA(t) = (FAmax(x, T0) - FAmin(x, 40GW)) * math.exp(-(t_in_GW - T0_in_GW)/tau) + FAmin(x, 40GW))
        # --> See [22] : FAmin(x, 40GW) ~ 3/8* FAmax(x, T0)for ferret

        # HUMAN
        #######
        # TRIVEDI, Richa, GUPTA, Rakesh K., HUSAIN, Nuzhat, et al. Region-specific maturation of cerebral cortex in human fetal brain: diffusion tensor imaging and histology. Neuroradiology, 2009, vol. 51, p. 567-576.
        # --> FA decay in parietal lobe. at 21GW, FA ~ 0.3; at 36GW, FA ~ 0.1 => --> FA(t) = (FAmax(x, 21GW) - FAmin(x, 36GW)) * math.exp(-(t_in_GW - T0_in_GW=21GW)/tau) + FAmin(x, 36GW))
        
        # normalized_FA_temporal = (normalized_FA.vector()[scalarDOF] - 1/3*normalized_FA.vector()[scalarDOF]) * math.exp(-(t_in_GW - T0_in_GW)/tau) + 1/3*normalized_FA.vector()[scalarDOF]
        # alphaTAN.vector()[scalarDOF] = 1 / (1 + normalized_FA_temporal) 
        # TODO: Rigorously, should be: 1 / (1 + (normalized_FA(x) - FA_min(x)) * math.exp(-(t_in_GW - T0_in_GW)/tau) + FA_min(x))

        t_in_GW = kwargs.get('t_in_GW')
        T0_in_GW = kwargs.get('T0_in_GW')
        tau = kwargs.get('tau')

        for vertex, scalarDOF in enumerate(vertex2dofs_S):
            normalized_FA_temporal = (normalized_fa.vector()[scalarDOF] - 1/3*normalized_fa.vector()[scalarDOF]) * math.exp(-(t_in_GW - T0_in_GW)/tau) + 1/3*normalized_fa.vector()[scalarDOF]
            alphaTAN.vector()[scalarDOF] = 1 / (1 + normalized_FA_temporal) 
                
    return alphaTAN