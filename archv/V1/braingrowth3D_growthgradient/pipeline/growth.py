"""Define braingrowth differential (binary and adaptative here) growth tensor, creating material deformations"""

import fenics
import numpy as np


class GrowthTensor:


    def __init__(self, ScalarSpace_CG1_mesh):

        self.dg_TAN = fenics.Function(ScalarSpace_CG1_mesh, name="dgTAN")
        self.dg_RAD = fenics.Function(ScalarSpace_CG1_mesh, name="dgTAN")

        """ # tangential growth coefficient 
        dgTAN0 = fenics.Constant(0.) # Here: growth variation is between mesh t-dt and mesh t. In original BrainGrowth: the growth variation is between mesh 0 and mesh t
        self.dg_TAN = fenics.Expression('dgTAN', degree=1, dgTAN=dgTAN0) # Here: g_TAN[Point] = 1 + dg_TAN[Point] = = 1 + gr_braingrowthFEniCS*alpha_TAN[Point]*dt. In original BrainGrowth: g_TAN[Point] = 1 + at_TAN[Point] * gm_TAN[Point] = 1 + alpha_TAN[Point]*t * gr_BrainGrowth_TAN[Point]/(1 + exp**(10(Point.y/cortical_thickness - 1))

        # radial growth coefficient
        dgRAD0 = fenics.Constant(0.)
        self.dg_RAD = fenics.Expression('dgRAD', degree=1, dgRAD=dgRAD0) """
    
    def define_adaptative_growth_tensor(self, u, Mesh_Nt): 

        # Define adaptative and tangential growth tensor
        # ----------------------------------------------
        Nt_Nt_proj = fenics.outer(Mesh_Nt, Mesh_Nt)

        d = u.geometric_dimension() # making Id3 dependent from 'u', <Id3, v> becomes a bilinear form. No error.
        Id = fenics.Identity(d) 
        Fg =  (1 + self.dg_TAN) * ( Id - Nt_Nt_proj ) + (1 + self.dg_RAD) * Nt_Nt_proj # # g_TAN[Point] = 1 + dg_TAN[Point] = 1 + gr_TAN[Point] * alpha_TAN * dt

        return Fg


    def compute_topboundary_normals(self, mesh, ds, VectorSpace_CG1, brainsurface_mark): 
        """
        Compute normals at the mesh boundary. Prerequisite for computation of adaptative Mesh_Nt and Growth Tensor.
        Returns: Function of the Vector Space defined on all mesh (FEniCS object). DOF indexation.
        """
        u_trial = fenics.TrialFunction(VectorSpace_CG1)
        v_test = fenics.TestFunction(VectorSpace_CG1)
        Boundary_Nt = fenics.Function(VectorSpace_CG1)
        n = fenics.FacetNormal(mesh)
        a = fenics.inner(u_trial, v_test) * ds(brainsurface_mark) 
        l = fenics.inner(n, v_test) * ds(brainsurface_mark)
        A = fenics.assemble(a, keep_diagonal=True)
        L = fenics.assemble(l)
        A.ident_zeros()
        fenics.solve(A, Boundary_Nt.vector(), L)

        return Boundary_Nt 

    #@jit(parallel=True)
    def compute_mesh_projected_normals(self, mesh_vertex_coords, bmeshB_vertex_coords, vertexB_2_dofinVref_mapping, vertex2dofs_V, Mesh_Nt, Boundary_Nt):
        """
        Project the normal vector at closest boundary Point to each of the mesh Points. Prerequisite for computation of adaptative Growth Tensor.
        Returns: Function of the Vector Space defined on all mesh (FEniCS object). DOF indexation.
        
        Ressources: https://abhigupta.io/2021/01/25/visualize-dofs.html 
        See this in case of approximation with higher order than P1 basis functions: https://fenicsproject.org/pub/tutorial/html/._ftut1019.html
        """

        # Project B normals to all mesh Points/Vertices
        for vertex_index, coords in enumerate(mesh_vertex_coords): #N vertex indices in the whole Mesh ref
            
            # Compute vertex index (int) in B (Function Space) reference (i.e. less indices) of the closest Point at Top Boundary to given Mesh Point ('vertex_index')   
            closestBPointIndex_inBref = np.argmin(np.linalg.norm(bmeshB_vertex_coords - mesh_vertex_coords[vertex_index], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
            closestBPointDOF_inVref = vertexB_2_dofinVref_mapping[closestBPointIndex_inBref]

            # Allocate to given mesh Point ('vertex_index') the normal vector of the closest Point at Top Boundary (=> get projected normal vector for all mesh Points) 
            Mesh_Nt.vector()[vertex2dofs_V[vertex_index]] = Boundary_Nt.vector()[closestBPointDOF_inVref] 
            
        return Mesh_Nt 

    