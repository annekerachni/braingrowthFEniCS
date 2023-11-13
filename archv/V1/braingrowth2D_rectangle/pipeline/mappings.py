import fenics
from scipy.spatial import cKDTree
from numba import jit, prange
import numpy as np
import time 


class Mapping:


    def __init__(self, mesh, VectorSpace_CG1_mesh, VectorSpace_CG1_bmesh):
        self.mesh = mesh
        self.gdim = self.mesh.geometry().dim()

        self.VectorSpace_CG1_mesh = VectorSpace_CG1_mesh
        # Compute vertex to dof map for V Lagrangian Function Space 
        self.vertex2dofs_V = fenics.vertex_to_dof_map(self.VectorSpace_CG1_mesh)
        self.vertex2dofs_V = self.vertex2dofs_V.reshape((-1, self.gdim)) # to associate one over the 2D 'N' vertices to a 2D dof array (e.g. vertex 0 -> dof [880, 881])


        self.VectorSpace_CG1_bmesh = VectorSpace_CG1_bmesh
        # Compute vertex to dof map for B Lagrangian Function Space
        self.vertex2dofs_B = fenics.vertex_to_dof_map(self.VectorSpace_CG1_bmesh)
        self.vertex2dofs_B = self.vertex2dofs_B.reshape((-1, self.gdim))

        print("creating mapping from the brainsurface boundary function space to the whole mesh function space...")
        self.boundaryFunctionSpace_2_meshFunctionSpace_dofmap()
        self.boundaryVertex_2_meshDOF_map()


    """ 
    def boundaryFunctionSpace_2_meshFunctionSpace_dofmap_V1(self):
        # code source: https://fenicsproject.org/qa/8522/mapping-degrees-of-freedom-between-related-meshes/

        # make a map of dofs from the boundarymesh to the original mesh
        bmsh_dof_coordinates = self.VectorSpace_CG1_bmesh.tabulate_dof_coordinates().reshape(-1, self.gdim) #L.dofmap().tabulate_all_coordinates(left_mesh).reshape(-1, gdim)
        mesh_dof_coordinates = self.VectorSpace_CG1_mesh.tabulate_dof_coordinates().reshape(-1, self.gdim) #V.dofmap().tabulate_all_coordinates(mesh).reshape(-1, gdim)
        
        self.B_2_V_dofmap = {}
        One_ofPointVDOFs_already_checked = {}

        print("creating mapping from the brainsurface boundary function space to the whole mesh function space...")
        for Bdof_nr, Bdof_coords in enumerate(bmsh_dof_coordinates):
            corresponding_Vdof = [i for i, Vdof_coords in enumerate(mesh_dof_coordinates) if np.array_equal(Vdof_coords, Bdof_coords)]
            if len(corresponding_Vdof) == 2: # 2 dofs associated to one Point since we use VectorFunctionSpaces V and TB
                point = tuple(Bdof_coords)
                if point not in One_ofPointVDOFs_already_checked: # if key "point" does not exist yet
                    self.B_2_V_dofmap[Bdof_nr] = corresponding_Vdof[0] # allocate first Vdof of the Point
                    One_ofPointVDOFs_already_checked[point] = 1
                else: # One DOF index was already associated to the Point
                    self.B_2_V_dofmap[Bdof_nr] = corresponding_Vdof[1] # allocate seonnd Vdof  of the Point
            else:
                raise NameError("Degrees of freedom not matching.") 
    """


    @jit(parallel=True, forceobj=True)
    def boundaryFunctionSpace_2_meshFunctionSpace_dofmap(self):
        """Build the map of DOFs from the boundarymesh to the original mesh"""
        # source: https://fenicsproject.discourse.group/t/how-to-map-dofs-of-vector-functions-between-meshes-boundarymeshes-submeshes/45
        
        DOF_coords_V = self.VectorSpace_CG1_mesh.tabulate_dof_coordinates()
        DOF_coords_B = self.VectorSpace_CG1_bmesh.tabulate_dof_coordinates()

        # Find coupled-DOFsV corresponding to each single DOFB
        # ----------------------------------------------------
        Vtree = cKDTree(DOF_coords_V)
        _, self.singleDOFB_2_coupledDOFsV_dofmap = Vtree.query(DOF_coords_B, k=self.gdim) # e.g. np.array([[8809, 8808], [8809, 8808], [8811, 8810], [8811, 8810], [8806, 8807], [8806, 8807], ...]) corresponding to following single-DOFB:  np.array([0, 1, 2, 3, 4, 5,  ...])
        # self.B_2_V_dofmap should be np.array([[8808, 8809], [8810, 8811], [8806, 8807], ...]) corresponding coupled-DOFsV to following coupled-DOFB:  np.array([[0, 1], [2, 3], [4, 5],  ...])
        
        # supress coupled-DOFsV redundancies
        self.coupledDOFsV_dofmap = []
        for dofB in prange(len(self.singleDOFB_2_coupledDOFsV_dofmap)):
            if dofB % self.gdim == 0:
                self.coupledDOFsV_dofmap.append(self.singleDOFB_2_coupledDOFsV_dofmap[dofB])
        self.coupledDOFsV_dofmap = np.array(self.coupledDOFsV_dofmap)

        # sort each coupled-DOFsV with first DOF the smallest value
        for DOFscouple_idx in prange(len(self.coupledDOFsV_dofmap)):
            self.coupledDOFsV_dofmap[DOFscouple_idx].sort()

        # coupled-DOFB
        # ------------
        DOFsB = self.vertex2dofs_B # e.g. np.array([[418, 419], [414, 415], [416, 417], [410, 411],  ...]) corresponding to 210 Points /420 coupled-DOFs in brainsurface boundary
        DOFsB[DOFsB[:, 0].argsort()] # e.g. np.array([[0, 1], [2, 3], [4, 5], [6, 7],  ...])

        # Build dofmap
        # ------------
        self.coupledDOFsV_dofmap = self.coupledDOFsV_dofmap.flatten() # e.g. np.array([8809, 8808, 8811, 8810, 8806, 8807, ...]) 
        DOFsB = DOFsB.flatten() # e.g. np.array([0, 1, 2, 3, 4, 5, 6, 7,  ...])
        self.B_2_V_dofmap = []
        for dofB in prange(len(DOFsB)):
            self.B_2_V_dofmap.append(self.coupledDOFsV_dofmap[dofB])
        self.B_2_V_dofmap = np.array(self.B_2_V_dofmap)


    def boundaryVertex_2_meshDOF_map(self):
        """to be used in 'compute_mesh_projected_normals()"""

        print("creating mapping from the Points indices at brainsurface boundary mesh to Points DOFs at whole mesh...")

        # vertex to DOF_Bref --> vertex to DOF_Vref
        self.vertexB_2_dofinVref_mapping = []
        for vertex, dof_inBref in enumerate(self.vertex2dofs_B):
            dof_B_1 = dof_inBref[0]
            dof_B_2 = dof_inBref[1]
            #dof_B_3 = dof_inBref[2]

            dof_V_1 = self.B_2_V_dofmap[dof_B_1]
            dof_V_2 = self.B_2_V_dofmap[dof_B_2]

            self.vertexB_2_dofinVref_mapping.append([dof_V_1, dof_V_2])

        self.vertexB_2_dofinVref_mapping = np.asarray(self.vertexB_2_dofinVref_mapping)
        