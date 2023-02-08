"""Define mechanical properties of the material use to model brain tissue"""

import fenics 


class Material:


    def __init__(self, shared_parameters):
        self.shared_parameters = shared_parameters
        self.allocate_shared_parameters()
    

    def allocate_shared_parameters(self):
        self.rho = self.shared_parameters['rho']
        self.damping = self.shared_parameters['damping']
        self.k = self.shared_parameters['k']


class NeoHookeanElasticMaterial(Material):


    def __init__(self, shared_parameters, kinematics, neohookean_cortex_parameters, neohookean_core_parameters, gm):
        super().__init__(shared_parameters)
        self.kinematics = kinematics
        self.neohookean_cortex_parameters = neohookean_cortex_parameters # mu, K
        self.neohookean_core_parameters = neohookean_core_parameters # mu, K
        self.gm = gm # Scalar Function of the mesh

        self.allocate_shared_parameters()
    
    def compute_gradientbased_stiffnesses(self):
        muCortex = self.neohookean_cortex_parameters['mu']
        muCore = self.neohookean_core_parameters['mu']
        self.mu = muCore * (1.0 - self.gm) + muCortex * self.gm # Global modulus of white matter and gray matter
        #self.mu = fenics.project(mu_expression, ScalarFunctionSapce)

        return self.mu


    # Energy density W
    def compute_strain_energy_density(self):
        """
        # https://fenics-solid-tutorial.readthedocs.io/en/latest/2DNonlinearElasticity/2DNonlinearElasticity.html (stored energy function)
        # https://tel.archives-ouvertes.fr/tel-03624456/document (theoritical mechanics and stresses)
        """
        if (self.kinematics.dim == 3):
            self.We = 0.5 * self.mu * (self.kinematics.Tre * pow(self.kinematics.Je, -1/2) - 3) + 0.5 * self.k * (self.kinematics.Je - 1) * (self.kinematics.Je - 1) # T. Tallinen 

            #self.We =  0.5*self.mu*(self.kinematics.Tre - 3 - 2*fenics.ln(self.kinematics.Je)) + 0.5*fenics.lda*2*fenics.ln(self.kinematics.Je)*fenics.ln(self.kinematics.Je) # S. Wang & M.A. Holland, 2019: "Numerical investigation of biomechanically coupled growth in cortical folding"; Carlos Lugo

        elif (self.kinematics.dim == 2):
            self.We =  0.5*self.mu*(self.kinematics.Tre - 2 - 2*fenics.ln(self.kinematics.Je)) + 0.5*fenics.lda*2*fenics.ln(self.kinematics.Je)*fenics.ln(self.kinematics.Je) # https://gitlab.inria.fr/mgenet/dolfin_mech/-/blob/master/dolfin_mech/Material_Elastic_NeoHookean.py

        return self.We
    

    # Cauchy stress (elastic part of the deformation only)
    def compute_Cauchy_stress(self):
        """
        Te = 1/Je * Pe * Fe.T  # Cauchy-stress tensor # "Gyrification from constrained cortical expansion", T.Tallinen et al., 2014 
        """
        self.Te = self.mu * (self.kinematics.Be - self.kinematics.Tre/3 * self.kinematics.Id) * pow(self.kinematics.Je, -5/3) + self.k * (self.kinematics.Je - 1) * self.kinematics.Id # X.Wang (originel BrainGrowth simulation code)

        """ sigma_2d = 2 * inv(Je) * Fe * diff(W_2d, variable(Ce)) * Fe.T  """ # ---> not working  
        #self.Te = (self.mu*self.kinematics.Be + (fenics.lda * fenics.ln(self.kinematics.Je) - self.mu) * self.kinematics.Id)/self.kinematics.Je # # S. Wang & M.A. Holland, 2019: "Numerical investigation of biomechanically coupled growth in cortical folding"; Carlos Lugo

        return self.Te


    # 1st Piola-Kirchhoff stress (elastic part of the deformation only)
    def compute_PKI_stress(self):
        """
        Se = diff(We, Fe) # 1st Piola-Kirchhoff stress tensor # 1st Piola-Kirchhoff-stress tensor, T.Tallinen et al.
        """

        return 


    # 2nd Piola-Kirchhoff stress (elastic part of the deformation only)
    def compute_PKII_stress(self):
        """
        Pe = inv(Fe) * Pe # 2nd Piola-Kirchhoff stress tensor, T.Tallinen et al.
        """
        self.Pe = self.kinematics.Je * self.compute_Cauchy_stress() * fenics.inv(self.kinematics.Fe.T) # X.Wang (originel BrainGrowth simulation code)

        return self.Pe

