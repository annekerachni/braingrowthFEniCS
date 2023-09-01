"""Define braingrowth kinematics: growth induced elastic deformations --> F = Fe.Fg"""

import fenics


class Kinematics:

    def __init__(self, u, Fg): 
        """u: unknown function standing for displacement vector"""
        self.u = u
        self.dim = self.u.geometric_dimension() # making Id3 dependent from 'u', <Id3, v> becomes a bilinear form. No error.
        self.Id = fenics.Identity(self.dim)

        # F: deformation gradient
        self.F = self.Id + fenics.grad(self.u) # F = I₃ + ∇u / [F]: 3x3 matrix

        # Fe: elastic part of the deformation gradient
        self.Fg = Fg
        Fg_inv = fenics.inv(self.Fg)
        self.Fe = self.F * Fg_inv # F_t * (F_g)⁻¹

        # Cauchy-Green tensors (elastic part of the deformation only)
        self.Ce = self.Fe.T * self.Fe # right Cauchy-Green tensor 
        self.Be = self.Fe * self.Fe.T # left Cauchy-Green tensor

        # Invariants 
        self.Je = fenics.det(self.Fe) 
        self.Tre = fenics.tr(self.Be)

