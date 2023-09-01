import fenics

# Mass form
###########
def m(rho, a, v_test):
    return rho * fenics.dot(a, v_test) * fenics.Measure("dx") 

#

# Damping form
##############
def c(damping_coef, v, v_test):
    return damping_coef * fenics.dot(v, v_test) * fenics.Measure("dx") 

# 

# Stiffness form
################
def k(u, v_test, Fg, mu, K, gdim):

    # kinematics
    # ----------
    Id = fenics.Identity(gdim)

    # F: deformation gradient
    F = fenics.variable( Id + fenics.grad(u) ) # F = I₃ + ∇u / [F]: 3x3 matrix

    # Fe: elastic part of the deformation gradient
    Fg_inv = fenics.variable( fenics.inv(Fg) )
    Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹

    # Cauchy-Green tensors (elastic part of the deformation only)
    Ce = fenics.variable( Fe.T * Fe )
    Be = fenics.variable( Fe * Fe.T )

    # Invariants 
    Je = fenics.variable( fenics.det(Fe) ) 
    Tre = fenics.variable( fenics.tr(Be) )
    
    # Neo-Hookean strain energy density function
    # ------------------------------------------
    We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
    #We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang

    # Cauchy stress (elastic part)
    # -------------
    Te = fenics.variable( mu * (Be - Tre/3 * Id) * pow(Je, -5/3) \
                        + K * (Je - 1) * Id ) # X.Wang (original BrainGrowth simulation code)# Cauchy stress
    
    # 1st Piola-Kirchhoff stress (elastic part)
    # --------------------------
    PK1e = fenics.variable( Je * Te * fenics.inv(Fe.T) )# X.Wang (originel BrainGrowth simulation code)
    #PK1e = fenics.diff(We, Fe)
    
    return fenics.inner(PK1e, fenics.grad(v_test)) * fenics.Measure("dx") 

# 

# Work of external forces
#########################
def Wext(body_forces_V, v_test):
    return fenics.dot(body_forces_V, v_test) * fenics.Measure("dx") 

# 

# Traction
##########
def traction(tract_V, ds, v_test):
    return fenics.dot(tract_V, v_test) * ds

# 

# Contact (Penalty Force form)
##############################
def contact(fcontact_global_V, ds, v_test):
    return fenics.dot(fcontact_global_V, v_test) * ds(101)

