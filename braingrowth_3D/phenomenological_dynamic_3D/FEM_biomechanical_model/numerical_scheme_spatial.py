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
    #We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
    We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang https://github.com/rousseau/BrainGrowth/

    # Cauchy stress (elastic part)
    # -------------
    Te = fenics.variable( mu * (Be - Tre/3 * Id) * pow(Je, -5/3) \
                        + K * (Je - 1) * Id ) # X.Wang https://github.com/rousseau/BrainGrowth/
    
    # 1st Piola-Kirchhoff stress (elastic part)
    # --------------------------
    """ PK1e = fenics.variable( Je * Te * fenics.inv(Fe.T) ) """# X.Wang https://github.com/rousseau/BrainGrowth/
    #PK1e = fenics.diff(We, Fe)

    # 1st Piola-Kirchhoff stress (total)
    # --------------------------
    # Need to translate Pelastic into Ptotal to apply the equilibrium balance momentum law.
    PK1tot = fenics.diff(We, F) 
    
    """
    # 2nd Piola-Kirchhoff stress (elastic part)
    # --------------------------
    PK2e = 2 * fenics.diff(We, Tre) * Id + Je * Je * fenics.diff(We, Je) * fenics.inv(Ce)

    # 2nd Piola-Kirchhoff stress (total)
    # --------------------------
    PK2tot = fenics.variable( fenics.det(Fg) ) * Fg_inv * PK2e * Fg_inv

    # Total first Piola-Kirchhoff stress tensor
    PK1tot = F * PK2tot
    """
    
    #return fenics.inner(PK1e, fenics.grad(v_test)) * fenics.Measure("dx") 
    return fenics.inner(PK1tot, fenics.grad(v_test)) * fenics.Measure("dx") 

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

# Contact 
#########
def contact(fcontact_global_V, ds, v_test):
    return fenics.dot(fcontact_global_V, v_test) * ds(101)

"""
def contact_penalty_method_residual_term(ForceS_penalty_integrand, ds):
    return  ForceS_penalty_integrand * ds(101) # return additional penalization residual term
"""

def contact_penalty_method_residual_PENALTY_SelfContact(ForceV_penalty_integrand, ds, v_test):
    """residual form for penalty algorithm considering self-contact"""
    return  fenics.dot(ForceV_penalty_integrand, v_test) * ds(101) # return additional penalization residual term

def contact_residual_term_2Hemispheres_PENALTY_2Hemispheres(fcontact_Hemisphere1_V, fcontact_Hemisphere2_V, ds, v_test):
    """residual form for penalty algorithm considering 2 distinguised brain hemispheres "master" (102) and "slave" (103)"""
    return fenics.dot(fcontact_Hemisphere1_V, v_test) * ds(102) + \
           fenics.dot(fcontact_Hemisphere2_V, v_test) * ds(103)

"""
def contact_residual_term_2Hemispheres_PENALTY_theory(fcontact_Hemisphere1_V, fcontact_Hemisphere2_V, ds, v_test):
    return ( fenics.dot(fcontact_Hemisphere1_V, v_test) + fenics.dot(fcontact_Hemisphere2_V, v_test) ) * ds(101) # cf. integration on "master" contact boundary only (cf penalty method equations for FEM residual form)
"""

