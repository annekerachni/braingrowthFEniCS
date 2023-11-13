# braingrowthFEniCS
SIMULATING THE EMERGENCE OF THE EARLY HUMAN BRAIN CORTICAL FOLDS WITH FENICS (2D/3D model)
 
## Presentation
### Brain growth mechanics model:
- problem:
  - balance of the linear momentum -> damped elastodynamics PDE
  - kinematics: F = Fe.Fg (multiplicative decomposition of the deformation into elastic and growth deformations), with Fg tangential and differential.
  - 
- boundary conditions: traction-free (Neumann)
- growth kinetics: linear and homogeneous 

### Computational model implemented with the FEniCS library:
- Discretization of the spatial domain with Finite Element Method (Lagrange, degree 1)
- Solver 
  - linearization method: Newton-Raphson
  - lienar solver + preconditioner : ‘gmres’ + ‘sor’ 
- Temporal integration method : generalized-alpha method 

## References
- [1] - T.Tallinen, F. Rousseau, J.Lefèvre, X.Wang et al. https://github.com/rousseau/BrainGrowth
- [2] - https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html
- [1] - Carlos Lugo, https://github.com/calugo/wrinkles 
- [2] - Martin Genet, https://gitlab.inria.fr/mgenet/dolfin_mech/-/tree/master/dolfin_mech 
- [3] - Miguel A. Rodriguez and Christoph M. Augustin, https://github.com/ElsevierSoftwareX/SOFTX_2018_73
- [4] - S.Wang, K.Garikipati et al., https://github.com/mechanoChem/mechanoChemML/tree/master/mechanoChemML/workflows 

