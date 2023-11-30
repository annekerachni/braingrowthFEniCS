# braingrowthFEniCS
 
## Presentation
SIMULATING THE EMERGENCE OF THE EARLY HUMAN BRAIN CORTICAL FOLDS WITH FENICS (2D/3D model)

#### Brain growth mechanics model:
- problem:
  - balance of the linear momentum &#8594; damped elastodynamics PDE
  - kinematics: F = Fe.Fg (multiplicative decomposition of the deformation into elastic and growth deformations), with Fg tangential and differential.
- constitutive model: Neo-Hookean
- boundary conditions: traction-free (Neumann)
- growth kinetics: linear and homogeneous 

#### Computational model implemented with the FEniCS library:
- Discretization of the spatial domain with Finite Element Method (Lagrange, degree 1)
- Solver 
  - linearization method: Newton-Raphson
  - lienar solver + preconditioner : ‘gmres’ + ‘sor’ 
- Temporal integration method : generalized-alpha method

## Launch brain growth simulation:
The brain growth mechanics problem is solved with the open-source FEniCS code (version 2019.1.0).
- Create a virtual env with anaconda to use FEniCS: `conda create -n fenicsvenv -c conda-forge fenics`
- Launch the command:
`python3 -i ./braingrowth/main_solverFg0.py -i './data/dhcp21GW_17K_refined10.xml' -n True -p '{"H0": 0.04, "K": 100.0, "muCortex": 20.0, "muCore": 1.0, "rho": 0.01, "damping_coef": 10.0, "alphaTAN": 4.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "alphaM": 0.2, "alphaF": 0.4, "T0": 0.0, "Tmax": 0.5, "Nsteps": 100, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}' -o './simulations/dhcp/' `

## References
- [1]  T.Tallinen, F. Rousseau, J.Lefèvre, X.Wang et al. https://github.com/rousseau/BrainGrowth
- [2] https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html
- [3] Carlos Lugo, https://github.com/calugo/wrinkles 
- [4] Martin Genet, https://gitlab.inria.fr/mgenet/dolfin_mech/-/tree/master/dolfin_mech 
- [5] Miguel A. Rodriguez and Christoph M. Augustin, https://github.com/ElsevierSoftwareX/SOFTX_2018_73
- [6] S.Wang, K.Garikipati et al., https://github.com/mechanoChem/mechanoChemML/tree/master/mechanoChemML/workflows
- [7] M.Alenyà et al.  Computational pipeline for the generation and validation of patient-specific mechanical models of brain development. Brain Mutliphysics, Volume 3, 2022, 100045.
- [8] S.Budday, On the influence of inhomogeneous stiffness and growth on mechanical instabilities in the developing brain. 2018
- [9] P.Bayly et al., A cortical folding model incorporating stress-dependent growth explains gyral wavelengths and stress patterns in the developing brain. Physics Biology, 2013.
- [10] S.Wang et al. Numerical investigation of biomechanically coupled growth in cortical folding. Biomechanics and Modeling in Mechanobiology, 20:555-567, 2021.

