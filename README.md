# braingrowthFEniCS
 2D/3D FEniCS-based model to investigate brain growth-induced folding process
 
## Presentation
- brain growth mechanics model developped with FEniCS library
- Non linear elastodynamics (dynamics: consider high wave frequency model) 
- Discretization of spatial domain (FEM method) and temporal domain (generalized-alpha method)
- Non linear solver: Newton
- 2D and 3D separated codes to improve performance (avoid "if dimension == n" loops). But the architecture and parameters types are the same.

## Usage limits
- bilayer model
- the non-linear residual form is minimized using the Newton-Raphson iterations method on linearized problem => condition for the approximation to be correct: displacement 'du' must be small (and also 'dt')
- traction considered as nul (but needs to be clarified)
- code not parallelized

## Still to be done
- optimize solver parameters
- fine tune parameters
- collisions management
- add 3D bock geometry
- add separatedvisualization package
- add metrics package including slam/Spangy
- "Errors" management
- gradient of layers / growth tensor
- simulation time / real time

## References
- [1] - T.Tallinen, F. Rousseau, J.Lefèvre, X.Wang et al. https://github.com/rousseau/BrainGrowth
- [2] - https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html
- [1] - Carlos Lugo, https://github.com/calugo/wrinkles 
- [2] - Martin Genet, https://gitlab.inria.fr/mgenet/dolfin_mech/-/tree/master/dolfin_mech 
- [3] - Miguel A. Rodriguez and Christoph M. Augustin, https://github.com/ElsevierSoftwareX/SOFTX_2018_73
- [4] - S.Wang, K.Garikipati et al., https://github.com/mechanoChem/mechanoChemML/tree/master/mechanoChemML/workflows 

