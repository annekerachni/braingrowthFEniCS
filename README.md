# braingrowthFEniCS
 
## Presentation
**braingrowthFEniCS** is a biomechanical framework for 3D human brain folding simulation. It includes a computational model of the brain growth dynamics, implemented in Python with the FEniCS library (version 2019.1.0). 

![pipeline](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/b3fea58f-2579-4bf0-99ee-1fdcecfd5d1d)

- The biomechanical model `FEM_biomechanical_model` (resp. `FEM_biomechanical_model_quasistatic`) is developed as a modular framework with adjustable components (2D or 3D geometries; phenomenological, biophysical or partially MRI-based features; growth tensor; constitutive model of the material; variational formulation of the non-linear mechanical problem; contact mechanics). The hypothesis used in the provided version are presented below.
  
- The whole simulation pipeline relies on open source tools such as 3D Slicer, Meshlab, Netgen, FEniCS and slam (https://github.com/gauzias/slam) and uses `niftitomesh`, `metrics`, `utils`, `MRI_driven_parameters`.

## Biomechanical model
#### Brain growth mechanics:
- growth induced deformations:
  - only Cortex grows
  - growth tensor $F^g$: tangential and adaptative
  - tangential growth ratio *alphaTAN*: homogeneous; MRI-based 
  - growth kinetics: linear

- growth kinematics: $F = F^e.F^g$ (multiplicative decomposition of the deformation into elastic and growth deformations)

- constitutive model: Neo-Hookean
  
- problem:
  - balance of the linear momentum &#8594; damped elastodynamics PDE or quasistatic ODE
  - Lagrangian; total Piola-Kirchhoff I stress
  - boundary conditions: traction-free (Neumann); interhemisphere fixed (Dirichlet); contact (for phenomenological dynamic case)

![input_parameters](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/a78adb94-2124-4d9e-999b-ab49c2702268)

#### Dicretization of the problem + FEniCS solver parameters:
- Discretization of the spatial domain with Finite Element Method (Lagrange, degree 1)
- Solver 
  - linearization method: Newton-Raphson
  - lienar solver + preconditioner : ‘gmres’ + ‘sor’ (for 3D)
- Temporal integration method : generalized-alpha method (dynamic case)

## Simulation 
#### Input parameters:
- Brain geometry:
  - *H0*: cortical thickness at t0 [m]
 
- Brain material:
  - *K*: bulk modulus of the brain material [Pa]
  - *muCortex*, *muCore*: shear modulus of Cortex, resp. inner layers of the brain [Pa]
 
- Growth:
  - *alphaTAN*: tangential growth coefficient [s⁻¹]
  - *alphaRAD*: radial growth coefficient [s⁻¹]
  - *grTAN*: weigth for tangential growth [-]
  - *grRAD*: weigth for radial growth [-]

- Simulation:
  - *T0*: initial numerical time, when mesh is smooth [s]
  - *Tmax*: final numerical time [s]
  - *Nsteps*: Number of steps
 
#### Additional dynamic parameters:
- Motion:
  - *rho*: mass density of the brain material (supposed constant) [kg/m⁻³]
  - *damping_coef*:  factor for elastic wave damping, standing for energy dissipation [kg.m⁻³.s⁻¹] 

 - time integration scheme for dynamical PDE: 
  - *alphaM*, *alphaF*: parameters that determine the type of scheme (generalized-α method: $\alpha_{M}=0.2$, $\alpha_{F}=0.4$; Newmark-β method: $\alpha_{M}=0.$, $\alpha_{F}=0.$) 

#### Launch brain growth simulation:
- Create a virtual env with anaconda to use FEniCS: 
  - `conda create -n fenicsvenv`
  - `conda install -c conda-forge fenics` (fenics==2019.1.0)
- Then install dedicated libraries: `pip install -r requirements.txt`
- Simulations can be launched within each simulation case in both `braingrowth_2D` or `braingrowth_3D` folders. e.g. `main_sphere_growth_Fgt.py`; `main_brain_growth_Fgt.py`; `main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand.py`; `main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA_CortexDelineation.py`

## References
- T. Tallinen et al., On the growth and form of cortical convolutions. Nature Physics, 12(6):588–593, 2016 
- X. Wang et al. "The influence of biophysical parameters in a biomechanical model of cortical folding patterns." Scientific Reports 11.1:7686, 2021 + GitHub: https://github.com/rousseau/BrainGrowth 

- M.S. Alnaes et al., The FEniCS project version 1.5. Archive of Numerical Software, 3, 2015
- FEniCS Tutorial "Time-integration of elastodynamics equation" (https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html)
- M. Alenyà et al.  Computational pipeline for the generation and validation of patient-specific mechanical models of brain development. Brain Mutliphysics, 3:100045, 2022
- C.A. Lugo et al. "Morphoelastic modelling of pattern development in the petal epidermal cell cuticle." Journal of the Royal Society Interface 20.204:20230001, 2023. 

- S. Wang et al. Numerical investigation of biomechanically coupled growth in cortical folding. Biomechanics and Modeling in Mechanobiology, 20:555-567, 2021
- S. Budday and P. Steinmann, On the influence of inhomogeneous stiffness and growth on mechanical instabilities in the developing brain. 2018
- M.A. Holland et al, "Emerging brain morphologies from axonal elongation." Annals of biomedical engineering 43:1640-1653, 2015
- P.V. Bayly et al., A cortical folding model incorporating stress-dependent growth explains gyral wavelengths and stress patterns in the developing brain. Physics Biology, 2013

- fetal MRI atlas from the dHCP project: https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas

