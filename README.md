# braingrowthFEniCS
 
## Presentation
**braingrowthFEniCS** is a biomechanical framework for 3D human brain folding simulation. It includes a computational model of the brain growth dynamics, implemented in Python with the FEniCS library (version 2019.1.0). 

![pipeline](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/b3fea58f-2579-4bf0-99ee-1fdcecfd5d1d)

- The biomechanical model `FEM_biomechanical_model` is developed as a modular framework with adjustable components (input mesh and biophysical features; growth tensor; constitutive model of the material; variational formulation of the non-linear mechanical problem; contact mechanics). The hypothesis used in the provided version are presented below.
  
- The whole simulation pipeline relies on open source tools such as 3D Slicer, Meshlab, Netgen and FEniCS and uses `niftitomesh`, `metrics`, `utils`

## Biomechanical model
#### Brain growth mechanics:
- problem:
  - balance of the linear momentum &#8594; damped elastodynamics PDE
  - kinematics: $F = F^e.F^g$ (multiplicative decomposition of the deformation into elastic and growth deformations), with $F^g$ tangential and differential.
- constitutive model: Neo-Hookean
- boundary conditions: traction-free (Neumann) + contact
- growth kinetics: linear and homogeneous
- tangential growth tensor

![input_parameters](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/a78adb94-2124-4d9e-999b-ab49c2702268)

#### Dicretization of the problem + FEniCS solver parameters:
- Discretization of the spatial domain with Finite Element Method (Lagrange, degree 1)
- Solver 
  - linearization method: Newton-Raphson
  - lienar solver + preconditioner : ‘gmres’ + ‘sor’ 
- Temporal integration method : generalized-alpha method

## Simulation 
#### Input parameters:
- Brain geometry:
  - *H0*: cortical thickness at t0 [mm]
 
- Brain material:
  - *K*: bulk modulus of the brain material [kPa]
  - *muCortex*, *muCore*: shear modulus of Cortex, resp. inner layers of the brain [kPa]

- Motion:
  - *rho*: mass density of the brain material (supposed constant) [mm/kg⁻¹]
  - *damping_coef*:  ratio for elastic wave damping, standing for energy dissipation [-] 
 
- Growth:
  - *alphaTAN*: tangential growth coefficient [-]
  - *alphaRAD*: radial growth coefficient [-]
  - *grTAN*: weigth for tangential growth [-]
  - *grRAD*: weigth for radial growth [-]

- time integration scheme for dynamical PDE: 
  - *alphaM*, *alphaF*: parameters that determine the type of scheme (generalized-α method: $\alpha_{M}=0.2$, $\alpha_{F}=0.4$; Newmark-β method: $\alpha_{M}=0.$, $\alpha_{F}=0.$) 

- Simulation:
  - *T0*: initial numerical time, when mesh is smooth.
  - *Tmax*: final numerical time 
  - *Nsteps*: Number of steps 
 

#### Launch brain growth simulation:
- Create a virtual env with anaconda to use FEniCS: 
  - `conda create -n fenicsvenv`
  - `conda install -c conda-forge fenics` (fenics==2019.1.0)
- Then install dedicated libraries: `pip install -r requirements.txt`
- Simulations can be launched on sphere geometries via `main_sphere_growth.py` or on brain ones via `main_brain_growth.py`
- e.g. launch command: `python main_sphere_growth.py -i sphere.xdmf  -p ‘{"H0": 0.03, "K": 100.0, "muCortex": 20.0, "muCore": 1.0, "rho": 0.01, "damping_coef": 0.5, "alphaTAN": 3.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "alphaM": 0.2, "alphaF": 0.4, "T0": 0.0, "Tmax": 1.0, "Nsteps": 100, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}’ -o results`

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

