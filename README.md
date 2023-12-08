# braingrowthFEniCS
 
## Presentation
"braingrowthFEniCS" is simulation framework for 3D human brain folding simulation. It includes a computational model of the brain growth dynamics, implemented in Python with the FEniCS library (version 2019.1.0). 
- The brain growth model `braingrowth` is developed as a modular framework with adjustable components (input mesh and biophysical features; growth tensor; constitutive model of the material; variational formulation of the non-linear mechanical problem; contact mechanics). The hypothesis used in the provided version are presented below.
- The simulation pipeline and relies on open source tools such as 3D Slicer, Meshlab, Netgen and FEniCS and uses `nifti2mesh`, `metrics`, `utils`.

## The brain growth model
#### Brain growth mechanics:
- problem:
  - balance of the linear momentum &#8594; damped elastodynamics PDE
  - kinematics: F = Fe.Fg (multiplicative decomposition of the deformation into elastic and growth deformations), with Fg tangential and differential.
- constitutive model: Neo-Hookean
- boundary conditions: traction-free (Neumann)
- growth kinetics: linear and homogeneous
- tangential growth tensor
  ![dgTAN](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/35f7b097-d48e-47f8-b2f8-2b4fb122e099)

#### Dicretization of the problem + FEniCS solver parameters:
- Discretization of the spatial domain with Finite Element Method (Lagrange, degree 1)
- Solver 
  - linearization method: Newton-Raphson
  - lienar solver + preconditioner : ‘gmres’ + ‘sor’ 
- Temporal integration method : generalized-alpha method

## Simulation 
#### Framework
![simulation_framework](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/abd59aaf-22aa-4c5f-a8dd-89c3fc85addc)

#### Launch brain growth simulation:
- Create a virtual env with anaconda to use FEniCS: `conda create -n fenicsvenv -c conda-forge fenics`
- Launch the command:
`python3 -i ./braingrowth/main_solverFg0.py -i './data/dhcp21GW_17K_refined10.xml' -n True -p '{"H0": 0.04, "K": 100.0, "muCortex": 20.0, "muCore": 1.0, "rho": 0.01, "damping_coef": 10.0, "alphaTAN": 4.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "alphaM": 0.2, "alphaF": 0.4, "T0": 0.0, "Tmax": 0.5, "Nsteps": 100, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}' -o './simulations/dhcp/' `

## References
- T. Tallinen et al., On the growth and form of cortical convolutions. Nature Physics, 12(6):588–593, 2016 
- X. Wang et al. "The influence of biophysical parameters in a biomechanical model of cortical folding patterns." Scientific Reports 11.1:7686, 2021 + GitHub: https://github.com/rousseau/BrainGrowth 

- M.S. Alnaes et al., The FEniCS project version 1.5. Archive of Numerical Software, 3, 2015
- FEniCS Tutorial "Time-integration of elastodynamics equation" (https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html)
- C.A. Lugo et al. "Morphoelastic modelling of pattern development in the petal epidermal cell cuticle." Journal of the Royal Society Interface 20.204:20230001, 2023. 
- M.Alenyà et al.  Computational pipeline for the generation and validation of patient-specific mechanical models of brain development. Brain Mutliphysics, 3:100045, 2022

- S.Wang et al. Numerical investigation of biomechanically coupled growth in cortical folding. Biomechanics and Modeling in Mechanobiology, 20:555-567, 2021
- S.Budday, On the influence of inhomogeneous stiffness and growth on mechanical instabilities in the developing brain. 2018
- M.A. Holland et al, "Emerging brain morphologies from axonal elongation." Annals of biomedical engineering 43:1640-1653, 2015
- P.Bayly et al., A cortical folding model incorporating stress-dependent growth explains gyral wavelengths and stress patterns in the developing brain. Physics Biology, 2013

- fetal MRI atlas from the dHCP project: https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas

