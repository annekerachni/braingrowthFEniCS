# braingrowthFEniCS
 
## Presentation
**braingrowthFEniCS** is a biomechanical framework for 3D human brain folding simulation. It includes a solid mechanics computational model of the non-linear brain growth phenomenon (quasistatic), implemented in Python with the FEniCS library (version 2019.1.0). 

![simulation_framework](https://github.com/user-attachments/assets/2f3908ad-ff39-4348-ab21-e74448f4edbe)

- The biomechanical model `FEM_biomechanical_model` (resp. `FEM_biomechanical_model_2D`) is developed as a modular framework with adjustable components (2D or 3D geometries; biophysical or partially MRI-based features; growth tensor; constitutive model of the material; variational formulation of the problem; contact mechanics). The hypothesis used in the provided version are presented below.
  
- The whole simulation pipeline relies on open source tools such as Gmsh, 3D Slicer, Meshlab, Netgen, FEniCS and slam (https://github.com/gauzias/slam) and uses `niftitomesh`, `metrics`, `utils`, `MRI_driven_parameters`.

## Biomechanical model
#### Brain growth mechanics:
- growth induced deformations:
  - only Cortex grows
  - growth tensor $F^g$: tangential and adaptative
  - tangential growth ratio *alphaTAN*: homogeneous; MRI-based
  - growth kinetics: linear
  - N.B. the model includes a potential radial growth in the brain sub-layers, so it can be easily added modifying the radial growth ratio *alphaRAD*

- growth kinematics: $F = F^e.F^g$ (multiplicative decomposition of the deformation into elastic and growth deformations)

- material law: Neo-Hookean
  
- discretized problem:
  - FEM residual formulation:
    - conservation law: balance of the linear momentum (quasistatic ODE)
    - penalization of the self-collisions between the two hemispheres for brain simulations (contact mechanics)
    - Lagrangian; total Piola-Kirchhoff I stress
      
  - boundary conditions:
    - Neumann: traction-free
    - Dirichlet: brain regions defined as "no-growth zones" are fixed (option 1) / mostly in the case of the ellipsoid, inner ellipsoid is emptied in the mesh to define Ventricular Zone that are fixed (option 2) as in [M.J. Razavi et al. 2015]

#### FEniCS solver parameters:
  - linearization method: Newton-Raphson
  - lienar solver: ‘mumps’

## Simulation 
#### Biophysical parameters:
- Brain geometry:
  - *H0*: cortical thickness at t0 [m]
 
- Brain material:
  - *K*: bulk modulus of the brain material [Pa]
  - *muCortex*, *muCore*: shear modulus of Cortex, resp. inner layers of the brain [Pa]
 
- Growth:
  - *alphaTAN*: tangential growth coefficient [(m).s⁻¹]
  - *alphaRAD*: radial growth coefficient [(m).s⁻¹]
  - *grTAN*: weigth for tangential growth [-]
  - *grRAD*: weigth for radial growth [-]
 
- Contact:
  - *epsilon*: penalty coefficient

- Simulation:
  - *T0*: initial numerical time, when mesh is smooth [gestational weeks (GW)]
  - *Tmax*: final numerical time [GW]
  - *dt*: timestep [s]

#### MRI-informed parameters
![input_parameters](https://github.com/annekerachni/braingrowthFEniCS/assets/89976599/a78adb94-2124-4d9e-999b-ab49c2702268)

#### Launch brain growth simulation:
- Create a virtual env with anaconda to use FEniCS: 
  - `conda create -n fenicsvenv`
  - `conda install -c conda-forge fenics` (fenics==2019.1.0)
- Then install dedicated libraries: `pip install -r requirements.txt`
- Simulations can be launched within each simulation case in both `BrainGrowth2D`, `BrainGrowth3D` or `BrainGrowth3D_MRI` folders. e.g. `main_brain_growth.py`; `main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA_CortexDelineation.py`

## References
- T. Tallinen et al., On the growth and form of cortical convolutions. Nature Physics, 12(6):588–593, 2016 
- X. Wang et al. "The influence of biophysical parameters in a biomechanical model of cortical folding patterns." Scientific Reports 11.1:7686, 2021 + GitHub: https://github.com/rousseau/BrainGrowth
- M.J. Razavi et al. Role of mechanical factors in cortical folding development. Physical Review E, 2015, vol. 92, no 3, p. 032701.

- M.S. Alnaes et al., The FEniCS project version 1.5. Archive of Numerical Software, 3, 2015
- FEniCS Tutorial "Time-integration of elastodynamics equation" (https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html)
- M. Alenyà et al.  Computational pipeline for the generation and validation of patient-specific mechanical models of brain development. Brain Mutliphysics, 3:100045, 2022
- C.A. Lugo et al. "Morphoelastic modelling of pattern development in the petal epidermal cell cuticle." Journal of the Royal Society Interface 20.204:20230001, 2023. 

- S. Wang et al. Numerical investigation of biomechanically coupled growth in cortical folding. Biomechanics and Modeling in Mechanobiology, 20:555-567, 2021
- S. Budday and P. Steinmann, On the influence of inhomogeneous stiffness and growth on mechanical instabilities in the developing brain. 2018
- M.A. Holland et al, "Emerging brain morphologies from axonal elongation." Annals of biomedical engineering 43:1640-1653, 2015
- P.V. Bayly et al., A cortical folding model incorporating stress-dependent growth explains gyral wavelengths and stress patterns in the developing brain. Physics Biology, 2013

- fetal MRI atlas from the dHCP project: https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas

