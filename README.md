# braingrowthFEniCS
 
## Presentation
**braingrowthFEniCS** is a biomechanical computational model of cortical folding for simulations on human whole-brain geometry, formulated within a finite element context and implemented in Python using the FEniCS code (version 2019.1.0). [Alnaes et al., 2015]. 

This model is integrated with an open-source simulation framework that facilitates mesh generation and model parameterization directly from MRI data, incorporating cortical surface biometrics used for model calibration. The whole simulation pipeline relies on open source tools such as Gmsh, 3D Slicer, Meshlab, Netgen, FEniCS and slam (https://github.com/gauzias/slam) and uses `niftitomesh`, `metrics`, `utils`, `MRI_driven_parameters`.

A multi-scale approach to cortical folding modeling is also proposed, which use anatomical and diffusion MRI data to inform and refine model parameters.

![braingrowthFEniCS](https://github.com/user-attachments/assets/80db78da-2220-4ec2-9ea5-a94087019ca2)

## An example of cortical folding simulation from real brain data
![Capture d’écran du 2025-07-07 14-56-13](https://github.com/user-attachments/assets/232a664f-8252-4a16-80df-2e37f8e266c6)

![REFblue_vs_dHCPsurfacebeige_1](https://github.com/user-attachments/assets/c658e7d7-8863-4557-8b58-b55d8e0231b0)

## Biomechanical model
#### Brain growth mechanics:
- growth-induced deformations:
  - only Cortex grows
  - growth tensor $F^g$: tangential and adaptative
  - tangential growth ratio *alphaTAN*: homogeneous; MRI-based
  - growth kinetics: linear
  - N.B. The model allows for radial growth in the cortex sub-layers. This can be implemented by adjusting the value of the radial growth ratio, referred to as *alphaRAD*.

- growth kinematics: $F = F^e.F^g$ (multiplicative decomposition of the deformation into elastic and growth deformations)

- material law: Neo-Hookean
  
- discretized problem:
  - FEM residual formulation:
    - conservation law: balance of the linear momentum (quasistatic ODE)
    - penalization of the self-collisions between the two hemispheres for brain simulations (contact mechanics)
    - updated Lagrangian; total Piola-Kirchhoff I stress
      
  - boundary conditions:
    - Neumann: traction-free
    - Dirichlet:
      - Brain regions defined as "no-growth zones" are fully fixed (option 1) [T. Tallinen et al., 2016].
      - Mainly for simplified 3D geometries (sphere, ellipsoid), the interior of the mesh is emptied to define the "Ventricular Zone" boundary that will be fully fixed (option 2) as in [M.J. Razavi et al. 2015]

#### FEniCS solver parameters:
  - linearization method: Newton-Raphson
  - lienar solver: ‘mumps’

## Simulation 
#### Biophysical parameters:
- Brain geometry:
  - The input brain mesh should be in RAS+ orientation and either in millimeters or in meters
  - *H0*: cortical thickness at t0 [m]
 
- Brain material:
  - *nu*: Poisson's ratio of the brain material [-]
  - *muCortex*, *muCore*: shear modulus of Cortex, resp. inner layers of the brain [Pa]
 
- Growth:
  - *alphaTAN*: tangential growth coefficient [s⁻¹]
  - *alphaRAD*: radial growth coefficient [s⁻¹]
  - *grTAN*: weigth for tangential growth [-]
  - *grRAD*: weigth for radial growth [-]
 
- Contact:
  - *epsilon*: penalty coefficient [kg.m⁻².s⁻²]

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
- Purely biomechanical simulations can be launched within each simulation case in both `BrainGrowth2D`, `BrainGrowth3D` folders. e.g. `main_sphere_growth.py`; `main_wholebrain_growth.py`.
- Simulations with MRI-informed parameters can be launched within `BrainGrowth3D_MRI` folder, with `main_wholebrain_growth_MRI_informed.py`.

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

