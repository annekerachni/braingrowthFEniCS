import fenics
import meshio

inputmesh_path = "./data/MRI_informed_simulations/dHCP_surface_21GW/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf"
# from dHCP surface gifti mesh: "./data/MRI_informed_simulations/dHCP_surface_21GW/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf"--> in millimeters
# from dHCP volume niftis: "./data/MRI_informed_simulations/dHCP_volume_21GW/dhcp_volume_t21_raw_130000faces_480112tets.xdmf"

output_file_xdmf = "./data/MRI_informed_simulations/dHCP_surface_21GW/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume_RAS.xdmf"

# import mesh
inputmesh_format = inputmesh_path.split('.')[-1]

if inputmesh_format == "xml":
    mesh = fenics.Mesh(inputmesh_path)

elif inputmesh_format == "xdmf":
    mesh = fenics.Mesh()
    with fenics.XDMFFile(inputmesh_path) as infile:
        infile.read(mesh)
    
# Revert X<>Y coordinates inversion 
mesh_coordinates = mesh.coordinates() # list of nodes coordinates
X = mesh_coordinates[:,0].copy()
Y = mesh_coordinates[:,1].copy()
mesh_coordinates[:,0] = Y
mesh_coordinates[:,1] = X 

# save reverted mesh file in .xdmf
meshio.write(output_file_xdmf, meshio.Mesh(points=mesh.coordinates(), cells={'tetra': mesh.cells()})) 

