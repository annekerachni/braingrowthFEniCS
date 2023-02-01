import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert Mesh to XML Fenics')
  parser.add_argument('-i', '--input', help='Input mesh (vtk, gmsh, mesh) ', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output XML mesh', type=str, required=True)

  args = parser.parse_args()
