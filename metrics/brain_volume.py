import numpy as np

def det_dim_3(a):
        size = a.shape[0]
        b = np.zeros(size, dtype=np.float64)
        for i in range(size):
            b[i] = (
            a[i, 0, 0] * a[i, 1, 1] * a[i, 2, 2]
            - a[i, 0, 0] * a[i, 1, 2] * a[i, 2, 1]
            - a[i, 0, 1] * a[i, 1, 0] * a[i, 2, 2]
            + a[i, 0, 1] * a[i, 1, 2] * a[i, 2, 0]
            + a[i, 0, 2] * a[i, 1, 0] * a[i, 2, 1]
            - a[i, 0, 2] * a[i, 1, 1] * a[i, 2, 0]
        )
        return b

def transpose_dim_3(a):
    # Purely equal to np.transpose (a, (0, 2, 1))
    size = a.shape[0]
    b = np.zeros((size, 3, 3), dtype=np.float64)
    for i in range(size):
        b[i, 0, 0] = a[i, 0, 0]
        b[i, 1, 0] = a[i, 0, 1]
        b[i, 2, 0] = a[i, 0, 2]
        b[i, 0, 1] = a[i, 1, 0]
        b[i, 1, 1] = a[i, 1, 1]
        b[i, 2, 1] = a[i, 1, 2]
        b[i, 0, 2] = a[i, 2, 0]
        b[i, 1, 2] = a[i, 2, 1]
        b[i, 2, 2] = a[i, 2, 2]

    return b

def compute_mesh_volume(mesh, characteristics):
    # mesh volume
    # Code source: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py#L165
    print("Computing the folded mesh volume...")

    tets = mesh.cells()
    coordinates = mesh.coordinates()

    mesh_volume = 0.0
    for tet in tets:

        tmp0 = coordinates[tet[1]] - coordinates[tet[0]]
        tmp1 = coordinates[tet[2]] - coordinates[tet[0]]
        tmp2 = coordinates[tet[3]] - coordinates[tet[0]]
        
        tetrahedron_matrix = np.array([ tmp0, tmp1, tmp2 ])
        #tetrahedron_matrix.shape
        det = np.linalg.det( np.transpose(tetrahedron_matrix) )

        mesh_volume += 1/6 * abs(det)

    return mesh_volume