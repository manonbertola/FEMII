import numpy as np


def getArc(elements='quad4'):

    file_name = 'arc_struct_{}.msh'.format(elements)

    with open(file_name) as file:

        lines = file.readlines()

        nodes_start = lines.index('$Nodes\n')
        nodes_end = lines.index('$EndNodes\n')

        connectivity_start = lines.index('$Elements\n')
        connectivity_end = lines.index('$EndElements\n')


    mesh = {}

    start, end = nodes_start+2, nodes_end-nodes_start-2
    coordinates = np.loadtxt(file_name, skiprows=start, max_rows=end)
    mesh['coordinates'] = coordinates[:, 1:-1]

    start, end = connectivity_start+2, connectivity_end-connectivity_start-2
    connectivity = np.loadtxt(file_name, skiprows=start, max_rows=end)
    mesh['connectivity'] = connectivity[:, 5:]-1


    BCs = {}

    phi = np.arctan2(mesh['coordinates'][:, 1], mesh['coordinates'][:, 0])
    index1 = np.where(phi <= np.pi/3+np.pi/5e3)
    index2 = np.where(phi >= 2*np.pi/3-np.pi/5e3)

    dofs1 = np.hstack([2*index1[0], 2*index1[0]+1])
    dofs2 = np.hstack([2*index2[0], 2*index2[0]+1])
    dofs = np.hstack([dofs1, dofs2])

    BCs['essential'] = np.zeros((len(dofs), 2), dtype=int)
    BCs['essential'][:, 0] = dofs

    index, P = np.where(mesh['coordinates'][:, 1] >= 101.5-1e-3), 1
    BCs['natural'] = np.array([
        [2*index[0][0]+1, -P]
        ])

    return mesh, BCs
