import numpy as np



def getMisesTruss():

    mesh = {}
    mesh['coordinates'] = np.array([[-1, 0], [1, 0], [0, np.sqrt(3)/3]])
    mesh['connectivity'] = np.array([[0, 2], [2, 1]])

    BCs = {}
    BCs['essential'] = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0]
        ])

    P = 1
    BCs['natural'] = np.array([
        [5, -P]
        ])

    return mesh, BCs



def getArcTruss():

    # Define nodes

    n_phi = 4
    nr = 1

    R1 = 49
    R2 = 51

    phi = np.pi/6

    phi_1 = np.pi/2 + phi/2
    phi_2 = np.pi/2 - phi/2

    dR = (R2-R1)/nr
    dphi = (phi_2-phi_1)/n_phi

    coordinates = np.zeros(((n_phi+1)*(nr+1), 2))

    c = 0

    for i in range(nr+1):
        for j in range(n_phi+1):
            coordinates[c, 0] = (R1+i*dR)*np.cos(phi_1+j*dphi)
            coordinates[c, 1] = (R1+i*dR)*np.sin(phi_1+j*dphi)
            c += 1


    # Define elements connectivity

    nel = n_phi*(2*nr+1) + (n_phi-1)*nr
    connectivity = np.zeros((nel, 2))

    cn = 0
    ce = 0

    for i in range(nr+1):
        for j in range(n_phi+1):

            if j < n_phi/2 and i < nr:
                connectivity[ce, :] = np.array([cn, cn+1])
                connectivity[ce+1, :] = np.array([cn, cn+n_phi+2])
                ce += 2

            if j >= n_phi/2 and j < n_phi and i < nr:
                connectivity[ce, :] = np.array([cn, cn+1])
                connectivity[ce+1, :] = np.array([cn+1, cn+n_phi+1])
                ce += 2

            if i == nr and j < n_phi:
                connectivity[ce, :] = np.array([cn, cn+1])
                ce += 1
                
            if j > 0 and j < n_phi and i < nr:
                connectivity[ce, :] = np.array([cn, cn+n_phi+1])
                ce += 1

            cn += 1


    mesh = {}
    mesh['coordinates'] = coordinates
    mesh['connectivity'] = connectivity


    BCs = {}

    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    index1, index2 = np.where(phi == phi_1), np.where(phi == phi_2)

    dofs1 = np.hstack([2*index1[0], 2*index1[0]+1])
    dofs2 = np.hstack([2*index2[0], 2*index2[0]+1])
    dofs = np.hstack([dofs1, dofs2])

    BCs['essential'] = np.zeros((len(dofs), 2), dtype=int)
    BCs['essential'][:, 0] = dofs


    index = np.where(coordinates[:, 1] >= R2-1e-8*R2)
    P = 1
    BCs['natural'] = np.array([
        [2*index[0][0]+1, -P]
        ])


    return mesh, BCs

