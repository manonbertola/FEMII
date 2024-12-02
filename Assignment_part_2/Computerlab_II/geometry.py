import numpy as np



def getBar():

    mesh = {}
    mesh['coordinates'] = np.array([[0, 1], [1, 1], [1, 0]])
    mesh['connectivity'] = np.array([[0, 1], [1, 2]])

    BCs = {}
    BCs['essential'] = np.array([
        [0, 0],
        [1, 0],
        [4, 0],
        [5, 0]
        ])

    P = 1
    BCs['natural'] = np.array([
        [2, P]
        ])

    return mesh, BCs



def getTruss():

    # Number of elements in x and y directions
    nx, ny = 20, 4

    # Lenght in x and y directions
    lx, ly = 20, 4 

    coordinates = np.zeros(((nx+1)*(ny+1), 2))

    counter = 0
    dlx, dly = lx/nx, ly/ny

    for j in range(ny+1):
        for i in range(nx+1):
            coordinates[counter, :] = np.array([i*dlx, j*dly])
            counter = counter+1
        

    counter = 0
    connectivity = np.zeros((nx*(1+ny*3), 2))

    for j in range(ny):
        for i in range(nx):
            connectivity[counter, :] = np.array([i+j*(nx+1), i+1+j*(nx+1)])
            connectivity[counter+1, :] = np.array([i+1+j*(nx+1), i+1+(j+1)*(nx+1)])
            connectivity[counter+2, :] = np.array([i+j*(nx+1), i+1+(j+1)*(nx+1)])
            counter = counter + 3
        

    for i in range(nx):
        connectivity[counter, :] = np.array([ny*(nx+1)+i, ny*(nx+1)+i+1])
        counter = counter + 1


    mesh = {}
    mesh['coordinates'] = coordinates
    mesh['connectivity'] = connectivity


    BCs = {}

    index = np.where(coordinates[:, 0] == 0)
    dofs = np.hstack([2*index[0], 2*index[0]+1])

    BCs['essential'] = np.zeros((len(dofs), 2), dtype=int)
    BCs['essential'][:, 0] = dofs

    P = 1
    index1 = set(np.where(coordinates[:, 0] > 0.999*lx)[0])
    index2 = set(np.where(coordinates[:, 1] > 0.999*ly)[0])
    index = list(index1.intersection(index2))

    BCs['natural'] = np.array([
        [2*index[0]+1, P]
        ])

    return mesh, BCs


