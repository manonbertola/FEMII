import element

import numpy as np
import matplotlib.pyplot as plt



class Model:
    """
    Object describing the finite element model

    Parameters
    ----------
    mesh: Mesh
        The type of mesh instance for the finite element model
    material: Material
        The type of material instance for the finite element model
    section: Section
        The type of cross-section instance for the finite element model
    element: Element
        The type of element instance for the finite element model

    Attributes
    ----------
    mesh: Mesh
        The type of mesh instance for the finite element model
    material: Material
        The type of material instance for the finite element model
    section: Section
        The type of cross-section instance for the finite element model
    element: Element
        The type of element instance for the finite element model
    equation: Equation
        An equation instance to store the system matrices for the finite element model
    equation.*: ndarray
        Arrays containing the system matrices and vectors of the model
    essential_BCs/natural_BCs: Dictionary
        Dictionaries containing the boundary conditions of the model
    

    Methods
    -------
    assemble(self):
        Assembles the system matrices and stores them in the self.equation attribute
    make(self):
        Initializes the necessary ndarrays in the self.equation attributes
    setBoundaryConditions(self, BCs):
        Stores the boundary condition information in the respective attributes
    getState(self):
        Extracts the state of the model in terms of displacemens and load factors
    getSystemMatrices(self, u, lfactor):
        Evaluates the system matrices of the model
    plotUndeformed(self):
        Plots the undeformed configuration of the model
    plotDeformed(self, scale=1, alpha=1):
        Ploes the deformed configuration of the model
    """

    def __init__(self):
        self.mesh = Mesh()
        self.material = Material()
        self.section = Section()
        self.element = element.Element(self.material, self.section)



    def assemble(self):
        """
        Assembles the system matrices and stores them in the self.equation attribute
    
        Returns
        ----------
        The method modifies the self.equation attribute and updates the system matrices
        """

        try:
            self.essential_BCs
        except AttributeError:
            raise AttributeError('Undefined boundary conditions')

        dofs_per_node = self.mesh.dofs_per_node
        nodes_per_element = self.mesh.nodes_per_element

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]

        ndofs = nnodes*dofs_per_node

        self.equation.resetStiffnessAndResidual()
        displacements = np.reshape(self.equation.u, (nnodes, -1))

        for element in range(nelements):

            dofs = np.zeros(dofs_per_node*nodes_per_element, dtype=int)
            nodes = self.mesh.connectivity[element, :]
            dofs = self.element.getDegreesOfFreedom(dofs, nodes)

            x, u = self.mesh.nodes[nodes, :], displacements[nodes, :]
            Ke, Re = self.element.getStiffnessAndResidual(x, u)

            self.equation.K[np.ix_(dofs, dofs)] += Ke
            self.equation.R[dofs] += Re


        # Apply essential boundary conditions

        dofs = self.essential_BCs[:, 0]
        values = self.essential_BCs[:, 1]

        self.equation.R[dofs] -= self.equation.K[np.ix_(dofs, dofs)].dot(values)
        self.equation.R[dofs] = values

        self.equation.K[dofs, :] = 0
        self.equation.K[:, dofs] = 0
        self.equation.K[dofs, dofs] = 1

        # Apply natural boundary conditions

        self.equation.f[self.natural_BCs[:, 0]] = self.natural_BCs[:, 1]

        # Substract external loads
        self.equation.R -= self.equation.lfactor*self.equation.f



    def make(self):
        """
        Initializes the necessary ndarrays in the self.equation attributes
    
        Returns
        ----------
        The method modifies the self.equation attribute to initialize the system matrices
        """

        dofs_per_node = self.mesh.dofs_per_node
        nodes_per_element = self.mesh.nodes_per_element

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]
        ndofs = nnodes*dofs_per_node

        self.equation = self.Equation(ndofs)



    def setBoundaryConditions(self, BCs):
        """
        Stores the boundary condition information in the respective attributes
    
        Returns
        ----------
        The method modifies the suitable attributes
        """
        
        self.essential_BCs = BCs['essential']
        self.natural_BCs = BCs['natural']



    def getState(self):
        """
        Extracts the state of the model in terms of displacemens and load factors
    
        Returns
        ----------
        The method modifies the self.equation attribute to extract the system matrices
        """


        u, lfactor = self.equation.u, self.equation.lfactor

        return u, lfactor #self.equation.u, self.equation.lfactor



    def getSystemMatrices(self, u, lfactor):
        """
        Evaluates the system matrices of the model

        Parameters
        ----------
        u: ndarray
            The (current) displacement vector
        lfactor:ndarray
            The load factor

        Returns
        ----------
        K, f, R: ndarray
            The (tangent) stiffness matrix, the external forces and the residual internal ones 
        """

        self.equation.u = u
        self.equation.lfactor = lfactor

        self.assemble()

        K, f, R = self.equation.K, self.equation.f, self.equation.R

        return K, f, R



    def plotUndeformed(self):
        """
        Plots the undeformed configuration of the model
        """

        coordinates = self.mesh.nodes
        displacements = np.zeros_like(coordinates)

        plt.figure()
        
        for nodes in self.mesh.connectivity:
            x, u = coordinates[nodes, :], displacements[nodes, :]
            self.element.plot(x, u, 0, 1)

        plt.axis('equal')



    def plotDeformed(self, scale=1, alpha=1):
        """
        Plots the deformed configuration of the model

        Parameters
        ----------
        scale: float
            Scaling factor for visualization of deformed configuration
        alpha: float
            Parameter controlling the opacity of the visualization
        """

        nnodes = self.mesh.nodes.shape[0]
        coordinates = self.mesh.nodes
        displacements = np.reshape(self.equation.u, (nnodes, -1))

        # plt.figure()
        
        for nodes in self.mesh.connectivity:
            x, u = coordinates[nodes, :], displacements[nodes, :]
            self.element.plot(x, u, scale, alpha)

        plt.axis('equal')


    class Equation:
        """
        Object to store system vectors and matrices

        Parameters
        ----------
        ndofs: float
            The number of degrees of freedom

        Attributes
        ----------
        lfactor : float
            The load factor value
        R: ndarray
            The internal (residual) forces
        f: ndarray
            The external forcing
        u: ndarray
            The current displacements
        du: ndarray
            The displacement increment
        K: ndarray
            The (tangent) stiffness matrix

        Methods
        -------
        resetStiffnessAndResidual(self):
            Resets the system matrices
        """

        def __init__(self, ndofs):
            self.lfactor = 0
            self.R = np.zeros(ndofs)
            self.f = np.zeros(ndofs)
            self.u = np.zeros(ndofs)
            self.du = np.zeros(ndofs)
            self.K = np.zeros((ndofs, ndofs))


        def resetStiffnessAndResidual(self):
            """
            Resets the system matrices
        
            Returns
            ----------
            The method modifies the necessary attributes to reset the system matrices
            """
            self.K = np.zeros_like(self.K)
            self.R = np.zeros_like(self.R)



class Material:
    """
    Class describing the material type

    Parameters
    ----------
    E: float
        The Young Modulus
    n: float
        The Poisson ratio

    Attributes
    ----------
    E: float
        The Young Modulus
    n: float
        The Poisson ratio

    Methods
    -------
    setProperties(self, E, n):
        Sets the material properties
    """

    def setProperties(self, E, n):
        self.E = E
        self.n = n



class Section:
    """
    Class describing the cross-section type

    Parameters
    ----------
    A: float
        The cross-sectional area

    Attributes
    ----------
    A: float
        The cross-sectional area

    Methods
    -------
    setProperties(self,A):
        Sets the cross-section properties
    """

    def setProperties(self, A):
        self.A = A



class Mesh:
    """
    Class describing the mesh of the model

    Attributes
    ----------
    dofs_per_node: int
        The number of degrees of freedom per node
    nodes_per_element: int
        The number of nodes per element
    nodes: ndarray
        The nodal coordinates
    connectivity: ndarray
        The connectivity matrix of the mesh

    Methods
    -------
    addNodes(self,nodes):
        Stores the nodal coordinates array
    addConnectivity(self, connectivity):
        Stores the nodal connectivity array
    """
    

    def __init__(self):
        self.dofs_per_node = 2
        self.nodes_per_element = 2


    def addNodes(self, nodes):
        self.nodes = nodes


    def addConnectivity(self, connectivity):
        self.connectivity = connectivity.astype(int)

