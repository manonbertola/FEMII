import element

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs


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
    damping: Damping
        Damping instance for the finite element model
    

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
        self.damping = Damping()
        self.element = element.Element(self.material, self.section)

    def assemble(self):
        """
        Assembles the system matrices and stores them in the self.equation attribute
    
        Returns
        ----------
        The method modifies the self.equation attribute and updates the system matrices along with the material history parameters
        """

        try:
            self.essential_BCs
        except AttributeError:
            raise AttributeError("Undefined boundary conditions")

        dofs_per_node = self.mesh.dofs_per_node
        nodes_per_element = self.mesh.nodes_per_element

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]

        ndofs = nnodes * dofs_per_node

        self.equation.resetStiffnessAndResidual()
        increment = self.equation.u - self.history.displacements
        displacements = np.reshape(increment, (nnodes, -1))

        for element in range(nelements):

            dofs = np.zeros(dofs_per_node * nodes_per_element, dtype=int)
            nodes = self.mesh.connectivity[element, :]
            dofs = self.element.getDegreesOfFreedom(dofs, nodes)

            x, u = self.mesh.nodes[nodes, :], displacements[nodes, :]

            # Retrieve element history
            stress, strain, kappa, tangent, yielded = self.history.retrieve(element)
            self.element.setHistory(stress, strain, kappa, tangent, yielded)

            Me = self.element.getMass(x)
            Ke, Re = self.element.getStiffnessAndResidual(x, u)

            # Update element history
            self.historyTemp.update(element, self.element.history)

            self.equation.K[np.ix_(dofs, dofs)] += Ke
            self.equation.M[np.ix_(dofs, dofs)] += Me
            self.equation.R[dofs] += Re

        # Apply essential boundary conditions

        dofs = self.essential_BCs[:, 0]
        values = self.essential_BCs[:, 1]

        self.equation.R[dofs] -= self.equation.K[np.ix_(dofs, dofs)].dot(values)
        self.equation.R[dofs] = values

        self.equation.K[dofs, :] = 0
        self.equation.K[:, dofs] = 0
        self.equation.K[dofs, dofs] = 1

        self.equation.M[dofs, :] = 0
        self.equation.M[:, dofs] = 0
        self.equation.M[dofs, dofs] = 1

        # Save the displacement vector
        self.history.displacements = self.equation.u

    def make(self):
        """
        Initializes the necessary ndarrays in the self.equation attributes and the material history parameters
    
        Returns
        ----------
        The method modifies the self.equation attribute to initialize the system matrices and the self.History for the material history variables
        """

        dofs_per_node = self.mesh.dofs_per_node
        nodes_per_element = self.mesh.nodes_per_element

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]
        ndofs = nnodes * dofs_per_node

        self.equation = self.Equation(ndofs)
        self.history = self.History(ndofs, nelements)
        self.historyTemp = self.History(ndofs, nelements)

    def setBoundaryConditions(self, BCs):
        """
        Stores the boundary condition information in the respective attributes
    
        Returns
        ----------
        The method modifies the suitable attributes
        """

        self.essential_BCs = BCs["essential"]
        self.natural_BCs = BCs["natural"]

    def getState(self):
        """
        Extracts the state of the model in terms of displacements, velocities and accelerations
    
        Returns
        ----------
        The method modifies the self.equation attribute to extract the system matrices
        """

        u, v, a = self.equation.u, self.equation.v, self.equation.a

        return u, v, a

    def getSystemMatrices(self, u):
        """
        Evaluates the system matrices of the model

        Parameters
        ----------
        u: ndarray
            The (current) displacement vector

        Returns
        ----------
        K, M, R: ndarray
            The (tangent) stiffness matrix, the mass matrix and the residual internal ones 
        """

        self.equation.u = u

        self.assemble()

        K, M, R = self.equation.K, self.equation.M, self.equation.R

        return K, M, R

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

        plt.axis("equal")

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

        for nodes in self.mesh.connectivity:
            x, u = coordinates[nodes, :], displacements[nodes, :]
            self.element.plot(x, u, scale, alpha)

        plt.axis("equal")

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
        u,v,a: ndarray
            The current displacements, velocities, accelerations
        du: ndarray
            The displacement increment
        K,M,C: ndarray
            The (tangent) stiffness matrix, the mass and the damping matrix

        Methods
        -------
        resetStiffnessAndResidual(self):
            Resets the system matrices
        """
        def __init__(self, ndofs):
            self.R = np.zeros(ndofs)
            self.f = np.zeros(ndofs)
            self.u = np.zeros(ndofs)
            self.v = np.zeros(ndofs)
            self.a = np.zeros(ndofs)
            self.du = np.zeros(ndofs)
            self.K = np.zeros((ndofs, ndofs))
            self.M = np.zeros((ndofs, ndofs))
            self.C = np.zeros((ndofs, ndofs))

        def resetStiffnessAndResidual(self):
            """
            Resets the system matrices
        
            Returns
            ----------
            The method modifies the necessary attributes to reset the system matrices
            """
            self.K = np.zeros_like(self.K)
            self.M = np.zeros_like(self.M)
            self.R = np.zeros_like(self.R)

    class History:
        """
        Object to store all material history related variables

        Attributes
        ----------
        displacements: ndarray
            The last (converged) displacement increment of the model
        stress/strain: ndarray
            The last (converged) stresses/strains for every element
        kappa: ndarray
            The last (converged) hardening parameters for every element
        tangent: ndarray
            The last (converged) tangent constitutive matrices for every element    
        yielded: Boolean    
            Variable to keep track if element has yielded

        Methods
        -------
        update(self, element, history):
            Update the history variables of the element
        retrieve(self,element):
            Retrieve the history variables of the element
        """
        def __init__(self, ndofs, nelements):
            self.displacements = np.zeros(ndofs)
            self.stress = np.zeros((nelements, 2))
            self.strain = np.zeros((nelements, 2))
            self.kappa = np.zeros((nelements, 2))
            self.tangent = np.zeros((nelements, 2))
            self.yielded = np.zeros((nelements, 2))

        def update(self, element, history):
            self.stress[element, :] = history["stress"]
            self.strain[element, :] = history["strain"]
            self.kappa[element, :] = history["kappa"]
            self.tangent[element, :] = history["tangent"]
            self.yielded[element, :] = history["yielded"]

        def retrieve(self, element):
            stress = self.stress[element, :]
            strain = self.strain[element, :]
            kappa = self.kappa[element, :]
            tangent = self.tangent[element, :]
            yielded = self.yielded[element, :]

            return stress, strain, kappa, tangent, yielded


class Damping:
    """
    Class describing the damping of the model

    Attributes
    ----------
    alphas: float
        The alpha coefficient of the Rayleigh method
    betas: float
        The beta coefficient of the Rayleigh method
    zetas: ndarray
        The modal damping rations for the Rayleigh method

    Methods
    -------
    getDampingMatrix(self, Mass, Stiffness):
        Compute the damping matrix using the Rayleigh formulation
    getRayleighCoeffs(Zetas, Omegas):
        Compute the Rayleigh coefficients
    """
    def setRayleighParameters(self, alpha=None, beta=None, zetas=None):
        self.alpha = alpha
        self.beta = beta
        self.zetas = zetas

    def getDampingMatrix(self, Mass, Stiffness):
        """
        Compute the damping matrix using the Rayleigh formulation
        """

        if self.alpha == None and self.beta == None:
            alpha, beta = self.getRayleighDamping()
            Cmatrix = alpha * Stiffness + beta * Mass
        else:
            Cmatrix = self.alpha * Stiffness + self.beta * Mass

        return Cmatrix

    def getRayleighCoeffs(Zetas, Omegas):
        """
        Compute the Rayleigh coefficients

        Parameters
        ----------
        Zetas: ndarray
            The modal damping ratios
        Omegas: ndarray
            The frequencies of interest of the system

        Returns
        -------
        alpha, beta: float
            The Rayleigh damping coefficients
        """

        zeta1, zeta2 = Zetas[0], Zetas[1]
        Omega1, Omega2 = Omegas[0], Omegas[1]

        # Task 2.1: Compute matrices of Rayleigh damping equation

        # Square matrix of coefficients (to be inverted)
        CoeffMatrix = np.array([[Omega1**2, 1], [Omega2**2, 1]])
        # Righ-hand side vector
        rhs = np.array([[2*zeta1*Omega1, 2*zeta2*Omega2]])

        # Task 2.2: Solve equation to compute Rayleigh coeffs
        x = np.linalg.inv(CoeffMatrix).dot(rhs.T)

        alpha, beta = x[0], x[1]

        return alpha, beta


class Material:
    """
    Class describing the material type

    Parameters
    ----------
    E: float
        The Young Modulus
    n: float
        The Poisson ratio
    rho: float
        The material density
    yield_stress: float
        The yielding stress of the material
    Hhardening: float
        The hardening modulus

    Attributes
    ----------
    E: float
        The Young Modulus
    n: float
        The Poisson ratio
    C: ndarray  
        The constitutive matrix of the material
    yield_stress: float
        The yielding stress of the material
    Hhardening: float
        The hardening modulus 

    Methods
    -------
    setProperties(self, E, n):
        Sets the material properties
    """
    def setProperties(self, E, n, rho, yield_stress, Hhardening):
        self.E = E
        self.n = n
        self.rho = rho

        self.yield_stress = yield_stress
        self.Hhardening = Hhardening

    def getYieldFunction(self, stress, kappa):
        """
        Evaluate the yield function 

        Parameters
        ----------
        stress: ndarray
            The current stress vector
        kappa: ndarray
            The current hardening variable

        Returns
        -------
        fy: float
            The evaluation of the yield function
        """

        fy = np.abs(stress) - (self.yield_stress + self.Hhardening * kappa)

        return fy

    def getNormalVector(self, stress):
        """
        Get the vector normal to the yield surface and its derivative.
        
        Parameters
        ----------
        stress: ndarray
            The current stress vector

        Returns
        -------
        m: ndarray
            The vector normal to the yield surface
        dm: ndarray
            The derivative of the normal vector
        """

        m, dm = np.sign(stress), 0

        return m, dm

    def getStressAndTangent(self, stress, kappa, depsilon):
        """
        Perform the return mapping and get the stress and tangent constitutive matrix.

        Parameters
        ----------
        stress: ndarray
            The current stress vector
        kappa: ndarray
            The current hardening variable
        depsilon: ndarray
            The strain increment

        Returns
        -------
        stress_upd: ndarray
            The updated stress
        kappa_upd:ndarray
            The updated hardening variable
        tangentC: ndarray
            The updated tangent constitutive matrix       
        yielded: Boolean
            Index to track if element has yielded or not
        """

        tol = 1e-4
        yielded = False
        prediction = stress + self.E * depsilon
        yfunction = self.getYieldFunction(prediction, kappa)

        if yfunction <= tol * self.yield_stress:
            sigma_upd = prediction
            kappa_upd = kappa
            tangent = self.E
        else:

            m, dm = self.getNormalVector(stress)
            p = 1
            sigma_upd = prediction
            kappa_upd = kappa

            deltaLambda_upd, epsilon_s, epsilon_k = 0, 0, 0
            epsilon_f = self.getYieldFunction(sigma_upd, kappa_upd)
            residual = np.hstack([epsilon_s, epsilon_k, np.abs(epsilon_f)])

            maxit, counter = 20, 0

            while np.linalg.norm(residual) > tol * self.yield_stress and counter < maxit:

                jacobian = np.array([
                    [1 + self.E*dm*float(deltaLambda_upd),       0, self.E*float(m)],
                    [                           0,       1,              -p],
                    [                    float(m), -self.Hhardening,               0]
                    ])

                # Incremental values
                delta_unknowns = np.linalg.inv(jacobian).dot(residual)
                sigma_upd = sigma_upd - delta_unknowns[0]
                kappa_upd = kappa_upd - delta_unknowns[1]
                deltaLambda_upd = deltaLambda_upd - delta_unknowns[2]

                m, dm = self.getNormalVector(sigma_upd)

                epsilon_s = sigma_upd - prediction + self.E*m*deltaLambda_upd
                epsilon_k = kappa_upd - kappa - p*deltaLambda_upd

                epsilon_f = self.getYieldFunction(sigma_upd, kappa_upd)
                residual = np.hstack([epsilon_s, epsilon_k, np.abs(epsilon_f)])

                counter = counter + 1

            jacobian_inverse = np.linalg.inv(jacobian)
            tangent = -jacobian_inverse[0, 0]*self.E
            yielded = True

        return sigma_upd, kappa_upd, tangent, yielded


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
    def setProperties(self, A, I1=None, I2=None, I3=None):
        self.A = A
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3


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
