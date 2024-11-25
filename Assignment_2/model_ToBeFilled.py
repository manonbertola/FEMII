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
    history: History (dedicated class)
        The history instance containing all material history variables and parameters
    equation: Equation
        An equation instance to store the system matrices for the finite element model
    equation.* (e.g. equation.K): ndarray
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
        Plots the deformed configuration of the model
    """

    def __init__(self):
        self.mesh = Mesh()
        self.material = Material()
        self.section = Section()
        self.element = element.Quad4(self.material,self.section)


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
            raise AttributeError('Undefined boundary conditions')

        dofs_per_node = self.mesh.dofs_per_node
        nodes_per_element = self.mesh.nodes_per_element

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]

        ndofs = nnodes*dofs_per_node

        self.equation.resetStiffnessAndResidual()
        increment = self.equation.u - self.history.displacements
        increment = np.reshape(increment, (nnodes, -1))

        for element in range(nelements):

            dofs = np.zeros(dofs_per_node*nodes_per_element, dtype=int)
            nodes = self.mesh.connectivity[element, :]
            dofs = self.element.getDegreesOfFreedom(dofs, nodes)

            x, du = self.mesh.nodes[nodes, :], increment[nodes, :]
            
            # Retrieve element histoy
            stress, kappa, tangent, yielded = self.history.retrieve(element)
            self.element.setHistory(stress, kappa, tangent, yielded)

            Ke, Re = self.element.getStiffnessAndResidual(x, du)

            # Do not update element history here
            # Just store a temporary version of the history
            self.history_temp.update(element, self.element.history)

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

        nelements = self.mesh.connectivity.shape[0]
        nnodes = self.mesh.nodes.shape[0]
        ndofs = nnodes*dofs_per_node

        self.equation = self.Equation(ndofs)
        self.history = self.History(ndofs, nelements)
        self.history_temp = self.History(ndofs, nelements)



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

        return u, lfactor



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



    def plotUndeformed(self, size=(16, 4)):
        """
        Plots the undeformed configuration of the model
        """

        coordinates = self.mesh.nodes
        displacements = np.zeros_like(coordinates)

        plt.figure(figsize=size)
        
        for nodes in self.mesh.connectivity:
            x, u = coordinates[nodes, :], displacements[nodes, :]
            self.element.plot(x, u, 0, 1)

        plt.axis('equal')

    def UpdateHistory(self):
        """
        Updates the self.history field of the model containing the material parameters for every element
        """
        
        nelements = self.mesh.connectivity.shape[0]
        for element in range(nelements):
            stress, kappa, tangent, yielded = self.history_temp.retrieve(element)
            self.element.setHistory(stress, kappa, tangent, yielded)
            self.history.update(element, self.element.history)


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


    class History:
        """
        Object to store all material history related variables

        Attributes
        ----------
        displacements: ndarray
            The last (converged) displacement increment of the model
        stress: ndarray
            The last (converged) stresses for every element
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

            self.stress = np.zeros((nelements, 4, 3))
            self.kappa = np.zeros((nelements, 4))
            self.tangent = np.zeros((nelements, 4, 3, 3))
            self.yielded = np.zeros((nelements, 4))


        def update(self, element, history):
            self.stress[element, ...] = history['stress']
            self.kappa[element, ...] = history['kappa']
            self.tangent[element, ...] = history['tangent']
            self.yielded[element, ...] = history['yielded']


        def retrieve(self, element):
            stress = self.stress[element, ...]
            kappa = self.kappa[element, ...]
            tangent = self.tangent[element, ...]
            yielded = self.yielded[element, ...]

            return stress, kappa, tangent, yielded



class Material:
    """
    Class describing the material type

    Parameters
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

    def setProperties(self, E, n, yield_stress, Hhardening):
        self.E = E
        self.n = n
        self.C_plane_stress = E/(1-n**2)*np.array([
            [1, n, 0],
            [n, 1, 0],
            [0, 0, (1-n)/2]
        ])


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

        # TO DO (2.1.1) Compute the effective Von Mises stress
        #need to modify for plane stress
        #P = np.array([[ 2/3, -1/3, -1/3,   0,   0,   0],
                      #[-1/3,  2/3, -1/3,   0,   0,   0],
                      #[-1/3, -1/3,  2/3,   0,   0,   0],
                      #[   0,    0,    0,   2,   0,   0],
                      #[   0,    0,    0,   0,   2,   0],
                      #[   0,    0,    0,   0,   0,   2],
                      #])
        P = np.array([[ 2/3, -1/3, -1/3,   0,   0,   0],
                      [-1/3,  2/3, -1/3,   0,   0,   0],
                      [-1/3, -1/3,    0,   0,   0,   0],
                      [   0,    0,    0,   2,   0,   0],
                      [   0,    0,    0,   0,   0,   0],
                      [   0,    0,    0,   0,   0,   0],
                      ])
        VMstress = np.sqrt(1.5 * stress.T @ P @ stress)
        #VMstress = ...

        # TO DO (2.1.2) Evaluate the yield function
        fy = VMstress - (self.yield_stress + self.Hhardening * kappa)
        #fy = ...
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
        # TO DO (2.2.1) Compute the effective Von Mises stress
        #VMstress = ...
        #need to modify for plane stress
        #P = np.array([[ 2/3, -1/3, -1/3,   0,   0,   0],
                      #[-1/3,  2/3, -1/3,   0,   0,   0],
                      #[-1/3, -1/3,  2/3,   0,   0,   0],
                      #[   0,    0,    0,   2,   0,   0],
                      #[   0,    0,    0,   0,   2,   0],
                      #[   0,    0,    0,   0,   0,   2],
                      #])
        P = np.array([[ 2/3, -1/3, -1/3,   0,   0,   0],
                      [-1/3,  2/3, -1/3,   0,   0,   0],
                      [-1/3, -1/3,    0,   0,   0,   0],
                      [   0,    0,    0,   2,   0,   0],
                      [   0,    0,    0,   0,   0,   0],
                      [   0,    0,    0,   0,   0,   0],
                      ])
        VMstress = np.sqrt(1.5 * np.dot(np.dot(stress.T , P ), stress)) #scalar

        # TO DO (2.2.2) Compute the vector normal to the yield surface
        #m = ...
        m = 3 * np.dot(P , stress) / (2*VMstress) #6x1
        
         # TO DO (2.2.2) Compute the derivative of the normal vector
        dm = (6 * P * VMstress - 6 * np.dot(np.dot(P , stress) , m.T)) / (4 * VMstress) #6x6 not sure if m needs transpose
        #dm = ...

        return m,dm

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
        
        #Variable to indicate if yielding has occur
        yielded = False
        #Tolerance variables
        tolerance_yield, tolerance_residual = 1e-3, 1e-3
        #Define hardening scalar function
        p = 1
        #Evaluate elastic constitutive matrix
        De = self.C_plane_stress

        # TO DO (2.3.1) Stress estimation employing the elastic predictor
        #elastic_stress_prediction = ...
        # need to flatten De somehow here i think, unless stress here is 3x3
        De_flat = np.array([[De[0,0], De[1,1], De[2,2], De[0,1], De[0,2], De[1,2]]])
        elastic_stress_prediction = stress + De_flat @ depsilon
        # TO DO (2.3.2) Evaluate yield function at estimation
        yfunction = self.getYieldFunction(elastic_stress_prediction, kappa)
        #yfunction = ...

        if yfunction <= tolerance_yield*self.yield_stress: #Evaluate if yielding has occur
            #Elastic regime
            #TO DO (2.3.3) Update corresponding values directly
            stress_upd = elastic_stress_prediction
            kappa_upd = kappa
            tangentC = De
        else:
            #Plastic regime - Initiate return mapping

            # TO DO (2.3.4) Compute normal vector to the yielding surface
            m,dm = self.getNormalVector(stress)
            #m,dm = self.getNormalVector(...)

            # TO DO (2.3.5) Initialize values
            stress_upd = elastic_stress_prediction
            kappa_upd = kappa
            deltaLambda_upd = 0

            # TO DO (2.3.6): Calculate stress and hardening residuals
            epsilon_s = np.linalg.norm(stress - elastic_stress_prediction + De_flat @ m * deltaLambda_upd) #should this be a scalar?
            epsilon_k = kappa_upd - kappa - deltaLambda_upd * p

            # TO DO (2.3.7): Evaluate yield function
            epsilon_f = self.getYieldFunction(stress_upd, kappa_upd)
            # TO DO (2.3.8): Form the total residual vector
            residuals = np.hstack(([epsilon_s, epsilon_k, epsilon_f]))

            maxit, counter = 50, 0

            while np.linalg.norm(residuals) > tolerance_residual*self.yields and counter < maxit:
                # TO DO (2.3.9): Construct Jacobian matrix
                jacobian = np.zeros((5,5))
                I = np.array([[1,0,0],
                              [0,1,0],
                              [0,0,1]]
                            )
                #stress derivation matrix
                jacobian[:2,:2]=I+np.dot(De, dm) 
                jacobian[0,0]=0
                jacobian[0,1]=0
                jacobian[0,2]=0
                jacobian[1,0]=0
                jacobian[1,1]=0
                jacobian[1,2]=0
                jacobian[2,0]=0
                jacobian[2,1]=0
                jacobian[2,2]=0



                jacobian = np.array([[I + De_flat @ dm * deltaLambda_upd, De_flat * (-self.Hhardening) * deltaLambda_upd, De_flat @ m],
                                     []])
                
                # TO DO (2.3.10): Compute incremental values
                delta_unknowns = ...
                # TO DO (2.3.11): Update quantities
                stress_upd = ...
                kappa_upd = ...
                deltaLambda_upd = ...


                # TO DO (2.3.12) Update normal vector
                m, dm = self.getNormalVector(...)

                # TO DO (2.3.13) Update residuals based on computed values
                epsilon_s = ...
                epsilon_k = ...
                # TO DO (2.3.14) Evaluate yield function at updated prediction
                epsilon_f = self.getYieldFunction(...)
                # TO DO (2.3.15): Update residual vector
                residuals = np.hstack((...))
            
                counter = counter + 1

            # TO DO (2.3.16): Update tangent constitutive matrix
            tangentC = ...
            yielded = True

        return stress_upd, kappa_upd, tangentC, yielded



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
        self.nodes_per_element = 4


    def addNodes(self, nodes):
        self.nodes = nodes


    def addConnectivity(self, connectivity):
        self.connectivity = connectivity.astype(int)
