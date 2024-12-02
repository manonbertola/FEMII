import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


class Quad4:
    """
    Element class

    Parameters
    ----------
    _nodes : int
        Number of element nodes
    material: Material
        The type of material instance representing the element's material
    section: Section
        The type of section instance representing the element's cross-section

    Attributes
    ----------
    material: Material
        The type of material instance representing the element's material
    section: Section
        The type of section instance representing the element's cross-section
    history: Dictionary
        A dictionary containing the material history variables of the previous converged increment required for the return mapping
    history_temp: Dictionary
        A dictionary containing the material history variables of the last return mapping iteration

    Methods
    -------
    setHistory(self, stress, kappa, tangentC, yielded):
        Set the material history parameters and variables
    getStiffnessAndResidual(self):
        Get the (tangent) stiffness matrix and the internal (residual) forces and update material history
    getDegreesOfFreedom(self, dofs, nodes):
        Evaluate the degree of freedom numbering
    plot(self, coordinates, displacements, scale, alpha):
        Visualize the element in its (un)deformed state
    getLinearDeformationMatrix(self, coordinates, r, s):
        Evaluates the linear portion of the strain-displacements matrix
    getShapeFunctionDerivatives(self, r, s):
        Evaluates the derivatives of the element's shape functions

    """

    _nodes = 4

    def __init__(self, material, section):
        self.material = material
        self.section = section

        self.history = {}
        self.history_temp = {}
        self.setHistory(
            np.zeros((4, 3)),    #Stress values per Gauss point
            np.zeros(4),         #Kappa hardening parameter per Gauss point
            np.zeros((4, 3, 3)), #Tangent constitutive matrix per Gauss point
            np.zeros(4)          #Boolean variable to track yield per Gauss point
            )


    def setHistory(self, stress, kappa, tangentC, yielded):
        """
        Set the material history parameters and variables by updating the instance

        Parameters
        ----------
        stress: ndarray
            The stress vector
        kappa: ndarray
            The hardening parameter
        tangentC: ndarray
            The tangent constitutive matrix
        yielded: Boolean
            Variable indicating if yielding has occured
        """

        self.history['stress'] = stress
        self.history['kappa'] = kappa
        self.history['tangent'] = tangentC
        self.history['yielded'] = yielded

    def getStiffnessAndResidual(self, coordinates, displacements):
        """
        Get the (tangent) stiffness matrix and the internal (residual) forces and updates the material history dictionary

        Parameters
        ----------
        coordinates: ndarray
            Array of nodal coordinates
        displacements: ndarray
            Array containing the displacements of each node
    
        Returns
        ----------
        K: ndarray
            The (tangent) stiffness matrix
        R: ndarray
            The internal (residual) forces
        """

        try: 
            C = self.material.C_plane_stress
        except AttributeError:
            raise AttributeError('undefined material properties')
            
            
        #Gauss points coordinates
        points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)],
            [ 1/np.sqrt(3), -1/np.sqrt(3)],
            [-1/np.sqrt(3),  1/np.sqrt(3)],
            [ 1/np.sqrt(3),  1/np.sqrt(3)]
            ])
        
        #Gauss points weights
        weights = np.array([1, 1, 1, 1])
        
        #Initialization of system matrices
        K = np.zeros((self._nodes*2, self._nodes*2))
        R = np.zeros((self._nodes*2,1))
        R = np.zeros(self._nodes*2)
        du = displacements.flatten()

        for j, (point, weight) in enumerate(zip(points, weights)):
            
            # Get the Gauss-point coordinates
            r, s = point[0], point[1]

            # Get the linear deformation matrix
            B, det, derivatives = self.getLinearDeformationMatrix(
                coordinates, r, s)

            # Incremental strain
            depsilon = B.dot(du)

            # Retrieve Gauss Point history
            sigma = self.history['stress'][j, :]
            kappa = self.history['kappa'][j]

            # Perform the return mapping
            sigma, kappa, tangentC, yielded = self.material.getStressAndTangent(sigma, kappa, depsilon)
            
            # Update tangent stiffness and residual
            K = K + weight*B.T.dot(tangentC).dot(B)*det
            R = R + weight*B.T.dot(sigma)*det

            # Update history
            self.history['stress'][j, ...] = sigma
            self.history['kappa'][j] = kappa
            self.history['tangent'][j, ...] = tangentC
            self.history['yielded'][j] = yielded

        return K, R



    def getDegreesOfFreedom(self, dofs, nodes):
        """
        Evaluates the degree of freedom numbering

        Parameters
        ----------
        dofs: ndarray
            Initialized array with the size of degrees of freedom
        nodes: ndarray
            Array containing the connectivity of the mesh

        Returns
        ----------
        dofs: ndarray
            Array with the degrees of freedom numbering
        """
        
        for j in range(2):
            dofs[j::2] = j+nodes*2

        return dofs


    def getLinearDeformationMatrix(self, coordinates, r, s):
        """
        Evaluates the degree of freedom numbering

        Parameters
        ----------
        coordinates: ndarray
            Array of nodal coordinates
        s , t: ndarray
            Gauss points for the integration
        
        Returns
        ----------
        B: ndarray
            Array with the linear portion of the strain-displacement matrix
        determinant : ndarray
            Array with the jacobian determinant       
        derivatives: ndarray
            Array with the shape function derivatives
        """
        
        # Get the local shape function derivatives
        derivatives = self.getShapeFunctionDerivatives(r, s)

        # Get the Jacobian matrix and the determinant
        jacobian = derivatives.dot(coordinates)
        determinant = np.linalg.det(jacobian)

        # Get the shape function derivatives
        derivatives = np.linalg.inv(jacobian).dot(derivatives)

        B = np.zeros((3, self._nodes*2))

        for j in range(self._nodes):
            B[:, 2*j:2*j+2] = np.array([
                [derivatives[0, j], 0],
                [0, derivatives[1, j]],
                [derivatives[1, j], derivatives[0, j]]
                ])

        return B, determinant, derivatives


    def getShapeFunctionDerivatives(self, r, s):
        """
        Evaluates the shape function derivatives

        Parameters
        ----------
        r, s: ndarray
            Gauss points for the integration
        
        Returns
        ----------
        derivatives: ndarray
            Array with the shape function derivatives
        """
        
        derivatives = np.array(
            [
                [1 / 4 * (s - 1), 1 / 4 * (1 - s), 1 / 4 * (s + 1), -1 / 4 * (s + 1)],
                [1 / 4 * (r - 1), -1 / 4 * (r + 1), 1 / 4 * (r + 1), -1 / 4 * (r - 1)],
            ]
        )

        return derivatives


    def plot(self, coordinates, displacements, scale, alpha):
        """
        Plots the element

        Parameters
        ----------
        coordinates: ndarray
            Array of nodal coordinates
        displacements: ndarray
            Array containing the displacements of each node
        scale: float
            Scaling factor for visualization of deformed configuration
        alpha: float
            Parameter controlling the opacity of the visualization
        """
        
        x = np.hstack((coordinates[:, 0],coordinates[0, 0]))
        xd = np.hstack((displacements[:, 0],displacements[0, 0]))
        y = np.hstack((coordinates[:, 1],coordinates[0, 1]))
        yd = np.hstack((displacements[:, 1],displacements[0, 1]))
        
        plt.plot(x+scale*xd, y+scale*yd, '-o', alpha=alpha)
