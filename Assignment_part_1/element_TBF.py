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

    Methods
    -------
    getStiffnessAndResidual(self):
        Get the (tangent) stiffness matrix and the internal (residual) forces
    getDegreesOfFreedom(self, dofs, nodes):
        Evaluate the degree of freedom numbering
    plot(self, coordinates, displacements, scale, alpha):
        Visualize the element in its (un)deformed state
    getLinearDeformationMatrix(self, coordinates, displacements, s, t):
        Evaluates the linear portion of the strain-displacements matrix
    getNonlinearDeformationMatrix(self, displacements, derivatives):
        Evaluates the nonlinear portion of the strain-displacement matrix
    getShapeFunctionDerivatives(self, s, t):
        Evaluates the derivatives of the element's shape functions

    """

    _nodes = 4

    def __init__(self, material, section):
        self.material = material
        self.section = section

    def getStiffnessAndResidual(self, coordinates, displacements):
        """
        Get the (tangent) stiffness matrix and the internal (residual) forces
    
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
            raise AttributeError("undefined material properties")

        # TO DO (1.4.1): Select the Gauss-integration points and weights
        points = np.array([
                    [-1/np.sqrt(3), -1/np.sqrt(3)],
                    [1/np.sqrt(3), -1/np.sqrt(3)],
                    [1/np.sqrt(3), 1/np.sqrt(3)],
                    [-1/np.sqrt(3), 1/np.sqrt(3)]
                ])
        weights = np.array([[1], [1], [1], [1]])
        #points = ...
        #weights = ...

        Kl = np.zeros((self._nodes * 2, self._nodes * 2))
        Knl = np.zeros((self._nodes * 2, self._nodes * 2))
        R = np.zeros(self._nodes * 2)
        displacements = displacements.flatten()

        for point, weight in zip(points, weights):

            # Get the Gauss-point coordinates
            r, s = point[0], point[1]

            # Get the linear deformation matrix
            Blinear, det, derivatives = self.getLinearDeformationMatrix(
                coordinates, displacements, r, s
            )

            # TO DO (1.4.2): Update the linear stiffness matrix
            Kl += weight * (Blinear.T @ C @ Blinear) * det
            #Kl = ...
            
            # Get the nonlinear deformation matrix
            Bnonlinear, Egreenlagrance = self.getNonlinearDeformationMatrix(displacements, derivatives)

            # TO DO (1.4.3): Calculate the Second Piola-Kirchoff stress
            S_vector = C.dot(Egreenlagrance).flatten()
            S_matrix = np.array([
                        [S_vector[0], 0, 0, 0],
                        [0, S_vector[2], 0, 0],
                        [0, 0, S_vector[2], 0],
                        [0, 0, 0, S_vector[1]]
                    ])
            #S_matrix = ...

            # TO DO (1.4.4): Update the nonlinear stiffness matrix
            Knl += weight * (Bnonlinear.T @ S_matrix @ Bnonlinear) * det

            # TO DO (1.4.5): Update the internal force vector
            R += weight * (Blinear.T @ S_vector) * det
            #R = ...

        K = Kl + Knl

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
            dofs[j::2] = j + nodes * 2

        return dofs

    def getLinearDeformationMatrix(self, coordinates, displacements, r, s):
        """
        Evaluates the degree of freedom numbering

        Parameters
        ----------
        coordinates: ndarray
            Array of nodal coordinates
        displacements: ndarray
            Array containing the displacements of each node
        s , t: ndarray
            Gauss points and weights for the integration
        
        Returns
        ----------
        Blinear: ndarray
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

        # TO DO (1.2.1): Get the linear deformation matrix
        dN_dr = derivatives[0]  # dNi/dr
        dN_ds = derivatives[1]  # dNi/ds
        B_L0 = np.zeros((3, 8))
        
        B_L0[0, 0] = dN_dr[0]
        B_L0[0, 2] = dN_dr[1]
        B_L0[0, 4] = dN_dr[2]
        B_L0[0, 6] = dN_dr[3]
    
        B_L0[1, 1] = dN_ds[0]
        B_L0[1, 3] = dN_ds[1]
        B_L0[1, 5] = dN_ds[2]
        B_L0[1, 7] = dN_ds[3]
    
        B_L0[2, 0] = dN_ds[0]
        B_L0[2, 1] = dN_dr[0]
        B_L0[2, 2] = dN_ds[1]
        B_L0[2, 3] = dN_dr[1]
        B_L0[2, 4] = dN_ds[2]
        B_L0[2, 5] = dN_dr[2]
        B_L0[2, 6] = dN_ds[3]
        B_L0[2, 7] = dN_dr[3]
        
        # Extract displacements for each node in each direction
        u1_1, u1_2 = displacements[0], displacements[1]
        u2_1, u2_2 = displacements[2], displacements[3]
        u3_1, u3_2 = displacements[4], displacements[5]
        u4_1, u4_2 = displacements[6], displacements[7]
        
        # Compute lij
        l11 = dN_dr[0] * u1_1 + dN_dr[1] * u2_1 + dN_dr[2] * u3_1 + dN_dr[3] * u4_1
        l12 = dN_ds[0] * u1_1 + dN_ds[1] * u2_1 + dN_ds[2] * u3_1 + dN_ds[3] * u4_1
        l21 = dN_dr[0] * u1_2 + dN_dr[1] * u2_2 + dN_dr[2] * u3_2 + dN_dr[3] * u4_2
        l22 = dN_ds[0] * u1_2 + dN_ds[1] * u2_2 + dN_ds[2] * u3_2 + dN_ds[3] * u4_2
        
        # Initialize the B_LI matrix with zeros, it will have shape (3, 8)
        B_LI = np.zeros((3, 8))
    
        # Fill in the B_LI matrix according to the pattern above
        B_LI[0, 0] = l11 * dN_dr[0]
        B_LI[0, 1] = l21 * dN_dr[0]
        B_LI[0, 2] = l11 * dN_dr[1]
        B_LI[0, 3] = l21 * dN_dr[1]
        B_LI[0, 4] = l11 * dN_dr[2]
        B_LI[0, 5] = l21 * dN_dr[2]
        B_LI[0, 6] = l11 * dN_dr[3]
        B_LI[0, 7] = l21 * dN_dr[3]
    
        B_LI[1, 0] = l12 * dN_ds[0]
        B_LI[1, 1] = l22 * dN_ds[0]
        B_LI[1, 2] = l12 * dN_ds[1]
        B_LI[1, 3] = l22 * dN_ds[1]
        B_LI[1, 4] = l12 * dN_ds[2]
        B_LI[1, 5] = l22 * dN_ds[2]
        B_LI[1, 6] = l12 * dN_ds[3]
        B_LI[1, 7] = l22 * dN_ds[3]
    
        B_LI[2, 0] = l11 * dN_ds[0] + l12 * dN_dr[0]
        B_LI[2, 1] = l21 * dN_ds[0] + l22 * dN_dr[0]
        B_LI[2, 2] = l11 * dN_ds[1] + l12 * dN_dr[1]
        B_LI[2, 3] = l21 * dN_ds[1] + l22 * dN_dr[1]
        B_LI[2, 4] = l11 * dN_ds[2] + l12 * dN_dr[2]
        B_LI[2, 5] = l21 * dN_ds[2] + l22 * dN_dr[2]
        B_LI[2, 6] = l11 * dN_ds[3] + l12 * dN_dr[3]
        B_LI[2, 7] = l21 * dN_ds[3] + l22 * dN_dr[3]
        
        Blinear = B_L0+ B_LI
        #Blinear = ...

        return Blinear, determinant, derivatives

    def getNonlinearDeformationMatrix(self, displacements, derivatives):
        """
        Evaluates the degree of freedom numbering

        Parameters
        ----------
        displacements: ndarray
            Array containing the displacements of each node
        derivatives: ndarray
            Array with the shape function derivatives
        
        Returns
        ----------
        Bnonlinear: ndarray
            Array with the nonlinear portion of the strain-displacement matrix
        Egreenlagrance : ndarray
            The strain vector
        """

        # TO DO (1.3.1): Get the nonlinear deformation matrix
        dN_dr = derivatives[0]  # dNi/dr
        dN_ds = derivatives[1]  # dNi/ds
        # Initialize the B_NL matrix with zeros, it will have shape (4, 8)
        B_NL = np.zeros((4, 8))
    
        # Fill in the B_NL matrix according to the pattern above
        B_NL[0, 0] = dN_dr[0]
        B_NL[0, 2] = dN_dr[1]
        B_NL[0, 4] = dN_dr[2]
        B_NL[0, 6] = dN_dr[3]
    
        B_NL[1, 0] = dN_ds[0]
        B_NL[1, 2] = dN_ds[1]
        B_NL[1, 4] = dN_ds[2]
        B_NL[1, 6] = dN_ds[3]
    
        B_NL[2, 1] = dN_dr[0]
        B_NL[2, 3] = dN_dr[1]
        B_NL[2, 5] = dN_dr[2]
        B_NL[2, 7] = dN_dr[3]
    
        B_NL[3, 1] = dN_ds[0]
        B_NL[3, 3] = dN_ds[1]
        B_NL[3, 5] = dN_ds[2]
        B_NL[3, 7] = dN_ds[3]

        Bnonlinear = B_NL
        #Bnonlinear = ...

        e = Bnonlinear.dot(displacements)
        gradient = np.array([[e[0], e[1]], [e[2], e[3]]])
        epsilon = 1 / 2 * (gradient + gradient.T + gradient.T.dot(gradient))

        # TO DO (1.3.2): Calculate the strain vector
        Egreenlagrance = np.array([epsilon[0, 0], epsilon[1, 1], 2 * epsilon[0, 1]])
        #Egreenlagrance = ...

        return Bnonlinear, Egreenlagrance

    def getShapeFunctionDerivatives(self, r, s):
        """
        Evaluates the shape function derivatives

        Parameters
        ----------
        s , t: ndarray
            Gauss points and weights for the integration
        
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

        x = coordinates[:, 0] + scale * displacements[:, 0]
        y = coordinates[:, 1] + scale * displacements[:, 1]

        plt.plot(x, y, "-o", alpha=alpha)
