import numpy as np
import matplotlib.pyplot as plt


class Element:
    """
    Element class

    Parameters
    ----------
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
    getMass(self, coordinates):
        Get the system mass matrix
    setHistory(self, stress, kappa, tangentC, yielded):
        Set the material history parameters and variables
    getStiffnessAndResidual(self, coordinates, displacements):
        Get the (tangent) stiffness matrix and the internal (residual) forces
    getDegreesOfFreedom(self, dofs, nodes):
        Evaluate the degree of freedom numbering
    plot(self, coordinates, displacements, scale, alpha):
        Visualize the element in its (un)deformed state
    """

    def __init__(self, material, section):
        self.material = material
        self.section = section

        self.history = {}
        self.setHistory(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))

    def setHistory(self, stress, strain, kappa, tangent, yielded):
        """
        Set the material history parameters and variables by updating the instance

        Parameters
        ----------
        stress: ndarray
            The stress vector
        strain: ndarray
            The strain vector
        kappa: ndarray
            The hardening parameter
        tangentC: ndarray
            The tangent constitutive matrix
        yielded: Boolean
            Variable indicating if yielding has occured
        """
        self.history["stress"] = stress
        self.history["strain"] = strain
        self.history["kappa"] = kappa
        self.history["tangent"] = tangent
        self.history["yielded"] = yielded

    def getMass(self, coordinates):
        """
        Get the mass matrix of the system

        Parameters
        ----------
        coordinates: ndarray
            Array of nodal coordinates

        Returns
        ----------
        M: ndarray
            The mass matrix of the system
        """
        try:
            rho = self.material.rho
        except AttributeError:
            raise AttributeError("undefined material density")

        try:
            A = self.section.A
        except AttributeError:
            raise AttributeError("undefined cross-section properties")

        dX = coordinates[1, :] - coordinates[0, :]
        L = np.linalg.norm(dX)
        theta = np.arctan2(dX[1], dX[0])

        T = np.array(
            [
                [np.cos(theta), np.sin(theta), 0, 0],
                [-np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, np.cos(theta), np.sin(theta)],
                [0, 0, -np.sin(theta), np.cos(theta)],
            ]
        )

        Ml = np.array(
            [
                [1, 0, 0.5, 0],
                [0, 1, 0, 0.5],
                [0.5, 0, 1, 0],
                [0, 0.5, 0, 1],
            ]
        )

        Ml = rho * A * L / 3 * Ml

        M = T.T.dot(Ml).dot(T)

        return M

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
            E = self.material.E
        except AttributeError:
            raise AttributeError("undefined material properties")

        try:
            A = self.section.A
        except AttributeError:
            raise AttributeError("undefined cross-section properties")

        dX = coordinates[1, :] - coordinates[0, :]
        L = np.linalg.norm(dX)
        theta = np.arctan2(dX[1], dX[0])

        T = np.array(
            [
                [np.cos(theta), np.sin(theta), 0, 0],
                [-np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, np.cos(theta), np.sin(theta)],
                [0, 0, -np.sin(theta), np.cos(theta)],
            ]
        )

        det = L / 2
        du = T.dot(displacements.flatten())

        weights = np.array([1, 1])
        points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])

        B = np.array([[-1, 0, 1, 0]])
        K = np.zeros((4, 4))
        R = np.zeros(4)

        for j, (point, weight) in enumerate(zip(points, weights)):

            # Incremental strain
            depsilon = B.dot(du)

            # Retrieve element history
            sigma, kappa = self.history["stress"][j], self.history["kappa"][j]

            # Call the return mapping
            sigma, kappa, tangent, yielded = self.material.getStressAndTangent(
                sigma, kappa, depsilon
            )

            # Update tangent stiffness and residual
            K = K + weight * B.T.dot(tangent).dot(B) * A * det
            R = R + weight * B.T.dot(sigma) * A * det

            # Update history
            self.history["stress"][j] = sigma
            self.history["strain"][j] = self.history["strain"][j] + depsilon
            self.history["kappa"][j] = kappa
            self.history["tangent"][j] = tangent
            self.history["yielded"][j] = yielded

        K = T.T.dot(K).dot(T)
        R = T.T.dot(R)

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
