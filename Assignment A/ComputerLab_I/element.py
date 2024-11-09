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


    def getStiffnessAndResidual(self, coordinates, displacements):
        """
        Gets the (tangent) stiffness matrix and the internal (residual) forces

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

        dX = coordinates[1, :] - coordinates[0, :]
        dx = dX + displacements[1, :] - displacements[0, :]

        try: 
            E = self.material.E
        except AttributeError:
            raise AttributeError('undefined material properties')

        try:
            A = self.section.A
        except AttributeError:
            raise AttributeError('undefined cross-section properties')

        l = np.linalg.norm(dx)
        L = np.linalg.norm(dX)
        theta = np.arctan2(dx[1], dx[0])

        Kl = E*A*l**2/L**3*np.array([
            [np.cos(theta)**2, np.cos(theta)*np.sin(theta), -np.cos(theta)**2, -np.cos(theta)*np.sin(theta)],
            [np.cos(theta)*np.sin(theta), np.sin(theta)**2, -np.cos(theta)*np.sin(theta), -np.sin(theta)**2],
            [-np.cos(theta)**2, -np.cos(theta)*np.sin(theta), np.cos(theta)**2, np.cos(theta)*np.sin(theta)],
            [-np.cos(theta)*np.sin(theta), -np.sin(theta)**2, np.cos(theta)*np.sin(theta), np.sin(theta)**2]
            ])

        P = E*A*(l**2-L**2)/(2*L**2)
        Knl = P/L*np.array([
            [ 1,  0, -1,  0], 
            [ 0,  1,  0, -1], 
            [-1,  0,  1,  0], 
            [ 0, -1,  0,  1]])
        K = Kl+Knl

        R = P*np.array([-np.cos(theta), -np.sin(theta), np.cos(theta), np.sin(theta)])

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

        x = coordinates[:, 0]+scale*displacements[:, 0]
        y = coordinates[:, 1]+scale*displacements[:, 1]

        plt.plot(x, y, '-o', alpha=alpha)