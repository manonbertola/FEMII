import os
from re import M
import sys
import abc

import numpy as np
import functools as ft
from copy import deepcopy

__author__ = "Konstantinos Tatsis"
__email__ = "konnos.tatsis@gmail.com"


class Dynamic:
    __constraints = {
        "Constant": "_getConstantLoad",
        "IncreasingLoad": "_getLinearlyIncreasingLoad",
        "Harmonic": "_getHarmonicLoad",
    }

    def __init__(self):
        pass

    def setConstraint(self, constraint):

        """
        Sepcify the type of solution constraint.

        Parameters
        ----------
        constraint: str or method
                The type of constraint to be used:
                                        'Load':	 load control
                        'Displacement':	 displacement control
                                         'Arc':	 arc-length constraint
                                        'Riks':  Riks constraint

        Raises
        ------
        ValueError
                If an invalid type of constraint is specified.
        """

        if isinstance(constraint, str):

            if constraint not in self.__constraints.keys():
                raise ValueError("Invalid type of constraint")

        elif callable(constraint):

            pass

        self.__constraint = getattr(self, self.__constraints[constraint])

    def setTolerance(self, tolerance):

        """
        Specify the convergence tolerance.

        Parameters
        ----------
        tolerance: float
                The solution tolerance.

        Raises
        ------
        ValueError
                If a non-positive value is specified for the tolerance.
        """

        if tolerance <= 0:
            raise ValueError("Invalid tolerance value")

        self.__tolerance = tolerance

    def setControlElement(self, element):

        """
        Specify the degree of freedom to store the stress-strain time history.
        Parameters
        ----------
        tolerance: float
                The solution tolerance.

        Raises
        ------
        ValueError
                If a non-positive value is specified for the tolerance.
        """

        if element < 0:
            raise ValueError("Invalid control element")

        self.__controlelement = element

    def setSolutionAttempts(self, attempts):

        """
        Specify the maximum number of solution attempts within each step.

        Parameters
        ----------
        attempts: int
                The maximum number of solution attempts at each step.

        Raises
        ------
        ValueError
                If a non-positive number solution attempts is specified.
        """

        if attempts < 0:
            raise ValueError("Invalid number of maximum solution attempts")

        self.__attempts = attempts

    def setMaxIterations(self, iterations):

        """
        Specify the maximum number of iterations within each solution step.

        Parameters
        ----------
        iterations: int
                The maximum number of iterations.

        Raises
        ------
        ValueError
                If a negative number of iterations is specified.
        """

        if iterations < 0:
            raise ValueError("Invalid number of maximum iterations")

        self.__iterations = iterations

    def _setNewmarkParameters(self, dt, gamma=1 / 2, beta=1 / 4):

        self.beta, self.gamma, self.dt = beta, gamma, dt

    def __getExcitation(self, t, Amp=None):

        if Amp is None:
            fext = self.__constraint(t)
        else:
            fext = self.__constraint(t, Amp)

        return fext

    def _getConstantLoad(self, t, Amp=2):
        """
        Define the excitation following a constant load. 

        Parameters
        ----------
        t: float
            The timestamp
        Amp: float
            The amplitude of the excitation

        Returns
        -------
        fext: float
            The excitation value for the given timestamp
        """
                
        fext = Amp

        return fext

    def _getLinearlyIncreasingLoad(self, t, Amp=0.05):
        """
        Define the excitation following a linearly increasing load. 

        Parameters
        ----------
        t: float
            The timestamp
        Amp: float
            The amplitude of the excitation

        Returns
        -------
        fext: float
            The excitation value for the given timestamp
        """

        fext = Amp * t

        return fext

    def _getHarmonicLoad(self, t, Params=[[0.5, 0.07484],[1.4,0.30426]]): #Params:[amplitude,frequency)]
        """
        Define the excitation following a harmonic load case 

        Parameters
        ----------
        t: float
            The timestamp
        Params: ndarray
            The parameters for synthesizing the excitation, given as Params=[[Amplitude, frequency],[Amplitude, frequency]]

        Returns
        -------
        fext: float
            The excitation value for the given timestamp
        """

        # Task 4.1: 
        # Apply a harmonic excitation with a certain frequency
        # content and an amplitude factor 
        
        #Number of frequency components
        no_components = np.shape(Params)[0]
        #Frequency, Amplitude
        fext = 0
        for i in range(no_components):
            fi = Params[i,1]
            amplitude = Params[i,0]
            
            fext = fext+amplitude*np.sin(fi*2*np.pi*t)


        return fext

    def solve(self, model, t, dt, Amp=None):

        """
        Solve the system for a prescribed number of increments.

        Parameters
        ----------
        model: Model
            The finite element model instance.
        t: ndarray
            The sequence of timesteps
        dt: float
            The time increment
        """

        model.make()

        ndofs = model.equation.K.shape[0]

        u_history = np.zeros((ndofs, t.shape[0]))
        v_history = np.zeros((ndofs, t.shape[0]))
        a_history = np.zeros((ndofs, t.shape[0]))
        stress_strain_history = np.zeros((2, t.shape[0]))

        u0, v0, a0 = model.getState()
        Stiffness_K, Mass_M, ElasticForces = model.getSystemMatrices(u0)
        f0 = model.equation.f
        fext = self.__getExcitation(t[0], Amp)
        f0[model.natural_BCs[:, 0]] = model.natural_BCs[:, 1] * fext

        model.equation.C = model.damping.getDampingMatrix(Mass_M,Stiffness_K)
        model.equation.a = np.linalg.inv(Mass_M).dot(
            (f0 - Stiffness_K.dot(u0) - model.equation.C.dot(v0))
        )
        self._setNewmarkParameters(dt)

        for step in range(len(t) - 1):

            attempt = 1
            convergence = False

            # displacement and load at the beginning of the step
            u0, v0, a0 = model.getState()
            fext = self.__getExcitation(t[step + 1], Amp)

            message = "Step {}\n"
            sys.stdout.write(message.format(step + 1))

            while (not convergence) and (attempt <= self.__attempts):

                message = "  Attempt {}\n"
                sys.stdout.write(message.format(attempt))

                u, v, a, convergence = self._getSolution(model, u0, v0, a0, fext)

                attempt += 1

            if not convergence:
                message = "    Failed to reach convergence after {} attempts\n"
                sys.stdout.write(message.format(step, attempt))

            u_history[:, step + 1] = u
            v_history[:, step + 1] = v
            a_history[:, step + 1] = a
            stress, strain, kappa, tangent, yielded = model.history.retrieve(
                self.__controlelement
            )
            stress_strain_history[0, step + 1] = stress[0]
            stress_strain_history[1, step + 1] = strain[0]
            model.equation.u, model.equation.v, model.equation.a = u, v, a
            model.history.displacements = u

        return u_history, stress_strain_history

    def _getSolution(self, model, u0, v0, a0, f0):

        """
        Get the solution for an incremental step.

        Parameters
        ----------
        model: Model
                The model instance.
        u0,v0,a0: ndarray
                The solution at the beginning of the step for displacements, velocities & accelerations
        f0: float
                The excitation at the beggining of the step
        
        Returns
        -------
        uk,vk,ak: ndarray
                The solution at the end of the step for displacements, velocities & accelerations 
        """

        # Task 1.1: Initial acceleration, velocity and displacement predictors
        ak = np.zeros_like(a0)
        vk = v0 + a0*(1-self.gamma)*self.dt + ak *self.gamma*self.dt
        uk = u0 + v0 * self.dt + a0*(0.5 - self.beta)* self.dt**2 + ak*self.beta*self.dt**2

        #Evaluate external forcing
        fext = model.equation.f
        fext[model.natural_BCs[:, 0]] = model.natural_BCs[:, 1] * f0

        # Evaluate internal force vector and tangent stiffness matrix
        # based on current displacement state
        Stiffness_K, Mass_M, ElasticForces = model.getSystemMatrices(uk)

        # Task 1.2: Evaluate initial residual (for the step)
        residual = fext - Mass_M.dot(ak) - ElasticForces - model.equation.C.dot(vk)

        # Task 1.3: Store the residual norm
        residual_norm = np.linalg.norm(residual)


        #Loop until convergence or maximum number of iterations is reached
        for iteration in range(self.__iterations):

            # Task 1.4: Assemble Jacobian matrix
            # Hint: Damping matrix -> model.equation.C
            #       Newmark parameters: self.gamma, self.dt, etc..
            Keff = (Mass_M + model.equation.C*self.gamma*self.dt + Stiffness_K*self.beta*self.dt**2
                
            )

            # Task 1.5: Evaluate acceleration, velocity and displacement increment
            da = np.linalg.inv(Keff).dot(residual)
            dv = da * self.gamma * self.dt
            du = da * self.beta * self.dt**2

            # Update accelerations, velocities and displacements
            ak, vk, uk = ak + da, vk + dv, uk + du

            # Task 1.6: Evaluate energy value
            if iteration == 0:
                energy_criterion = np.abs(du.T.dot(residual))

            # Evaluate internal force vector and tangent stiffness matrix
            # based on updated displacement state
            Stiffness_K, Mass_M, ElasticForces = model.getSystemMatrices(uk)

            # Task 1.7: Evaluate residual (force)
            residual = fext - Mass_M.dot(ak) - ElasticForces - model.equation.C.dot(vk)

            # Task 1.8: Evaluate energy residual
            residual_energy = np.abs(du.T.dot(residual)) #work done = residual force * displacement =fext-fint

            # Task 1.9: Check convergence based on either measures
            if (
                np.linalg.norm(residual) <= self.__tolerance*np.linalg.norm(residual_norm) #residual norm is the very first residual, works as scaling param
                or residual_energy <= self.__tolerance * energy_criterion
            ):
                model.history = deepcopy(model.historyTemp)
                message = "    Solution converged after {} iterations\n"
                sys.stdout.write(message.format(iteration + 1))

                message = "    Residual norm {:.3e}\n"
                sys.stdout.write(message.format(np.linalg.norm(residual) / residual_norm))

                convergence = True
                break

        else:
            model.history = deepcopy(model.historyTemp)
            message = "    Failed to reach convergence after {} iterations\n"
            sys.stdout.write(message.format(iteration + 1))

            message = "    Residual norm {:.3e}\n"
            sys.stdout.write(message.format(np.linalg.norm(residual) / residual_norm))

            convergence = False

        return uk, vk, ak, convergence
