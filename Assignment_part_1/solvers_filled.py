import os
import sys
import abc
#import tqdm
from tqdm.notebook import tqdm

import numpy as np
import functools as ft


__author__ = "Konstantinos Tatsis"
__email__ = "konnos.tatsis@gmail.com"


class Constraint:

    """
    Description

    Parameters
    ----------
    constraint: str
            The type of constraint to be used
                    'Arc'          : Arc-length method
                    'Displacement' : Displacement control
                    'Load'         : Load control
                    'Riks'         : Riks method

    Attributes
    ----------
    name: str
            The name of the constraint function.

    Methods
    -------
    get(x, c, x0, c0, dx, dc, S, length, *args)
            Get the value of the constraint function and its derivatives.
    predict(func, x, c, A, b, r)
            Predict the solution at the beginning of the step.

    Raises
    ------
    KeyError
            If an invalid type of constraint is specified.
    """

    __constraints = {
        "Displacement": "_Displacement",
        "Load": "_Load",
        "Arc": "_Arc",
        "Riks": "_Riks",
    }

    def __init__(self, constraint):

        module = sys.modules[__name__]

        try:
            constraint = getattr(module, self.__constraints[constraint])
        except KeyError:
            raise KeyError("Invalid type of constraint")

        self._constraint = constraint()

    def get(
        self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None, *args
    ):
        """
        Get the values of the constraint function and the gradients with
        respect to the independent variable and the scale factor.

        Parameters
        ----------
        u: ndarray
                The current value of the independent variable.
        llambdas: ndarray
                The current value of the scale factor.
        u0: ndarray
                The value of the independent variable at the beginning of the step.
        lambda0: ndarray
                The value of the scale factor at the beginning of the step.
        deltaUp: ndarray
                The increment of the independent variable.
        deltalambdap: ndarray
                The increment of the scale factor.
        T: ndarray
                The selection matrix of the independent variable.
        deltaS: float
                The increment of the arc length.

        Returns
        -------
        g: float
                The value of the constraint function.
        h: ndarray
                The derivative of the constraint function with respect to the
                independent variable.
        s: float
                The derivative of the constraint function with respect to the
                scaling factor.
        """

        g, h, s = self._constraint.get(
            u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None, *args
        )

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        """
        Get the prediction at the beginning of each solution step.

        Parameters
        ----------
        func: callable
                ...
        u: ndarray
                The initial value of the solution variable.
        llambda: float
                The initial value of the scale factor.
        deltaS: float
                The increment of the arc length.
        StiffnessK: ndarray
                The tangent coefficient matrix.
        fext: ndarray
                The constant vector.
        Residualsr: ndarray
                The residual vector.

        Returns
        -------
        u: ndarray
                The solution variable at the prediction point.
        llambda: float
                The scale factor at the prediction point.
        deltaUp: ndarray
                The predicted solution increment.
        deltalambdap: float
                The predicted scale factor increment.
        StiffnessK: ndarray
                The tangent coefficient matrix at the prediciton point.
        fext: ndarray
                The constant vector at the prediction point.
        Residualsr: ndarray
                The residual vector at the prediction point.
        """

        (
            u,
            llambda,
            deltaUp,
            deltalambdap,
            StiffnessK,
            fext,
            Residualsr,
        ) = self._constraint.predict(
            func, u, llambda, deltaS, StiffnessK, fext, Residualsr
        )

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr

    @property
    def name(self):
        return self._constraint.name


class _Constraint(abc.ABC):
    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class _Load(_Constraint):

    name = "Load control"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        # g = ...
        # h = ...
        # s = ...

        g = llambda - (llambda0 + deltaS)
        h = np.zeros_like(u)
        s = 1

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp, deltalambdap = np.zeros_like(u), 0

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Displacement(_Constraint):

    name = "Displacement control"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        if T is None:
        #     T = ...

            T = np.zeros_like(u)
            T[167] = 1 #top middle node at 0, 101.5 is 84, *2 for dofs, -1 bc it starts at 0

        # g = ...
        # h = ...
        # s = ...
        
        g = np.dot(T, u) - (np.dot(T, u0) + deltaS)
        h = T
        s = 0

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp, deltalambdap = np.zeros_like(u), 0

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Riks(_Constraint):

    name = "Riks"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        # g = ...
        # h = ...
        # s = ...

        u1 = u0 + deltalambdap * deltaUp
        llambda1 = llambda0 + deltalambdap
        
        g = np.dot(deltaUp.T , (u - u1)) + deltalambdap * (llambda - llambda1)
        h = deltaUp
        s = deltalambdap

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        # deltaUp = ...
        # deltalambdap = ...

        deltaUp = np.linalg.solve(StiffnessK, fext)
        deltalambdap = deltaS/np.linalg.norm(deltaUp)

        # if ... < 0:
        #     deltalambdap ...

        if (np.dot(fext.T , deltaUp)/np.dot(deltaUp.T , deltaUp)) < 0: #here comes the kappa
            deltalambdap = - deltalambdap

        # u, llambda = ...

        u, llambda = u + deltalambdap * deltaUp, llambda + deltalambdap
        StiffnessK, fext, Residualsr = func(u, llambda)

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Arc(_Constraint):

    name = "Arc-length"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        g = np.sqrt(np.dot((u - u0).T, (u - u0)) + (llambda - llambda0)**2) - deltaS
        h = (u - u0)/g
        s = (llambda - llambda0)/g

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp = np.linalg.solve(StiffnessK, fext)
        deltalambdap = deltaS/np.linalg.norm(deltaUp)

        if (np.dot(fext.T , deltaUp)/np.dot(deltaUp.T , deltaUp)) < 0:
            deltalambdap * (-1)

        u, llambda = u + deltalambdap * deltaUp, llambda + deltalambdap
        StiffnessK, fext, Residualsr = func(u, llambda)

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class Static:

    """
    Description

    Parameters
    ----------
    constraint: str
            The type of constraint ('Arc', 'Displacement', 'Load', 'Riks')
    tol: float
            The relative convergence tolerance.
    maxit: int
            The maximum number of iterations within each solution step.

    Methods
    -------
    setConstraint()
            Specify the type of solution constraint.
    setTolerance()
            Specify the relative convergence tolerance.
    setMaxIterations()
            Sepcify the maximum number of iterations to be allowed with each step.
    """

    def __init__(self, constraint="Load", tolerance=1e-3, maxit=20):

        self.setConstraint(constraint)
        self.setTolerance(tolerance)
        self.setMaxIterations(maxit)

    def setConstraint(self, constraint):

        """
        Specify the type of constraint.

        Parameters
        ----------
        constraint: str
                The type of constraint to be used.

        Raises
        ------
        ValueError
                If an invalid type of constraint is specified.
        """

        self.constraint = Constraint(constraint)

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

    def solve(self, model, increments_length):

        """
        Solve the system for a prescribed number of increments.

        Parameters
        ----------
        model: Model
                The finite element model instance.
        increments_length: ndarray
                The sequence of increments.
        """

        model.make()
        lambda_history = np.zeros(increments_length.shape[0] + 1)
        u_history = np.zeros(increments_length.shape[0] + 1)

        #for step in tqdm.tqdm_notebook(range(len(increments_length))):
        for step in tqdm(range(len(increments_length))):

            attempt = 1
            convergence_boolean = False

            # displacement and load at the beginning of the step
            u0, lambda0 = model.getState()

            message = "Step {}\n"
            sys.stdout.write(message.format(step + 1))

            while (not convergence_boolean) and (attempt <= self.__attempts):

                message = "  Attempt {}\n"
                sys.stdout.write(message.format(attempt))

                u, llambda, convergence_boolean = self._getSolution(
                    model, u0, lambda0, increments_length[step]
                )

                attempt += 1

                if (not convergence_boolean) and (attempt <= self.__attempts):
                    template = "    Reducing increment from {:.3e} to {:.3e}\n"
                    message = template.format(
                        increments_length[step], 0.5 * increments_length[step]
                    )
                    sys.stdout.write(message)
                    increments_length[step] *= 0.5

            if not convergence_boolean:
                message = "    Failed to reach convergence after {} attempts\n"
                sys.stdout.write(message.format(step, attempt))

            u_history[step + 1] = np.linalg.norm(u)
            lambda_history[step + 1] = llambda

        return u_history, lambda_history

    def _getSolution(self, model, u0, lambda0, increment):

        """
        Perform an incremental step and get the solution variables using
        Newton-Raphson iterations.

        Parameters
        ----------
        model: Model
                The model instance.
        u0: ndarray
                The initial value of the independent variable.
        lambda0: float
                The initial value of the scale factor.

        Returns
        -------
        u: ndarray
                The independent variable.
        llambda: float
                The scale factor.
        convergence_boolean: bool
                The convergence flag.
        """

        Stiffness_K, fext, ResidualsR = model.getSystemMatrices(u0, lambda0)
        # convergence_norm = ...
        convergence_norm = np.linalg.norm(fext)

        # Calculate the system prediction
        (
            u,
            llambda,
            deltaUp,
            deltalambdap,
            Stiffness_K,
            fext,
            ResidualsR,
        ) = self.constraint.predict(
            model.getSystemMatrices,
            u0,
            lambda0,
            increment,
            Stiffness_K,
            fext,
            ResidualsR,
        )

        for iteration in range(self.__iterations):

            # Evaluate the constraint function
            g, h, s = self.constraint.get(
                u, llambda, u0, lambda0, deltaUp, deltalambdap, increment
            )

            # Calculate the solution contributions
        #     du_tilde = ...
        #     du_double_tilde = ...

            du_tilde = np.linalg.inv(Stiffness_K) @ fext
            du_double_tilde = - np.linalg.inv(Stiffness_K) @ ResidualsR

            # Calculate the solution increments
        #     deltalambdap = ...
        #     deltaUp = ...
        
            deltalambdap = - ((g + h.T @ du_double_tilde)/(s + h.T @ du_tilde))
            deltaUp = deltalambdap * du_tilde + du_double_tilde

            # Update the solution variables
            u, llambda = u + deltaUp, llambda + deltalambdap

            # Evaluate the residual
            Stiffness_K, fext, ResidualsR = model.getSystemMatrices(u, llambda)

            # Check convergence criteria
        #     if ... <= self.__tolerance * ...:

            if np.linalg.norm(ResidualsR) <= self.__tolerance * convergence_norm:
                message = "    Solution converged after {} iterations\n"
                sys.stdout.write(message.format(iteration + 1))

                message = "    Residual norm {:.3e}\n"
                sys.stdout.write(message.format(np.linalg.norm(ResidualsR)))

                convergence_boolean = True
                break

        else:

            message = "    Failed to reach convergence after {} iterations\n"
            sys.stdout.write(message.format(iteration + 1))

            message = "    Residual norm {:.3e}\n"
            sys.stdout.write(message.format(np.linalg.norm(ResidualsR)))

            convergence_boolean = False

        return u, llambda, convergence_boolean
