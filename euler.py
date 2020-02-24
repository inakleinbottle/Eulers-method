
from math import floor, isclose
from numbers import Number
import warnings 

import numpy as np


def solve_ivp(func, y0, t_range, num_steps=None, step_size=None):
    """
    Solve a first order ODE initial value problem using Euler's method.

    Solves an ODE initial value problem of the form

        y' = f(t, y)

    where y is an unknown (possibly vector valued function) and f is a known
    (and possibly vector valued function).

    The number of steps or the step size can be customised by providing either
    the num_steps or step_size optional argument but not both. If neither are
    provided then a default of 100 steps will be used. The number of steps or
    step size will be computed if one of these values is provided.

    The function evaluates num_steps from the initial condition, so the
    arrays returned by this function will have num_steps+1 elements.

    Parameters
    ----------
        func - Callable defining the right hand side of the ODE above.
        y0 - Initial condition. Can be a single number or an object that
             can be converted to a Numpy array. The size should match the
             number of equations.
        t_range - Range on which to solve the ODE. Should be a tuple of
                  numbers, with the first element smaller than the second.
        num_steps (optional) - Number of steps to perform.
        step_size (optional) - Step size to use.

    Returns
    -------
        t - Array of t-values.
        y - Array of approximate values for y(t).
    """
    if not isinstance(t_range, (tuple, list, np.ndarray)):
        raise TypeError("T range must be provided as a tuple, list, or array")
    elif isinstance(t_range, (tuple, list)) and not len(t_range) == 2:
        raise ValueError("T range must consist of two values")
    elif isinstance(t_range, np.ndarray) and not t_range.shape == (2,):
        raise ValueError("T range must consist of two values")

    t0, tL = t_range

    if not isinstance(t0, Number) and isinstance(tL, Number):
        raise TypeError("T range must consist of numbers")

    if not tL > t0:
        raise ValueError("t_range[1] must be larger than t_range[0]")

    

    if step_size is None and num_steps is None:
        num_steps = 100
    elif step_size is not None and num_steps is not None:
        raise ValueError("Cannot specify both step_size and num_steps")
    
    if step_size is None:
        step_size = (tL - t0)/num_steps
    elif num_steps is None:
        num_steps = floor((tL - t0)/step_size)
        
        if not isclose(num_steps*step_size, tL - t0):
            # emit a warning if the step size doesn't evenly divide the
            # t range.
            warnings.warn("Step size does not divide the t_range evenly")

    if isinstance(y0, Number):
        y0 = np.array([y0])
    elif not isinstance(y0, np.ndarray):
        try:
            y0 = np.array(y0)
        except Exception:
            raise TypeError("Could not convert y0 to a Numpy array")
    else:
        np.ravel(y0)  # flatten the array to make sure it is 1 dimensional

    num_equations = y0.shape[0]

    # make sure func returns a Numpy array of the correct size
    f1 = func(t0, y0)
    if not isinstance(f1, np.ndarray):
        func = lambda t, y, f=func: np.array(f(t, y))
    
    if isinstance(f1, Number) and not num_equations == 1: 
        raise ValueError(
            "The function func returns a number but a vector"
            f" of length {num_equations} is expected"
        )
    elif isinstance(f1, (tuple, list)) and not len(f1) == num_equations:
        raise ValueError(
            f"The function func returns a {type(f1)} of length {len(f1)},"
            f" but a vector of length {num_equations} is expected"
        )

    t = np.linspace(t0, tL, num_steps+1, dtype=np.float64)
    y = np.empty((num_equations, num_steps+1), dtype=np.float64)

    y[:, 0] = y0

    for i, ti in enumerate(t[:-1]):
        yi = y[:, i]
        y[:, i+1] = yi + step_size*func(ti, yi)

    return t, y
