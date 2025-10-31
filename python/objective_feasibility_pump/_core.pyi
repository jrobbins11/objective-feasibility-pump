import collections.abc
import numpy
import numpy.typing
import scipy.sparse
import typing

class OFP_Info:
    def __init__(self) -> None:
        """__init__(self: objective_feasibility_pump._core.OFP_Info) -> None"""
    @property
    def alpha(self) -> float:
        """(self: objective_feasibility_pump._core.OFP_Info) -> float"""
    @property
    def feasible(self) -> bool:
        """(self: objective_feasibility_pump._core.OFP_Info) -> bool"""
    @property
    def iter(self) -> int:
        """(self: objective_feasibility_pump._core.OFP_Info) -> int"""
    @property
    def objective(self) -> float:
        """(self: objective_feasibility_pump._core.OFP_Info) -> float"""
    @property
    def perturbations(self) -> int:
        """(self: objective_feasibility_pump._core.OFP_Info) -> int"""
    @property
    def restarts(self) -> int:
        """(self: objective_feasibility_pump._core.OFP_Info) -> int"""
    @property
    def runtime(self) -> float:
        """(self: objective_feasibility_pump._core.OFP_Info) -> float"""

class OFP_Settings:
    T: int
    alpha0: float
    buffer_size: int
    delta_alpha: float
    lp_threads: int
    max_iter: int
    max_restarts: int
    phi: float
    rng_seed: int
    t_max: float
    tol: float
    verbose: bool
    verbosity_interval: int
    def __init__(self) -> None:
        """__init__(self: objective_feasibility_pump._core.OFP_Settings) -> None"""

class OFP_Solver:
    def __init__(self, c: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], A: scipy.sparse.csc_matrix[numpy.float64], l_A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], u_A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], l_x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], u_x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], bins: collections.abc.Sequence[typing.SupportsInt], settings: OFP_Settings = ...) -> None:
        '''__init__(self: objective_feasibility_pump._core.OFP_Solver, c: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], A: scipy.sparse.csc_matrix[numpy.float64], l_A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], u_A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], l_x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], u_x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], bins: collections.abc.Sequence[typing.SupportsInt], settings: objective_feasibility_pump._core.OFP_Settings = <objective_feasibility_pump._core.OFP_Settings object at 0x777ea0acceb0>) -> None'''
    def get_info(self) -> OFP_Info:
        """get_info(self: objective_feasibility_pump._core.OFP_Solver) -> objective_feasibility_pump._core.OFP_Info"""
    def get_solution(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[m, 1]']:
        '''get_solution(self: objective_feasibility_pump._core.OFP_Solver) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]'''
    def solve(self) -> bool:
        """solve(self: objective_feasibility_pump._core.OFP_Solver) -> bool"""
