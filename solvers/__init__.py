import gin

from solvers import solver_sokoban
from solvers import iterative_solver_sokoban
from solvers import bfs_solver_int
from solvers import iterative_solver_int
from solvers import bfs_solver_rubik
from solvers import iterative_solver_rubik


def configure_solver(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='solvers'
    )


BestFSSolverSokoban = configure_solver(solver_sokoban.BestFSSolverSokoban)
BestFSIterativeSolverSokoban = configure_solver(iterative_solver_sokoban.BestFSIterativeSolverSokoban)
BestFSSolverINT = configure_solver(bfs_solver_int.BestFSSolverINT)
BestFSIterativeSolverINT = configure_solver(iterative_solver_int.BestFSIterativeSolverINT)
BestFSSolverRubik = configure_solver(bfs_solver_rubik.BestFSSolverRubik)
BestFSIterativeSolverRubik = configure_solver(iterative_solver_rubik.BestFSIterativeSolverRubik)
