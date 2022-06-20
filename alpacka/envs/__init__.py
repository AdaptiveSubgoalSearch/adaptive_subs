# pylint: disable=invalid-name
"""Environments."""

from alpacka.envs import atari
from alpacka.envs import bin_packing
from alpacka.envs import cartpole
from alpacka.envs import gfootball
from alpacka.envs import octomaze
# from alpacka.envs import pycolab
from alpacka.envs import rubik
from alpacka.envs import sokoban
from alpacka.envs import wrappers
from alpacka.envs.base import *


# Configure envs in this module to ensure they're accessible via the
# alpacka.envs.* namespace.
def configure_env(env_class):
    return gin.external_configurable(
        env_class, module='alpacka.envs'
    )


# Envs not requiring any extra dependencies.
native_envs = []


# Core envs.
BinPacking = configure_env(bin_packing.BinPacking)
CartPole = configure_env(cartpole.CartPole)
Octomaze = configure_env(octomaze.Octomaze)
RubiksCube = configure_env(rubik.RubiksCube)

native_envs.extend([BinPacking, CartPole, Octomaze, RubiksCube])


# Atari.
Atari = configure_env(atari.Atari)


# Google Football.
GoogleFootball = configure_env(gfootball.GoogleFootball)


# Gridworld envs based on PyColab.
# ChainWalk = configure_env(pycolab.ChainWalk)
# CliffWalk = configure_env(pycolab.CliffWalk)
# FourRooms = configure_env(pycolab.FourRooms)
# OctoGrid = configure_env(pycolab.OctoGrid)
# MudWalk = configure_env(pycolab.MudWalk)
# TwoCorridors = configure_env(pycolab.TwoCorridors)

# native_envs.extend([ChainWalk, CliffWalk, FourRooms, OctoGrid, MudWalk,
#                     TwoCorridors])


# Sokoban envs.
ActionNoiseSokoban = configure_env(sokoban.ActionNoiseSokoban)
Sokoban = configure_env(sokoban.Sokoban)


# Wrappers.
FrameStackWrapper = configure_env(wrappers.FrameStackWrapper)
TimeLimitWrapper = configure_env(wrappers.TimeLimitWrapper)
StateCachingWrapper = configure_env(wrappers.StateCachingWrapper)
wrap = configure_env(wrappers.wrap)
