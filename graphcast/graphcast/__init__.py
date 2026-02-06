"""GraphCast package initialization.

This file ensures the local checkout behaves as a normal Python package so
imports like `from graphcast import autoregressive` work reliably.
"""

# Re-export key submodules for convenience.
from . import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
    xarray_jax,
    xarray_tree,
    predictor_base,
    deep_typed_graph_net,
    icosahedral_mesh,
)

__all__ = [
    "autoregressive",
    "casting",
    "checkpoint",
    "data_utils",
    "graphcast",
    "normalization",
    "rollout",
    "xarray_jax",
    "xarray_tree",
    "predictor_base",
    "deep_typed_graph_net",
    "icosahedral_mesh",
]
