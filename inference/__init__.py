from .config import InferenceRunConfig
from .fdm import compute_layer_fdm_coefficient, compute_layer_fdm_delta

__all__ = [
    "InferenceRunConfig",
    "compute_layer_fdm_coefficient",
    "compute_layer_fdm_delta",
    "rollout_multilayer_fdm",
]


def __getattr__(name):
    if name == "rollout_multilayer_fdm":
        from .multilayer import rollout_multilayer_fdm

        return rollout_multilayer_fdm
    raise AttributeError(name)
