from .config import InferenceRunConfig
from .fdm import compute_layer_fdm_coefficient, compute_layer_fdm_delta

__all__ = [
    "InferenceRunConfig",
    "compute_layer_fdm_coefficient",
    "compute_layer_fdm_delta",
    "render_multilayer_clouds_from_hdf5",
    "rollout_multilayer_fdm",
]


def __getattr__(name):
    if name == "rollout_multilayer_fdm":
        from .multilayer import rollout_multilayer_fdm

        return rollout_multilayer_fdm
    if name == "render_multilayer_clouds_from_hdf5":
        from .io import render_multilayer_clouds_from_hdf5

        return render_multilayer_clouds_from_hdf5
    raise AttributeError(name)
