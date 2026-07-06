from .config import InferenceRunConfig, SingleLayerInferenceRunConfig
from .fdm import (
    compute_layer_fdm_coefficient,
    compute_layer_fdm_delta,
    compute_layer_implicit_fdm_step,
)

__all__ = [
    "InferenceRunConfig",
    "SingleLayerInferenceRunConfig",
    "compute_layer_fdm_coefficient",
    "compute_layer_fdm_delta",
    "compute_layer_implicit_fdm_step",
    "render_multilayer_clouds_from_hdf5",
    "render_single_layer_surfaces_from_hdf5",
    "rollout_multilayer_fdm",
    "rollout_single_layer_static",
    "run_single_layer_inference_from_config",
]


def __getattr__(name):
    if name == "rollout_multilayer_fdm":
        from .multilayer import rollout_multilayer_fdm

        return rollout_multilayer_fdm
    if name == "render_multilayer_clouds_from_hdf5":
        from .io import render_multilayer_clouds_from_hdf5

        return render_multilayer_clouds_from_hdf5
    if name == "render_single_layer_surfaces_from_hdf5":
        from .single_layer import render_single_layer_surfaces_from_hdf5

        return render_single_layer_surfaces_from_hdf5
    if name == "rollout_single_layer_static":
        from .single_layer import rollout_single_layer_static

        return rollout_single_layer_static
    if name == "run_single_layer_inference_from_config":
        from .single_layer import run_single_layer_inference_from_config

        return run_single_layer_inference_from_config
    raise AttributeError(name)
