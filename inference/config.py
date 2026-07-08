from dataclasses import dataclass
from typing import Optional, Sequence


def _is_non_negative_integer(value):
    if isinstance(value, bool):
        return False
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return False
    return int_value >= 0 and int_value == value


@dataclass(frozen=True)
class InferenceRunConfig:
    num_layers: int
    layer_spacing: float
    output_path: str = "../runs/pdgcn/multilayer_prediction.h5"
    dataset_index: int = 0
    h5_path: Optional[str] = None
    h5_dir: Optional[str] = None
    output_dir: Optional[str] = None
    output_prefix: str = "pre_"
    prediction_group_path: str = "prediction/pdgcn_multilayer"
    batch_mode: bool = False
    steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    bottom_temperature_star: float = 0.0
    allow_unstable_fdm: bool = False
    layer_fiber_angles_deg: Optional[Sequence[float]] = None
    normal_offset_sign: int = -1
    return_dimensionless: bool = False
    write_vtk: bool = True
    use_pdgcn_inplane: bool = True
    pdgcn_inplane_top_layer_only: bool = False
    cloud_interval: int = 20
    layer_batch_size: Optional[int] = None
    delta_smoothing_alpha: float = 0.2
    delta_smoothing_steps: int = 1
    cloud_max_nodes_per_layer: Optional[int] = None
    vtk_output_dir: Optional[str] = None

    def __post_init__(self):
        if int(self.num_layers) < 2:
            raise ValueError(f"inference.num_layers must be at least 2, got {self.num_layers}.")
        if float(self.layer_spacing) <= 0:
            raise ValueError(f"inference.layer_spacing must be positive, got {self.layer_spacing}.")
        if int(self.dataset_index) < 0:
            raise ValueError(f"inference.dataset_index must be non-negative, got {self.dataset_index}.")
        if not isinstance(self.batch_mode, bool):
            raise ValueError("inference.batch_mode must be a boolean.")
        if self.h5_dir is not None and (not isinstance(self.h5_dir, str) or not self.h5_dir.strip()):
            raise ValueError("inference.h5_dir must be null or a non-empty string.")
        if self.output_dir is not None and (not isinstance(self.output_dir, str) or not self.output_dir.strip()):
            raise ValueError("inference.output_dir must be null or a non-empty string.")
        if not isinstance(self.output_prefix, str) or not self.output_prefix:
            raise ValueError("inference.output_prefix must be a non-empty string.")
        if not isinstance(self.prediction_group_path, str) or not self.prediction_group_path.strip():
            raise ValueError("inference.prediction_group_path must be a non-empty string.")
        _validate_hdf5_group_path(self.prediction_group_path, "inference.prediction_group_path")
        if self.steps is not None and int(self.steps) <= 0:
            raise ValueError(f"inference.steps must be positive when set, got {self.steps}.")
        if self.warmup_steps is not None and int(self.warmup_steps) < 0:
            raise ValueError(f"inference.warmup_steps must be non-negative when set, got {self.warmup_steps}.")
        if self.layer_fiber_angles_deg is not None:
            if len(self.layer_fiber_angles_deg) != int(self.num_layers):
                raise ValueError(
                    "inference.layer_fiber_angles_deg length must match num_layers, "
                    f"got {len(self.layer_fiber_angles_deg)} for num_layers={self.num_layers}."
                )
            if abs(float(self.layer_fiber_angles_deg[0])) > 1e-12:
                raise ValueError("inference.layer_fiber_angles_deg[0] must be 0.0 for the base layer.")
            for angle in self.layer_fiber_angles_deg:
                float(angle)
        if int(self.normal_offset_sign) not in (-1, 1):
            raise ValueError(f"inference.normal_offset_sign must be -1 or 1, got {self.normal_offset_sign}.")
        if int(self.cloud_interval) <= 0:
            raise ValueError(f"inference.cloud_interval must be positive, got {self.cloud_interval}.")
        if self.layer_batch_size is not None and int(self.layer_batch_size) <= 0:
            raise ValueError(f"inference.layer_batch_size must be positive when set, got {self.layer_batch_size}.")
        if not 0.0 <= float(self.delta_smoothing_alpha) <= 1.0:
            raise ValueError(
                "inference.delta_smoothing_alpha must be in [0, 1], "
                f"got {self.delta_smoothing_alpha}."
            )
        if not _is_non_negative_integer(self.delta_smoothing_steps):
            raise ValueError(
                "inference.delta_smoothing_steps must be a non-negative integer, "
                f"got {self.delta_smoothing_steps}."
            )
        if self.cloud_max_nodes_per_layer is not None and int(self.cloud_max_nodes_per_layer) < 3:
            raise ValueError(
                "inference.cloud_max_nodes_per_layer must be at least 3 when set, "
                f"got {self.cloud_max_nodes_per_layer}."
            )


@dataclass(frozen=True)
class SingleLayerInferenceRunConfig:
    output_path: str = "../runs/pdgcn/single_layer_prediction.h5"
    dataset_index: int = 0
    h5_path: Optional[str] = None
    h5_dir: Optional[str] = None
    output_dir: Optional[str] = None
    output_prefix: str = "pre_"
    prediction_group_path: str = "prediction/pdgcn_single_layer"
    batch_mode: bool = False
    steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    mode: str = "both"
    write_vtu: bool = True
    vtu_interval: int = 20
    vtu_output_dir: Optional[str] = None
    fem_temperature_dataset: str = "fem/temperature"
    fem_valid_mask_dataset: Optional[str] = "fem/valid_mask"
    write_fem_vtu: bool = True

    def __post_init__(self):
        if int(self.dataset_index) < 0:
            raise ValueError(
                f"single_layer_inference.dataset_index must be non-negative, got {self.dataset_index}."
            )
        if not isinstance(self.batch_mode, bool):
            raise ValueError("single_layer_inference.batch_mode must be a boolean.")
        if not isinstance(self.write_fem_vtu, bool):
            raise ValueError("single_layer_inference.write_fem_vtu must be a boolean.")
        if self.h5_dir is not None and (not isinstance(self.h5_dir, str) or not self.h5_dir.strip()):
            raise ValueError("single_layer_inference.h5_dir must be null or a non-empty string.")
        if self.output_dir is not None and (
            not isinstance(self.output_dir, str) or not self.output_dir.strip()
        ):
            raise ValueError("single_layer_inference.output_dir must be null or a non-empty string.")
        if not isinstance(self.output_prefix, str) or not self.output_prefix:
            raise ValueError("single_layer_inference.output_prefix must be a non-empty string.")
        if not isinstance(self.prediction_group_path, str) or not self.prediction_group_path.strip():
            raise ValueError("single_layer_inference.prediction_group_path must be a non-empty string.")
        _validate_hdf5_group_path(self.prediction_group_path, "single_layer_inference.prediction_group_path")
        if self.steps is not None and int(self.steps) <= 0:
            raise ValueError(f"single_layer_inference.steps must be positive when set, got {self.steps}.")
        if self.warmup_steps is not None and int(self.warmup_steps) < 0:
            raise ValueError(
                f"single_layer_inference.warmup_steps must be non-negative when set, got {self.warmup_steps}."
            )
        mode = str(self.mode).strip().lower()
        if mode not in {"autoregressive", "teacher_forcing", "both"}:
            raise ValueError(
                "single_layer_inference.mode must be one of "
                "'autoregressive', 'teacher_forcing', or 'both', "
                f"got {self.mode!r}."
            )
        if int(self.vtu_interval) <= 0:
            raise ValueError(
                f"single_layer_inference.vtu_interval must be positive, got {self.vtu_interval}."
            )
        if not isinstance(self.fem_temperature_dataset, str) or not self.fem_temperature_dataset.strip():
            raise ValueError("single_layer_inference.fem_temperature_dataset must be a non-empty string.")
        if self.fem_valid_mask_dataset is not None and (
            not isinstance(self.fem_valid_mask_dataset, str) or not self.fem_valid_mask_dataset.strip()
        ):
            raise ValueError(
                "single_layer_inference.fem_valid_mask_dataset must be null or a non-empty string."
            )


def _validate_hdf5_group_path(value: str, name: str):
    parts = str(value).strip("/").split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must be a relative HDF5 group path, got {value!r}.")
