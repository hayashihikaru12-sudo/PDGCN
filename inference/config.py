from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class InferenceRunConfig:
    num_layers: int
    layer_spacing: float
    output_path: str = "../runs/pdgcn/multilayer_prediction.h5"
    dataset_index: int = 0
    h5_path: Optional[str] = None
    steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    bottom_temperature_star: float = 0.0
    top_heat_source_only: bool = True
    allow_unstable_fdm: bool = False
    layer_fiber_angles_deg: Optional[Sequence[float]] = None
    normal_offset_sign: int = -1
    return_dimensionless: bool = False
    write_vtk: bool = True
    cloud_interval: int = 20
    layer_batch_size: Optional[int] = None
    cloud_max_nodes_per_layer: Optional[int] = None
    vtk_output_dir: Optional[str] = None

    def __post_init__(self):
        if int(self.num_layers) < 2:
            raise ValueError(f"inference.num_layers must be at least 2, got {self.num_layers}.")
        if float(self.layer_spacing) <= 0:
            raise ValueError(f"inference.layer_spacing must be positive, got {self.layer_spacing}.")
        if int(self.dataset_index) < 0:
            raise ValueError(f"inference.dataset_index must be non-negative, got {self.dataset_index}.")
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
        if self.cloud_max_nodes_per_layer is not None and int(self.cloud_max_nodes_per_layer) < 3:
            raise ValueError(
                "inference.cloud_max_nodes_per_layer must be at least 3 when set, "
                f"got {self.cloud_max_nodes_per_layer}."
            )
