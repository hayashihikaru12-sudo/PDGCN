from dataclasses import dataclass
from typing import Optional


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
    return_dimensionless: bool = False
    write_vtk: bool = True
    vtk_interval: int = 20
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
        if int(self.vtk_interval) <= 0:
            raise ValueError(f"inference.vtk_interval must be positive, got {self.vtk_interval}.")
