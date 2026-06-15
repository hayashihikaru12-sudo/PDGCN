import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from data.dimensionless import ScaleParams, derive_pde_constants
from inference.config import InferenceRunConfig
from models import PDGCNConfig

from .config import TrainConfig


DERIVED_PDGCN_FIELDS = {"inverse_pe", "pi_q", "source_coefficient", "dt_star"}
PHYSICS_LOSS_FIELDS = {
    "k_ratio",
    "lambda_pde",
    "lambda_outflow",
    "gradient_regularization",
    "dirichlet_temperature_star",
    "thermal_loss_beta",
    "thermal_loss_base_temperature_star",
    "residual_time_scheme",
}


@dataclass(frozen=True)
class DataRunConfig:
    h5_dir: str
    cache_dir: str
    checkpoint_path: str
    history_path: Optional[str] = None
    scan_velocity: Optional[float] = None


@dataclass(frozen=True)
class OutputRunConfig:
    checkpoint_path: str
    history_path: Optional[str] = None


@dataclass(frozen=True)
class MonitoringRunConfig:
    enabled: bool = True
    interval_epochs: int = 10
    temperature_frame_index: Optional[int] = None
    figures_dir: Optional[str] = None
    metrics_path: Optional[str] = None

    def __post_init__(self):
        if int(self.interval_epochs) <= 0:
            raise ValueError(f"monitoring.interval_epochs must be positive, got {self.interval_epochs}.")
        if self.temperature_frame_index is not None and int(self.temperature_frame_index) < 0:
            raise ValueError(
                "monitoring.temperature_frame_index must be non-negative when set, "
                f"got {self.temperature_frame_index}."
            )


@dataclass(frozen=True)
class SupervisionRunConfig:
    enabled: bool = False
    temperature_dataset: str = "fem/temperature"
    valid_mask_dataset: Optional[str] = "fem/valid_mask"
    lambda_temperature: float = 1.0
    lambda_rollout_temperature: float = 1.0
    rollout_window: Optional[int] = None
    mode: str = "teacher_forcing"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"supervision.enabled must be a boolean, got {self.enabled!r}.")
        if not isinstance(self.temperature_dataset, str) or not self.temperature_dataset.strip():
            raise ValueError("supervision.temperature_dataset must be a non-empty string.")
        if self.valid_mask_dataset is not None and (
            not isinstance(self.valid_mask_dataset, str) or not self.valid_mask_dataset.strip()
        ):
            raise ValueError("supervision.valid_mask_dataset must be null or a non-empty string.")
        if float(self.lambda_temperature) < 0:
            raise ValueError(
                "supervision.lambda_temperature must be non-negative, "
                f"got {self.lambda_temperature}."
            )
        if float(self.lambda_rollout_temperature) < 0:
            raise ValueError(
                "supervision.lambda_rollout_temperature must be non-negative, "
                f"got {self.lambda_rollout_temperature}."
            )
        if self.rollout_window is not None and int(self.rollout_window) <= 0:
            raise ValueError(
                "supervision.rollout_window must be null or a positive integer, "
                f"got {self.rollout_window}."
            )
        mode = str(self.mode).strip().lower()
        if mode not in {"teacher_forcing", "rollout", "mixed"}:
            raise ValueError(
                "supervision.mode must be one of 'teacher_forcing', 'rollout', or 'mixed', "
                f"got {self.mode!r}."
            )
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class ScaleRunConfig:
    L0: float
    v0: float
    T_amb: float
    delta_T0: float
    Q0: float
    K0: float
    rho: float
    Cp: float
    heat_source_effective_thickness: float
    heat_source_absorptivity: float = 1.0
    eps: float = 1e-12

    def __post_init__(self):
        if float(self.heat_source_effective_thickness) <= 0:
            raise ValueError(
                "scale.heat_source_effective_thickness must be positive, "
                f"got {self.heat_source_effective_thickness}."
            )
        if float(self.heat_source_absorptivity) < 0:
            raise ValueError(
                "scale.heat_source_absorptivity must be non-negative, "
                f"got {self.heat_source_absorptivity}."
            )

    def to_scale_params(self) -> ScaleParams:
        """转换为数据流水线使用的 ``ScaleParams``。"""

        return ScaleParams(
            L0=self.L0,
            v0=self.v0,
            T_amb=self.T_amb,
            delta_T0=self.delta_T0,
            Q0=self.Q0,
            K0=self.K0,
            rho=self.rho,
            Cp=self.Cp,
            heat_source_effective_thickness=self.heat_source_effective_thickness,
            heat_source_absorptivity=self.heat_source_absorptivity,
            eps=self.eps,
        )


@dataclass(frozen=True)
class DatasetRunConfig:
    h5_dir: str
    cache_dir: str
    scale: ScaleRunConfig
    name: str = ""
    scan_velocity: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    data: DataRunConfig
    scale: ScaleRunConfig
    model: Dict[str, Any]
    training: TrainConfig
    monitoring: MonitoringRunConfig = field(default_factory=MonitoringRunConfig)
    supervision: SupervisionRunConfig = field(default_factory=SupervisionRunConfig)
    inference: Optional[InferenceRunConfig] = None
    outputs: Optional[OutputRunConfig] = None
    datasets: Tuple[DatasetRunConfig, ...] = ()
    schema: str = "legacy"


def derive_dt_star(scale_params: ScaleParams, dt: float) -> float:
    """根据真实时间步计算无量纲时间步 ``dt*``。"""

    if float(dt) <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    return float(dt) / (float(scale_params.L0) / float(scale_params.v0))


def pdgcn_config_from_scale(
    scale_params: ScaleParams,
    *,
    dt: float,
    model_overrides: Optional[Dict[str, Any]] = None,
) -> PDGCNConfig:
    """从 ``ScaleParams`` 自动派生物理系数并构造 ``PDGCNConfig``。"""

    inverse_pe, source_coefficient = derive_pde_constants(scale_params)
    dt_star = derive_dt_star(scale_params, dt)
    overrides = _filter_dataclass_kwargs(PDGCNConfig, model_overrides or {}, context="model")
    for name in DERIVED_PDGCN_FIELDS:
        overrides.pop(name, None)
    overrides.update(
        {
            "inverse_pe": inverse_pe,
            "source_coefficient": source_coefficient,
            "pi_q": source_coefficient,
            "heat_source_absorptivity": float(scale_params.heat_source_absorptivity),
            "dt_star": dt_star,
        }
    )
    return PDGCNConfig(**overrides)


def load_run_config(path) -> RunConfig:
    """从 JSON 文件读取一键训练配置。"""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run config JSON must contain an object at the top level.")

    if _is_classified_schema(payload):
        return _load_classified_run_config(payload)
    return _load_legacy_run_config(payload)


def _load_legacy_run_config(payload: Dict[str, Any]) -> RunConfig:
    data = _build_dataclass(DataRunConfig, payload.get("data"), context="data")
    scale = _build_dataclass(ScaleRunConfig, payload.get("scale"), context="scale")
    model = _require_mapping(payload.get("model", {}), context="model")
    monitoring = _build_monitoring_run_config(payload.get("monitoring"))
    supervision = _build_supervision_run_config(payload.get("supervision"))
    inference = _build_inference_run_config(payload.get("inference"))
    training_kwargs = _filter_dataclass_kwargs(
        TrainConfig,
        _require_mapping(payload.get("training", {}), context="training"),
        context="training",
    )
    return RunConfig(
        data=data,
        scale=scale,
        model=dict(model),
        training=TrainConfig(**training_kwargs),
        monitoring=monitoring,
        supervision=supervision,
        inference=inference,
        datasets=(
            DatasetRunConfig(
                h5_dir=data.h5_dir,
                cache_dir=data.cache_dir,
                scale=scale,
                scan_velocity=data.scan_velocity,
            ),
        ),
        outputs=OutputRunConfig(
            checkpoint_path=data.checkpoint_path,
            history_path=data.history_path,
        ),
        schema="legacy",
    )


def _load_classified_run_config(payload: Dict[str, Any]) -> RunConfig:
    outputs = _build_dataclass(OutputRunConfig, payload.get("outputs"), context="outputs")
    monitoring = _build_monitoring_run_config(payload.get("monitoring"))
    supervision = _build_supervision_run_config(payload.get("supervision"))
    inference = _build_inference_run_config(payload.get("inference"))
    dataset_payloads = payload.get("datasets")
    if not isinstance(dataset_payloads, list) or not dataset_payloads:
        raise ValueError("'datasets' section must be a non-empty list.")
    datasets = tuple(
        _build_dataset_run_config(dataset_payload, index=index)
        for index, dataset_payload in enumerate(dataset_payloads)
    )

    hyperparameters = _require_mapping(payload.get("hyperparameters"), context="hyperparameters")
    model = _classified_model_overrides(hyperparameters)
    training_kwargs = _filter_dataclass_kwargs(
        TrainConfig,
        _require_mapping(hyperparameters.get("training"), context="hyperparameters.training"),
        context="hyperparameters.training",
    )

    first_dataset = datasets[0]
    data = DataRunConfig(
        h5_dir=first_dataset.h5_dir,
        cache_dir=first_dataset.cache_dir,
        checkpoint_path=outputs.checkpoint_path,
        history_path=outputs.history_path,
        scan_velocity=first_dataset.scan_velocity,
    )
    return RunConfig(
        data=data,
        scale=first_dataset.scale,
        model=model,
        training=TrainConfig(**training_kwargs),
        monitoring=monitoring,
        supervision=supervision,
        inference=inference,
        outputs=outputs,
        datasets=datasets,
        schema="classified",
    )


def run_config_to_dict(config: RunConfig) -> Dict[str, Any]:
    """将运行配置转换为可写入 JSON/checkpoint metadata 的字典。"""

    if config.schema == "classified":
        model, physics_loss = _split_model_and_physics(config.model)
        return {
            "schema": config.schema,
            "outputs": asdict(config.outputs) if config.outputs is not None else None,
            "datasets": [asdict(dataset) for dataset in config.datasets],
            "hyperparameters": {
                "model": model,
                "physics_loss": physics_loss,
                "training": asdict(config.training),
            },
            "monitoring": asdict(config.monitoring),
            "supervision": asdict(config.supervision),
            "inference": asdict(config.inference) if config.inference is not None else None,
        }
    return {
        "data": asdict(config.data),
        "scale": asdict(config.scale),
        "model": dict(config.model),
        "training": asdict(config.training),
        "monitoring": asdict(config.monitoring),
        "supervision": asdict(config.supervision),
        "inference": asdict(config.inference) if config.inference is not None else None,
    }


def _is_classified_schema(payload: Dict[str, Any]) -> bool:
    return any(name in payload for name in ("outputs", "datasets", "hyperparameters"))


def _build_dataset_run_config(value, *, index: int) -> DatasetRunConfig:
    mapping = _require_mapping(value, context=f"datasets[{index}]")
    valid = {"name", "h5_dir", "cache_dir", "scale", "scan_velocity"}
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in 'datasets[{index}]' section: {unknown}")
    for required_key in ("h5_dir", "cache_dir", "scale"):
        if required_key not in mapping:
            raise ValueError(f"Missing required 'datasets[{index}].{required_key}' field.")

    scale = _build_dataclass(ScaleRunConfig, mapping.get("scale"), context=f"datasets[{index}].scale")
    return DatasetRunConfig(
        name=str(mapping.get("name", f"dataset_{index}")),
        h5_dir=str(mapping["h5_dir"]),
        cache_dir=str(mapping["cache_dir"]),
        scale=scale,
        scan_velocity=mapping.get("scan_velocity"),
    )


def _classified_model_overrides(hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    model = dict(_require_mapping(hyperparameters.get("model", {}), context="hyperparameters.model"))
    physics_loss = dict(
        _require_mapping(hyperparameters.get("physics_loss", {}), context="hyperparameters.physics_loss")
    )
    overlap = sorted(set(model) & set(physics_loss))
    if overlap:
        raise ValueError(f"Duplicate keys between model and physics_loss hyperparameters: {overlap}")
    model.update(physics_loss)
    return model


def _split_model_and_physics(model_overrides: Dict[str, Any]):
    model = {}
    physics_loss = {}
    for key, value in model_overrides.items():
        if key in PHYSICS_LOSS_FIELDS:
            physics_loss[key] = value
        else:
            model[key] = value
    return model, physics_loss


def _build_dataclass(cls, value, *, context: str):
    mapping = _require_mapping(value, context=context)
    kwargs = _filter_dataclass_kwargs(cls, mapping, context=context)
    return cls(**kwargs)


def _build_monitoring_run_config(value) -> MonitoringRunConfig:
    if value is None:
        return MonitoringRunConfig()
    mapping = _require_mapping(value, context="monitoring")
    kwargs = _filter_dataclass_kwargs(MonitoringRunConfig, mapping, context="monitoring")
    return MonitoringRunConfig(**kwargs)


def _build_supervision_run_config(value) -> SupervisionRunConfig:
    if value is None:
        return SupervisionRunConfig()
    mapping = _require_mapping(value, context="supervision")
    kwargs = _filter_dataclass_kwargs(SupervisionRunConfig, mapping, context="supervision")
    return SupervisionRunConfig(**kwargs)


def _build_inference_run_config(value) -> Optional[InferenceRunConfig]:
    if value is None:
        return None
    mapping = _require_mapping(value, context="inference")
    kwargs = _filter_dataclass_kwargs(InferenceRunConfig, mapping, context="inference")
    return InferenceRunConfig(**kwargs)


def _require_mapping(value, *, context: str) -> Dict[str, Any]:
    if value is None:
        raise ValueError(f"Missing required '{context}' section in run config.")
    if not isinstance(value, dict):
        raise ValueError(f"'{context}' section must be an object.")
    return value


def _filter_dataclass_kwargs(cls, mapping: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    field_defs = fields(cls)
    valid = {field.name for field in field_defs}
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in '{context}' section: {unknown}")
    missing = [
        field.name
        for field in field_defs
        if field.default is MISSING and field.default_factory is MISSING and field.name not in mapping
    ]
    if missing:
        raise ValueError(f"Missing required keys in '{context}' section: {missing}")
    return dict(mapping)
