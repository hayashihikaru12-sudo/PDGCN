import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

from data.dimensionless import ScaleParams, derive_pde_constants
from models import PDGCNConfig

from .config import TrainConfig


DERIVED_PDGCN_FIELDS = {"inverse_pe", "pi_q", "dt_star"}


@dataclass(frozen=True)
class DataRunConfig:
    h5_path: str
    cache_dir: str
    checkpoint_path: str
    history_path: Optional[str] = None
    overwrite_cache: bool = False
    scan_velocity: Optional[float] = None


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
    dt: float
    eps: float = 1e-12

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
            eps=self.eps,
        )


@dataclass(frozen=True)
class RunConfig:
    data: DataRunConfig
    scale: ScaleRunConfig
    model: Dict[str, Any]
    training: TrainConfig


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

    inverse_pe, pi_q = derive_pde_constants(scale_params)
    dt_star = derive_dt_star(scale_params, dt)
    overrides = _filter_dataclass_kwargs(PDGCNConfig, model_overrides or {}, context="model")
    for name in DERIVED_PDGCN_FIELDS:
        overrides.pop(name, None)
    overrides.update({"inverse_pe": inverse_pe, "pi_q": pi_q, "dt_star": dt_star})
    return PDGCNConfig(**overrides)


def load_run_config(path) -> RunConfig:
    """从 JSON 文件读取一键训练配置。"""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run config JSON must contain an object at the top level.")

    data = _build_dataclass(DataRunConfig, payload.get("data"), context="data")
    scale = _build_dataclass(ScaleRunConfig, payload.get("scale"), context="scale")
    model = _require_mapping(payload.get("model", {}), context="model")
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
    )


def run_config_to_dict(config: RunConfig) -> Dict[str, Any]:
    """将运行配置转换为可写入 JSON/checkpoint metadata 的字典。"""

    return {
        "data": asdict(config.data),
        "scale": asdict(config.scale),
        "model": dict(config.model),
        "training": asdict(config.training),
    }


def _build_dataclass(cls, value, *, context: str):
    mapping = _require_mapping(value, context=context)
    kwargs = _filter_dataclass_kwargs(cls, mapping, context=context)
    return cls(**kwargs)


def _require_mapping(value, *, context: str) -> Dict[str, Any]:
    if value is None:
        raise ValueError(f"Missing required '{context}' section in run config.")
    if not isinstance(value, dict):
        raise ValueError(f"'{context}' section must be an object.")
    return value


def _filter_dataclass_kwargs(cls, mapping: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in '{context}' section: {unknown}")
    return dict(mapping)
