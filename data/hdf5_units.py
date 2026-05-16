"""HDF5 native-unit conversion helpers.

The generated slice HDF5 files use millimetre-based native units.  The data
pipeline converts them to SI before dimensionless scaling and PDE loss.
"""

from typing import Optional


MM_TO_M = 1e-3
MM_PER_S_TO_M_PER_S = 1e-3
W_PER_MM2_TO_W_PER_M2 = 1e6


def resolve_heat_source_effective_thickness(
    *,
    scale_params=None,
    heat_source_effective_thickness: Optional[float] = None,
) -> float:
    if heat_source_effective_thickness is None and scale_params is not None:
        heat_source_effective_thickness = getattr(scale_params, "heat_source_effective_thickness", None)
    if heat_source_effective_thickness is None:
        raise ValueError(
            "heat_source_effective_thickness is required to convert HDF5 dynamic/Q "
            "from W/mm^2 to W/m^3."
        )
    thickness = float(heat_source_effective_thickness)
    if thickness <= 0:
        raise ValueError(f"heat_source_effective_thickness must be positive, got {thickness}.")
    return thickness


def length_mm_to_m(value):
    return value * MM_TO_M


def velocity_mm_per_s_to_m_per_s(value):
    return value * MM_PER_S_TO_M_PER_S


def heat_flux_w_per_mm2_to_volume_w_per_m3(value, heat_source_effective_thickness: float):
    return value * (W_PER_MM2_TO_W_PER_M2 / float(heat_source_effective_thickness))
