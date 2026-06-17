"""诊断 Q=0 vs Q≠0 反直觉温度行为的脚本。

用法:
    cd d:\ProLab\PIGNN\PDGCN
    D:\ProgramData\CondaEnv\PIGNN\python.exe scripts\diagnose_q_zero.py

该脚本会:
1. 打印 source_coefficient、dt_star、inverse_pe 等关键物理参数
2. 打印 decoder 最后一层的 bias（检查是否存在系统性偏置）
3. 对 Q=0 和 Q>0 各运行少量步数，逐帧打印 ΔT_source、ΔT_inplane 和温度统计
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# 确保项目根目录在 path 中
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.dimensionless import ScaleParams, temperature_from_dimensionless
from data.static_cache import HDF5FrameReader, STATIC_FILE, META_FILE
from inference.io import load_model_from_checkpoint
from inference.single_layer import (
    _build_single_layer_inference_run_config,
    _ensure_static_cache,
    load_single_layer_inference_run_context,
)
from pde import apply_dirichlet_boundary
from training.graph_utils import clone_graph_with_temperature, graph_explicit_source_delta
from training.run_config import pdgcn_config_from_scale
from training.static_topology import GpuFeatureBuilder, StaticGraphState
from training.train_entry import derive_timing_from_hdf5


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_model_params(model, scale_params, timing):
    """打印关键物理参数和模型参数。"""
    config = model.config
    print_separator("关键物理参数")
    print(f"  L0                  = {scale_params.L0:.6g} m")
    print(f"  v0                  = {scale_params.v0:.6g} m/s")
    print(f"  T_amb               = {scale_params.T_amb:.6g} °C")
    print(f"  delta_T0            = {scale_params.delta_T0:.6g} K")
    print(f"  Q0                  = {scale_params.Q0:.6g} W/m²")
    print(f"  K0                  = {scale_params.K0:.6g} W/(m·K)")
    print(f"  rho                 = {scale_params.rho:.6g} kg/m³")
    print(f"  Cp                  = {scale_params.Cp:.6g} J/(kg·K)")
    print(f"  h_eff               = {scale_params.heat_source_effective_thickness:.6g} m")
    print(f"  absorptivity        = {scale_params.heat_source_absorptivity:.6g}")
    print()
    print(f"  dt (real)           = {timing['dt']:.6g} s")
    print(f"  dt_star             = {config.dt_star:.6g}")
    print(f"  inverse_pe          = {config.inverse_pe:.6g}")
    print(f"  source_coefficient  = {config.source_coefficient:.6g}")
    print(f"  k_ratio             = {config.k_ratio:.6g}")
    print(f"  residual_time_scheme = {config.residual_time_scheme}")
    print(f"  dirichlet_T*        = {config.dirichlet_temperature_star:.6g}")

    # 计算 ΔT_source 量级估计
    # 如果 Q = Q0，则 q* = 1，ΔT_Q* = source_coefficient * dt_star
    delta_per_step = config.source_coefficient * config.dt_star
    print(f"\n  若 q*=1 (即 Q=Q0): ΔT_Q* ≈ {delta_per_step:.6g} /step")
    print(f"  对应物理温升: {delta_per_step * scale_params.delta_T0:.6g} °C/step")


def print_decoder_bias(model):
    """打印 decoder 最后一层 bias，检查系统性偏置。"""
    print_separator("Decoder 偏置检查")
    decoder = model.decoder.decoder
    # build_mlp 创建的是 nn.Sequential，最后一层是 nn.Linear
    for name, module in decoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            bias = module.bias.data
            print(f"  Layer '{name}': bias shape={list(bias.shape)}, "
                  f"min={bias.min().item():.6g}, max={bias.max().item():.6g}, "
                  f"mean={bias.mean().item():.6g}")
    # 找到最后一层 Linear
    last_linear = None
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    if last_linear is not None:
        print(f"\n  最后一层 output_size={last_linear.out_features}")
        print(f"  最后一层 bias 均值: {last_linear.bias.data.mean().item():.6g}")
        print(f"  最后一层 bias 标准差: {last_linear.bias.data.std().item():.6g}")


@torch.no_grad()
def run_diagnostic_steps(
    model,
    frame_reader,
    static_state,
    feature_builder,
    scale_params,
    num_steps: int = 10,
    label: str = "",
):
    """运行少量推理步数并打印每步的诊断信息。"""
    print_separator(f"诊断推理: {label} ({num_steps} 步)")

    model.to(static_state.device)
    model.eval()

    current_temperature = feature_builder.initial_temperature()
    print(f"  初始 T*: min={current_temperature.min().item():.6g}, "
          f"max={current_temperature.max().item():.6g}, "
          f"mean={current_temperature.mean().item():.6g}")

    for step in range(min(num_steps, frame_reader.num_frames)):
        node_base_cpu, global_cpu = frame_reader.read_frame(step)
        graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)

        # 源项
        delta_source = graph_explicit_source_delta(graph, model.config)
        q_star = graph.q_surface_star if hasattr(graph, "q_surface_star") else None
        q_stats = ""
        if q_star is not None:
            q_vals = q_star.detach().cpu().numpy()
            q_stats = (f"q*: min={q_vals.min():.4g}, max={q_vals.max():.4g}, "
                       f"mean={q_vals.mean():.4g}, nonzero={(q_vals != 0).sum()}/{q_vals.size}")

        source_temperature = apply_dirichlet_boundary(
            current_temperature + delta_source,
            static_state.boundary_nodes,
            value=getattr(model.config, "dirichlet_temperature_star", 0.0),
        )

        # 模型推理
        graph_input = clone_graph_with_temperature(graph, source_temperature, delta_t_source_star=delta_source)
        delta_inplane = model(graph_input)

        next_temperature = apply_dirichlet_boundary(
            source_temperature + delta_inplane,
            static_state.boundary_nodes,
            value=getattr(model.config, "dirichlet_temperature_star", 0.0),
        )

        # 统计
        ds = delta_source.detach().cpu().numpy()
        di = delta_inplane.detach().cpu().numpy()
        t_curr = current_temperature.detach().cpu().numpy()
        t_src = source_temperature.detach().cpu().numpy()
        t_next = next_temperature.detach().cpu().numpy()

        # 物理温度
        t_next_phys = temperature_from_dimensionless(next_temperature.detach(), scale_params).cpu().numpy()

        print(f"\n  Step {step}:")
        print(f"    {q_stats}")
        print(f"    ΔT_source*:      min={ds.min():.6g}, max={ds.max():.6g}, mean={ds.mean():.6g}")
        print(f"    ΔT_inplane*:     min={di.min():.6g}, max={di.max():.6g}, mean={di.mean():.6g}")
        print(f"    T_curr*:         min={t_curr.min():.6g}, max={t_curr.max():.6g}, mean={t_curr.mean():.6g}")
        print(f"    T_source*:       min={t_src.min():.6g}, max={t_src.max():.6g}, mean={t_src.mean():.6g}")
        print(f"    T_next*:         min={t_next.min():.6g}, max={t_next.max():.6g}, mean={t_next.mean():.6g}")
        print(f"    T_next (phys °C): min={t_next_phys.min():.2f}, max={t_next_phys.max():.2f}, mean={t_next_phys.mean():.2f}")
        print(f"    净温升/步:       mean = {t_next.mean() - t_curr.mean():.6g}")

        # 关键检查：ΔT_inplane 是否过度为负
        if ds.mean() > 0:
            net = ds.mean() + di.mean()
            if net < -1e-8:
                print(f"    ⚠️  ΔT_source > 0 但净效应为负! (source={ds.mean():.6g}, inplane={di.mean():.6g}, net={net:.6g})")

        current_temperature = next_temperature

    # 最终物理温度统计
    final_phys = temperature_from_dimensionless(current_temperature.detach(), scale_params).cpu().numpy()
    print(f"\n  最终物理温度: min={final_phys.min():.2f}°C, max={final_phys.max():.2f}°C, mean={final_phys.mean():.2f}°C")


def main():
    # 配置路径
    config_path = REPO_ROOT / "configs" / "pdgcn_single_layer_infer.example.json"
    if not config_path.exists():
        print(f"错误: 找不到配置文件 {config_path}")
        print("请修改脚本中的 config_path 指向你的实际配置文件")
        return 1

    print(f"使用配置: {config_path}")

    # 加载配置上下文
    (
        run_config,
        inference_config,
        training_base_dir,
        inference_base_dir,
        training_config_path,
    ) = load_single_layer_inference_run_context(config_path)

    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()

    # 解析路径
    from inference.io import _resolve_path
    from training.train_entry import discover_hdf5_files

    selected_h5 = (
        _resolve_path(inference_base_dir, inference_config.h5_path)
        if inference_config.h5_path
        else discover_hdf5_files(_resolve_path(training_base_dir, dataset.h5_dir))[0]
    )
    selected_checkpoint = _resolve_path(
        training_base_dir,
        run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
    )

    print(f"HDF5 文件: {selected_h5}")
    print(f"Checkpoint: {selected_checkpoint}")

    # 检查文件是否存在
    if not Path(selected_h5).exists():
        print(f"⚠️  HDF5 文件不存在: {selected_h5}")
        print("请修改配置文件中的 h5_path 或使用 --h5 参数指定")
        return 1
    if not Path(selected_checkpoint).exists():
        print(f"⚠️  Checkpoint 不存在: {selected_checkpoint}")
        print("请先训练模型或修改配置文件中的 checkpoint_path")
        return 1

    # 派生时间和模型配置
    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=dataset.scan_velocity)
    fallback_model_config = pdgcn_config_from_scale(
        scale_params,
        dt=timing["dt"],
        model_overrides=run_config.model,
    )
    device = torch.device(run_config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备: {device}")

    # 加载模型
    model, checkpoint_payload = load_model_from_checkpoint(selected_checkpoint, fallback_model_config, device)
    print(f"模型 epoch: {checkpoint_payload.get('epoch', 'unknown')}")

    # 缓存
    cache_dir = _resolve_path(training_base_dir, dataset.cache_dir)
    _ensure_static_cache(selected_h5, cache_dir, scale_params, scan_velocity=dataset.scan_velocity)
    static_state = StaticGraphState.from_cache(cache_dir, device=device)

    # 1) 打印模型参数
    print_model_params(model, scale_params, timing)

    # 2) 打印 decoder bias
    print_decoder_bias(model)

    # 3) 使用实际 HDF5 运行诊断
    feature_builder = GpuFeatureBuilder(static_state, scale_params, model_config=model.config)
    with HDF5FrameReader(
        selected_h5,
        expected_num_nodes=static_state.num_nodes,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
    ) as frame_reader:
        q_values = frame_reader.q[:10, :, :]  # 读取前10帧的Q
        q_raw = np.asarray(q_values)
        print_separator("HDF5 Q 值检查 (前10帧)")
        print(f"  Q shape: {q_raw.shape}")
        print(f"  Q min: {q_raw.min():.6g} W/mm²")
        print(f"  Q max: {q_raw.max():.6g} W/mm²")
        print(f"  Q mean: {q_raw.mean():.6g} W/mm²")
        print(f"  Q 全零? {(q_raw == 0).all()}")

        run_diagnostic_steps(
            model,
            frame_reader,
            static_state,
            feature_builder,
            scale_params,
            num_steps=10,
            label=f"实际HDF5 ({selected_h5.name})",
        )

    # 4) 零输入测试：将温度固定为0，看模型输出什么
    print_separator("零输入测试 (T*=0 时模型输出)")
    model.eval()
    with torch.no_grad():
        # 构造一个全零温度的图
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        zero_temp = torch.zeros(static_state.num_nodes, 1, device=device, dtype=torch.float32)
        graph_zero = feature_builder.build(node_base_cpu, global_cpu, zero_temp)
        # 注意：graph_zero 的 x[:, 6:7] 已经 = 0
        delta_zero = model(graph_zero)
        d0 = delta_zero.detach().cpu().numpy()
        print(f"  输入 T*=0 时模型输出 ΔT*:")
        print(f"    min={d0.min():.6g}, max={d0.max():.6g}, mean={d0.mean():.6g}")
        print(f"    std={d0.std():.6g}")
        if abs(d0.mean()) > 1e-6:
            print(f"  ⚠️  模型在 T*=0 时输出非零均值 {d0.mean():.6g} —— 存在系统性偏置!")
            print(f"  这解释了 Q=0 时的虚假温度漂移。")

        # 再测试 T*=1.0 时的输出
        one_temp = torch.ones(static_state.num_nodes, 1, device=device, dtype=torch.float32)
        graph_one = feature_builder.build(node_base_cpu, global_cpu, one_temp)
        delta_one = model(graph_one)
        d1 = delta_one.detach().cpu().numpy()
        print(f"\n  输入 T*=1.0 时模型输出 ΔT*:")
        print(f"    min={d1.min():.6g}, max={d1.max():.6g}, mean={d1.mean():.6g}")
        print(f"    std={d1.std():.6g}")

        print(f"\n  对比:")
        print(f"    ΔT*(T*=0)  mean = {d0.mean():.6g}")
        print(f"    ΔT*(T*=1)  mean = {d1.mean():.6g}")
        print(f"    差值 = {d1.mean() - d0.mean():.6g}")

    print_separator("诊断完成")
    print("\n解读指南:")
    print("  1. 如果 '零输入测试' 中模型在 T*=0 时输出显著的均值偏移,")
    print("     说明模型在训练分布外 (T≈0) 存在系统性偏置。")
    print("  2. 对比 Q=0 vs Q≠0 推理中的 ΔT_inplane 均值差异,")
    print("     看模型是否在较高温度时过度预测冷却。")
    print("  3. 检查 decoder 最后一层 bias 是否存在较大的负偏置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
