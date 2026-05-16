› 现在介绍一下每一个epoch的流程


• 下面按当前修改后的代码说明。现在已经去掉了每个 HDF5 文件结束后重新评估首切片，所以一个 epoch 不再额外跑 evaluate_static_topology_sequence(readers[0])。

  总体流程

  一个 epoch 的主循环在 training/static_topology.py:251：

  for epoch:
      model.train()

      for 每个 HDF5 文件:
          训练这个文件的所有帧
    
      汇总所有 window loss
      写 epoch 监控
      判断是否 early stop

  更具体地说：

  1. 进入 epoch

     每个 epoch 开始时执行：

     model.train()
     window_records = []
     file_window_counts = []
     last_snapshot = None

     这里 window_records 会收集本 epoch 中所有 HDF5 文件、所有 TBPTT window 的损失记录。

  2. 逐个 HDF5 文件训练

     每个 HDF5 文件会调用一次：

     _train_one_static_sequence_epoch(...)

     也就是说，一个 epoch 会把训练目录里的所有 .h5/.hdf5 文件按自然排序扫一遍。

  3. 每个 HDF5 文件开始时重新初始化温度

     在每个文件内部，先生成该文件的初始温度。

     如果 warmup_steps > 0，会读取该 HDF5 文件第 0 帧，然后用当前模型做伪时间 warmup：

     current_temperature = cold initial temperature
     重复 warmup_steps 次:
         delta_T = model(graph)
         current_temperature = current_temperature + delta_T
         施加边界条件

     你的示例配置里是：

     "warmup_steps": 30

     所以每个 HDF5 文件正式训练前，会额外做 30 次 no-grad 前向，用来生成当前文件的初温。

  4. 按 TBPTT window 分段

     每个文件的帧序列会按 tbptt_window 切成多个窗口：

     for start in range(0, frame_reader.num_frames, tbptt_window):

     例如一个文件有 68 帧，tbptt_window=5，则大约是：

     ceil(68 / 5) = 14 个 window

  5. 每个 window 内逐帧前向

     每个 window 里，对每一帧依次执行：

     读取当前帧 HDF5 数据
     构建当前帧图 graph
     PDGCN 前向预测 delta_T
     更新温度 T_next = T_current + delta_T
     施加 Dirichlet 边界条件
     计算 PDE residual 和 loss 分量
     把 T_next 作为下一帧的 T_current

     这一段对应 training/static_topology.py:335。

  6. 每个 window 做一次反向传播

     window 内每一帧都会产生一个 loss_total。

     代码会对这些帧的 loss 求平均：

     loss = torch.stack(loss_terms).mean()

     然后：

     loss.backward()
     optimizer.step()

     所以参数更新频率是：

     每个 TBPTT window 更新一次参数

     不是每帧更新一次，也不是每个 HDF5 文件更新一次。

  7. 每个 window 记录损失

     每个 window 结束后记录：

     loss_total
     loss_pde
     loss_outflow
     loss_beta
     temperature_mean
     temperature_max
     temperature_min
     temperature_var

     其中训练真正用于反向传播的是：

     loss_total = loss_pde + lambda_outflow * loss_outflow

     loss_beta 当前只作为监控项记录，不加入 loss_total。

  8. 一个 HDF5 文件结束

     当前修改后，文件结束后只做：

     把该文件产生的 window_records 合并到 epoch 的 window_records
     记录该文件有多少个 window
     保存可能的 epoch 快照引用

     已经不再执行：

     重新评估第一个 HDF5 文件
     写 first_slice_loss_curve
     写 first_slice residual / temperature 快照

  9. 所有 HDF5 文件训练完后，汇总 epoch loss

     所有文件都跑完后，代码取出本 epoch 的所有 window loss：

     window_losses = [record["loss_total"] for record in window_records]

     然后计算 epoch 输出的 Loss：

     epoch_loss = sum(window_losses) / len(window_losses)

     所以你看到的：

     Epoch x/y - loss=...

     是：

     当前 epoch 内，所有 HDF5 文件、所有 TBPTT window 的 loss_total 平均值

  10. 写入监控数据

  如果 monitoring 开启，会调用 TrainingProcessMonitor。

  它会写：

  history.json
  metrics/monitor_data.h5

  monitor_data.h5 中会保存：

  epoch_metrics/
      epoch
      loss_total
      loss_pde
      loss_outflow
      loss_beta
      temperature_mean
      temperature_max
      temperature_min
      temperature_var

  epoch_snapshots/
      epoch_XXXX/
          coords
          residual
          temperature

  当前已去掉首切片监控，所以：

  slice_metrics 为空
  slice_snapshots 为空

  11. 判断是否提前停止

  如果配置了：

  "loss_threshold": ...

  并且：

  epoch_loss < loss_threshold

  则当前 epoch 会标记：

  stopped_early = True
  stop_reason = "loss_threshold"

  然后训练结束。

  一句话总结

  现在每个 epoch 的流程是：

  扫一遍所有 HDF5 文件
  每个文件先 warmup 得到初温
  每个文件按 TBPTT window 逐帧训练
  每个 window 更新一次模型参数
  最后把所有 window 的 loss_total 求平均，作为该 epoch 输出的 Loss
  写 epoch 级 HDF5 监控
  检查是否 early stop

  当前已经不再有“每个 HDF5 文件结束后额外重新评估首切片”的开销