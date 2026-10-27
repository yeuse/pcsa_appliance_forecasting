# 训练诊断与单因素对照

本次新增诊断不改变 control 的模型、损失、早停或最大训练轮数。旧完整结果目录不要覆盖；新版完整实验须使用新输出目录。诊断输出记录在 source snapshot 和 manifest 中。

## 先诊断服务器上的现有模型（不重训）

把更新后的 training 文件夹同步到服务器项目中，在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.diagnose --source-run outputs/source --output outputs/diagnostics --device cuda
```

默认读取三个种子的五个源域阶段，包括 pcsa_base、pcsa、nilm、two_stage、aggregate；只评估 train 和 val，不读取测试指标来做选择。原 checkpoint 只读，校验原始协议、种子、模型源码及数据哈希。输出记录 checkpoint 哈希和原 manifest 哈希。需要服务器已有 best.pt；仅包含 JSON 的结果压缩包不足以重建逐状态诊断。

若要同时诊断全部目标家庭已选中的迁移模型，使用 `--scope all` 和另一个输出目录。目标训练诊断限制在该比例的实际支持窗口内，不扩大到整个训练集。默认评估全量有效窗口；可用 `--max-windows 64` 做等间隔子集检查，子集结果会明确标注，不用于正式比较。

输出例子：

- seed_7/source/pcsa/train_diagnostics.json
- seed_7/source/pcsa/val_diagnostics.json
- seed_7/source/pcsa_base/val_diagnostics.json

## 新训练自动保存的文件

- initial_validation.json：训练前候选的验证指标与诊断，避免 epoch 0 只有一个总 MAE。
- history.json：每轮 train_diagnostics（训练模式、参数持续更新时的在线统计），以及 val.diagnostics（该轮结束的确定性验证统计）。
- selected_train.json、selected_validation.json：加载最佳 checkpoint 后、eval 模式下的训练/验证诊断。比较训练与验证泛化时优先使用这两个文件。

诊断始终 detached/no_grad，不增加反向目标，也不消耗随机数。会增加统计和额外前向的开销，新增运行时间不能直接和旧版混做效率排名。

## 字段解释

`diagnostics.groups` 按真实标签分为 all/on/off，每个数组按空调、冰箱、洗碗机、微波炉排列。positions 是有效预测位置数，重叠窗口重复计入，不是独立事件数；无 ON/OFF 样本时对应值为 null，不用 0 伪装。

- p_on_mean / soft_gate_mean：状态概率及实际软门控的均值。
- p_on_below_005_fraction / p_on_above_05_fraction：状态概率 <0.05 或 >=0.5 的比例，分别在真实 ON/OFF 条件下统计。
- amplitude_mean_kw / amplitude_mae_kw：门控前幅值均值和误差。历史 NILM 的幅值由当前协议的功率/软门控精确恢复，已标明 derived。
- prediction_mean_kw / target_mean_kw / mae_kw / bias_kw：最终预测、真实功率、绝对误差及有符号偏差。
- gate_suppression_kw：幅值减最终预测，表示门控抑制的功率。
- base_prediction_mean_kw / base_mae_kw / base_p_on_mean：PCSA 同次前向中的基础预测分支。
- residual_signed_kw / residual_absolute_kw：最终预测减基础预测的有符号/绝对变化，单位 kW；不是原始 logit 残差。
- residual_mae_gain_kw：基础绝对误差减最终绝对误差，正数才是残差改善。基础分支没有跟随历史重构头适配，因此它是当前 checkpoint 内的直接基础预测参照。

loss_batch_means 同时保存 raw/*、weighted/* 和 total。weighted 项之和应与 total 一致。原始未启用损失即使非零，也不代表参与训练，请看 weighted。损失统计是按 batch 样本数加权的 batch 均值；由于 mask/ON 分母不同，不应解释成整个数据集按位置重算的单一损失。不同实验使用不同损失时，不能直接按总 loss 排名。

## 单因素对照（仅训练/验证）

```bash
CUDA_VISIBLE_DEVICES=0 bash training/run_ofat_gpu.sh
```

依次运行源域三个种子和三种预测方法，共四组：

| 组 | 相对 control 的唯一训练因素变化 |
|---|---|
| control | 当前完整协议，附加诊断 |
| balanced_power | 主预测功率损失启用逐电器 ON/OFF 独立平衡，ON/OFF 各占 0.5；缺失状态按已有实现处理 |
| no_amp_on | 门控前 ON 幅值辅助损失权重从 0.05 设为 0 |
| old_residual_gate | PCSA 残差门控 logit 从 0 设回 -3；两种 baseline 不受此参数影响，可作为重复性参照 |

前三组使用对应的共同预测损失设置，历史 NILM 预训练目标保持不变。一次仅改变一项，不叠加；不改变早停准则和轮数。本轮暂不自动加入解冻、延长训练或新的选择指标。

所有组强制 --validation-only，并使用独立目录 outputs/diagnostic_ofat_v1/<组名>。脚本只运行源域 source，不启动迁移或测试。也可单独运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.run --phase source --validation-only --variant balanced_power --output outputs/diagnostic_ofat_v1/balanced_power --device cuda --seeds 7 42 123
```

不要只比较最佳一次或只看总体 MAE；比较三种子验证 MAE、ON/OFF 误差、残差增益及开启状态覆盖。确认某因素在验证集上的作用后，再事先确定下一步迁移对照。读取旧 checkpoint 的诊断无法补回过去每轮未保存的损失，但能补充最佳模型的当前诊断。
