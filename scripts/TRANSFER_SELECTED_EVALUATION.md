# 已选迁移模型：统一只评估

## v2：事件口径修正与验证集融合审计

新增依赖 `scripts/transfer_event_audit.py`，同步时必须一起上传。
模型权重、原功率指标、阈值及后处理不变。旧 `event*` 字段只作 legacy 复现，不能再当作公平事件对照。

- `event_comparison.csv` 为新事件对照，分开 start/stop，精确匹配和 ±2 分钟一对一匹配。
- `internal` 排除首点，而且在容差匹配前排除，防止首点预测匹配到内部真实事件。相邻两个目标位置均有效才计入事件。
- 同时报 raw 和 postprocessed_3on_2off，必须比较相同处理方式。事件按重叠窗口计数，不是去重后的家庭事件。
- `full_predicted_history` 优先使用实际 TCN 输入历史功率，其次 NILM/past_power；记录来源。无估计历史的模型不生成该项，不回退到真值。
- `full_true_history_DIAGNOSTIC_ONLY` 全部模型用真实历史末状态，独立标记为不可部署诊断。
- 全窗口首点假定数据集提供的 y_prev_state 有效；未来位置按 target_mask 过滤。
- 每个 PISA JSON 的 `results.val.fusion_audit` 记录 all/ON/OFF 的有效数、真实均值、各路输出的均值/绝对均值/P05/P50/P95，以及最终功率、基础功率、幅值 MAE。测试集不记录融合诊断。
- 原始残差及门控后 raw/logit 残差不是 kW；final_minus_base_power_kW 才是最终与基础路径的输出差，不可解读为独立因果贡献。
- 缺失字段记录 missing_fields，不伪造零值；无 ON 样本返回 null。

建议先执行 8565、10%：

```bash
python -m unittest discover -s tests -p 'test_transfer_event_audit.py' -v
python -m unittest discover -s tests -p 'test_transfer_selected_evaluation.py' -v
bash scripts/run_transfer_selected_evaluation.sh --homes 8565 --fractions 0.1 --smoke_windows 64 --batch_size 64 --num_workers 0
bash scripts/run_transfer_selected_evaluation.sh --homes 8565 --fractions 0.1
```

入口：`scripts/evaluate_transfer_selected.py`；启动：`scripts/run_transfer_selected_evaluation.sh`。
从仓库根目录执行，使用已有环境，不重新训练。

## 范围与保护

- 默认 3039、8386、8565，比例 0.010、0.050、0.100。
- PISA 两阶段 + original 的两种基线 + joint 的两种基线，共 45 个已选模型。
- PISA 使用 summary 指定的 selected.pt；基线按 selected_model 使用微调 best.pt 或源 checkpoint。
- source_zero_shot 保留源功率上限；其余使用保存的 support-only 上限，绝不重新拟合。
- 沿用原数据构建、源家庭归一化、顺序加载、状态阈值、事件容差 2 分钟、桶宽 5、最短 ON/OFF 3/2 的评估实现。
- 每个模型先重算验证集 MAE，误差超过 5e-5 kW 就停止，不评估该模型测试集。此前通过的模型结果保留。不要为绕过失败随意调大容差。
- 不改变阈值、优化器、模型选择或原实验文件。输出目录必须不存在。
- 不包括未来真实状态或真实历史替换；这些属于独立验证集诊断。
- 不能据此宣称训练预算相等；original/joint/PISA 分开报告，不按家庭选对自己有利的基线。

## 服务器

```bash
cd /path/to/pisa
python -m unittest discover -s tests -p 'test_transfer_selected_evaluation.py' -v
bash scripts/run_transfer_selected_evaluation.sh --preflight_only
bash scripts/run_transfer_selected_evaluation.sh --homes 3039 --fractions 0.1 --smoke_windows 64 --batch_size 64 --num_workers 0
bash scripts/run_transfer_selected_evaluation.sh
```

预检检查配置、协议、数据和已选 checkpoint 路径，不加载权重，不创建结果目录。
smoke 仅运行验证集前 64 个窗口，不访问测试集评估、不做完整 MAE 复核，不是正式结果。
正式执行按训练配置保留 AMP 设置；修改批大小可能产生微小数值差异。

## 输出

输出目录：`outputs/transfer_selected_eval_时间戳`。

- `comparison.csv`：家庭、方法组、方法、比例、split、宏平均 MAE、选择类型和 epoch。
- `metrics_long.csv`：全部原评估指标，包含按设备功率、状态和事件指标。
- 每模型 JSON：完整指标、原选择记录、配置、数据/checkpoint SHA256、窗口数、阈值和执行信息。
- `COMPLETE.json`：全部完成才生成；`smoke_only=true` 表示不能用于论文的试运行。

本地单元测试包含 mock 数据/评估流程和真实小型 torch 权重加载，不替代服务器真实模型评估。
脚本依赖仓库已有 evaluate_transfer_history_and_forecast.py、run_baseline_cross_home_transfer.py、run_pisa_cross_home_transfer.py 及 src。
源实验的绝对路径必须仍然有效；路径无效时直接失败，不模糊搜索替代 checkpoint。
