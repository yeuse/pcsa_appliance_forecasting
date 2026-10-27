# 固定预测器的历史输入对照

上传新增的 history_input_ablation.py 到服务器 training/。依赖已经上传过的 common.py、run.py、diagnostics.py 及 src；本次不修改这些训练文件，不需要再上传权重或数据。

服务器项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.history_input_ablation --source-run outputs/diagnostic_ofat_v1/control --output outputs/history_input_ablation_control --device cuda --seeds 7 42 123
```

三路输入分别为现有 PCSA checkpoint 的 future_tcn_input_power/p_on、NILM checkpoint 的 past_power/p_on、真实历史功率/状态。PCSA 与 NILM 提取器只读且冻结。每个种子三路预测器均为相同的独立电器 TCN（hidden=64，3层），从完全相同的初始参数重新训练；它们不是同一训练后预测器的直接换输入。只向预测器传入历史功率、历史状态与已知未来时间特征，禁止退回读取真实历史标签。

复用 source-run 的共同预测损失、AdamW、批量大小、早停与验证选择规则。最大轮数为原 two_stage_forecast 的 160 轮；初始候选参与选择，patience=30，最少20轮。相同种子使用相同数据打乱顺序，实际轮数可因早停不同而不同。每个种子依次跑3路，共9个训练阶段。

为确保真实历史可用，所有输入源共同限定到过去120分钟的四电器标签全部可用的窗口；原有未来标签/净电网规则不变。window_audit.json 记录保留窗口和完整历史/未来标签预算。该筛选可能使结果不能直接与之前表格横向比较，请比较本实验内部三路。预训练源模型的历史标签与预算不被消除，来源checkpoint哈希保存在protocol.json；训练阶段耗时不包含源模型预训练成本。

只评估训练集和验证集，不评估测试集。真实历史是具有额外信息的诊断参照，不是可部署输入，也不保证有限样本训练一定取得最小误差。

输出：
- comparison.csv / comparison.json：每种子每条路径的预测MAE和ON/OFF误差。
- seed_7/<路径>/train_history_quality.json 与 val_history_quality.json：历史重构本身的逐电器误差和条件状态概率。
- selected_train.json / selected_validation.json：最优预测器的诊断。
- history.json / summary.json / best.pt：训练轨迹、选择和权重。
- input_provenance.json：相同预测器初始权重哈希、实际输入哈希。

路径名为 pcsa_history、nilm_history、oracle_history。比较相同种子下的差值，再汇总三个种子。

解释边界：若PCSA历史喂给相同预测器仍明显落后，支持历史信息质量/可用性是瓶颈；若差距很小而完整PCSA较差，则更值得检查直接预测分支、融合与选择规则。这些是控制对照的证据，不是对总误差的严格因果分解。三路都差而oracle也差，说明历史质量不足以解释全部问题，需要检查预测任务、表示和训练。没有必要为追求第一反复调整测试集结果。

完成后只打包结果JSON/CSV，不含权重：

```bash
cd /home/dell/work_ywj/pisa_framework_v1/outputs
find history_input_ablation_control -type f \( -name '*.json' -o -name '*.csv' \) -print0 | tar --null -czf history_input_ablation_results.tar.gz --files-from=-
```

恢复仅跳过带完整summary.json的已完成阶段；中断中的阶段会重训，不是逐轮精确续训。新文件加入代码清单后，旧训练目录不能直接用新版入口续训，旧checkpoint仍可由本工具只读使用。
