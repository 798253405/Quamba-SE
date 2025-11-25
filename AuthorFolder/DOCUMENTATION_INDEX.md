# Quamba量化研究文档索引

**最后更新**: 2025-11-06 (Session 6完成)

---

## 📚 核心文档

### 0. SESSION_HISTORY.md ⭐⭐⭐
**会话历史总结**（最新）
- **Session 6 (2025-11-06)**: 命令行配置与模型评估 + 关键Bug修复
  - 澄清W4A4实现状态（当前仅W4A8）
  - 配置Quamba1/2实验命令（130M和2.7B模型）
  - 自定义输出目录：`--output_subdir`参数
  - 解决HuggingFace路径验证错误（需要`./`前缀）
  - 6条完整命令清单（量化+评估）
  - **修复4个Quamba1加载Bug**（详见 `QUAMBA1_LOAD_BUGS_FIX.md`）
- **Session 5 (2025-11-06)**: W4A5-SE Dual-Scale方案与SSM量化原理
  - GEMM vs SSM计算差异（先计算再dequant vs 先dequant再计算）
  - SSM必须FP32的数学原理（exp函数、递归累积）
  - 量化收益来源（带宽>计算类型）
  - 方案作用位置：Conv1D输出量化
  - 完整SSM文件架构（4层树状图）
- Session 4: Activation Scale机制与Quamba优势分析
- Session 3: Scale实现深度分析
- Session 2: Percentile影响分析
- Session 1: Quamba1/2差异理解

---

## 📚 技术文档

### 0b. QUAMBA1_LOAD_BUGS_FIX.md ⭐⭐⭐ **[重要，不要删除]**
**Quamba1 模型加载Bug修复详解**
- **4个连续Bug的完整分析**（KeyError, Missing bias, Tuple vs Tensor, dtype不匹配）
- **根本原因**：保存config时的逻辑错误 + 量化与加载逻辑不一致
- **修复方案**：`qNorm.py` (2处) + `quamba_mixer_seq.py` (3处)
- **回档方案**：完整的代码恢复步骤 + Git命令
- **触发条件**：按照作者建议不加 `--quantize_lm_head`（Quamba1标准用法）
- **影响范围**：只影响 Quamba1，Quamba2 不受影响
- **验证方法**：正确的量化和评估命令

### 1. QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md ⭐⭐⭐
**完整的量化机制与改进指南**
- 量化实现概览（数据类型、两阶段流程）
- Scale计算与精度（FP32实现，持久性原理）
- Reorder与分组机制（4×4分组，runtime开销<1%）
- 正负号与对称量化（Signed INT8, zero_point=0）
- 替换可行性评估（什么能改，什么不能改）
- **改进Scale的6个实验方向**（含代码框架）

### 2. MAMBA_QUANTIZATION_LITERATURE_REVIEW.md ⭐⭐⭐
**Mamba量化方法文献综述**（基于arXiv论文）
- **6种方法对比**：Mamba-PTQ, Quamba, Quamba2, MambaQuant, QMamba, PTQ4VM
- **实验数据**：Quamba 1.72×加速/0.9%损失，Quamba2 3×加速/1.6%损失
- **技术路线**：Rotation vs Smoothing vs Clustering vs Percentile
- **Quamba真实优势**：首个完整方案 + 工程实用性 + 边缘设备实测
- **改进方向**：结合MambaQuant rotation、优化percentile策略

### 3. QUAMBA_ADVANTAGES_ANALYSIS.md ⭐⭐
**Quamba优势分析：纯INT8 vs 混合精度**（理论分析）
- **核心问题**：现有混合精度方案已存在，Quamba优势在哪？
- 5大优势：架构特定、硬件效率、Piecewise策略、系统优化、部署友好
- 深层原因：为什么Mamba可以用纯INT8？
- 局限性分析：架构限制、静态scale风险
- 应用场景对比：边缘设备 vs 云端推理

### 4. RELATED_WORK_OUTLIER_QUANTIZATION.md ⭐⭐
**量化中Outlier处理的相关工作综述**
- LLM.int8()、SqueezeLLM、AWQ等方法对比
- 混合精度 vs 纯INT8策略分析
- MSE与输入分布的关系讨论
- Quamba的独特性与改进空间
- 论文写作建议（Related Work结构）

### 4. ACTIVATION_SCALE_STATIC_ANALYSIS.md ⭐⭐
**Activation Scale 静态/动态分析**
- 核心结论：Activation scale是**完全静态的**
- 完整证据链：Calibration → Quantization → Runtime
- Static vs Dynamic对比（开销、精度、适用场景）
- 静态scale的工作原理（LayerNorm归一化、统计平稳性）
- 代码验证方法（打印scale值、检查模型文件）

### 5. REORDER_INDEX_OVERHEAD_ANALYSIS.md ⭐⭐
**Reorder Index开销分析（GPU）**
- **核心问题**：Quamba2的index查找有多少开销？
- **内存开销**：72 KB / 2.7 GB ≈ 0.003%（可忽略）
- **Runtime开销**：8-12 cycles / 100-200 cycles < 1%（可忽略）
- **实测验证**：Quamba2仍达到3× speedup
- **对比Rotation方法**：Index查找 << 矩阵乘法

### 5b. ASIC_INDEX_OVERHEAD_ANALYSIS.md ⭐⭐⭐
**ASIC硬件上的Index开销分析**
- **SRAM面积**：2.88 mm² (Quamba2) vs 0.01 mm² (Quamba1) = **+287×**
- **功耗**：1 nJ/lookup (Quamba2) vs 0.01 nJ (Quamba1) = **+100×**
- **延迟**：7.4 ns (顺序查找) 可能成为critical path
- **相对开销**：面积4.4% (vs整个加速器)，功耗50% (vs Conv1D)
- **结论**：在ASIC上开销**显著**，需要权衡或优化

### 6. OUTLIER_AWARE_SCALE_GUIDE.md ⭐
**Outlier-Aware Scale使用指南**
- 核心思路：让99%值占用[-120,120]，1%outlier占用剩余范围
- 完整实验流程（测试脚本+真实数据验证）
- 理论分析（信息熵、MSE优化目标）
- 集成到observer.py的代码示例
- **结论**：主要收益来自Percentile vs Max，而非120 vs 127

### 7. claudeYZ_验证原理讲解.md ⭐
**Percentile Alpha验证原理讲解**
- percentile_alpha如何影响量化
- 量化模型中存储了什么
- 验证方法对比（4种方法）
- **最新**: W4A5-SE Dual-Scale方案（2025-11-06）

---

## 📁 实验文档（用户创建）

### 中文实验记录
- `完整实验文档_汇总.md`: 24个实验的完整记录
- `正确实验说明_20个.md`: 修正后的20个实验
- `现有结果分析报告.md`: 实验结果分析
- `Quamba1_vs_Quamba2_正确版.md`: Quamba1/2差异对比
- `作者回复分析.md`: 原作者的回复分析
- `实验结果汇总表格.md`: 结果表格
- `命令对比说明.md`: 命令差异说明
- `实验脚本说明.md`: 脚本使用说明

### 实验脚本
- `run_all_complete_experiments.sh`: 完整24个实验
- `run_correct_experiments.sh`: 修正后20个实验
- `run_mamba2_8b_experiments.sh`: Mamba2-8B实验
- `compare_percentile_effects.sh`: Percentile对比实验（4个）

---

## 🧪 代码工具

### 分析与测试脚本
- `outlier_aware_scale.py`: Outlier-aware策略测试（4种方法对比+可视化）
- `view_percentile_logs.py`: 查看percentile日志
- `utils.py`: 工具函数（含percentile_alpha参数）
- `main.py`: 主程序（量化+评估）

---

## 📋 项目文档

- `README.md`: Quamba项目说明
- `CODE_OF_CONDUCT.md`: 行为准则

---

## 🗂️ 历史文档（已归档）

以下文档内容已整合到 `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md`：
- ~~CALIBRATION_INFO.md~~ → 见第2节
- ~~COMPARISON_SCRIPT_USAGE.md~~ → 见第8节
- ~~PERCENTILE_LOGGING.md~~ → 见第2节
- ~~percentile_impact_analysis.md~~ → 见第6节

---

## 🎯 快速查找

### 我想了解...
- **Quamba vs 其他Mamba PTQ方法** (文献综述): `MAMBA_QUANTIZATION_LITERATURE_REVIEW.md` ⭐⭐⭐
- **Quamba相比混合精度的优势** (理论): `QUAMBA_ADVANTAGES_ANALYSIS.md` ⭐⭐
- **Reorder index的开销 (GPU)**: `REORDER_INDEX_OVERHEAD_ANALYSIS.md` ⭐⭐
- **Reorder index的开销 (ASIC)**: `ASIC_INDEX_OVERHEAD_ANALYSIS.md` ⭐⭐⭐
- **量化原理**: `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` 第1节
- **Scale如何计算**: `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` 第2节
- **Activation scale是静态还是动态**: `ACTIVATION_SCALE_STATIC_ANALYSIS.md` ⭐
- **如何改进Scale**: `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` 第6节
- **什么能改什么不能改**: `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` 第5节
- **Outlier处理方法**: `RELATED_WORK_OUTLIER_QUANTIZATION.md`
- **Outlier-aware策略**: `OUTLIER_AWARE_SCALE_GUIDE.md`
- **MSE与输入的关系**: `RELATED_WORK_OUTLIER_QUANTIZATION.md` 第6节
- **为什么Mamba可以用纯INT8**: `QUAMBA_ADVANTAGES_ANALYSIS.md` 第5节
- **实验结果**: `现有结果分析报告.md`
- **会话历史**: `SESSION_HISTORY.md`

### 我想做...
- **测试Outlier-aware策略**: 运行 `python3 outlier_aware_scale.py`
- **运行实验**: 见 `实验脚本说明.md` 或各个 `.sh` 脚本
- **分析percentile影响**: 运行 `compare_percentile_effects.sh`
- **查看日志**: 运行 `python view_percentile_logs.py`
- **修改量化策略**: 见 `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` 第6节

---

**维护者**: Claude (Sonnet 4.5) + Yizhi Chen
**仓库**: https://github.com/798253405/Log-QuambaPrivate
