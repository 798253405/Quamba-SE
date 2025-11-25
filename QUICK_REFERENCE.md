# 快速参考 - 层输出对比工具

## 🚀 三步快速开始

```bash
# 1. 保存FP32参考（只需运行一次）
./save_all_modes.sh fp_only

# 2. 保存目标mode
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode 2-1 --quantize

# 3. 对比
./comparewithfp 2-1
```

---

## 📋 常用命令速查

### 保存输出

| 命令 | 说明 |
|------|------|
| `./save_all_modes.sh` | 保存所有modes (FP32 + 7个量化modes) |
| `./save_all_modes.sh fp_only` | 只保存FP32参考 |
| `./save_all_modes.sh essential` | 保存关键modes (FP32, 0, 2-1, 2-2, 2-4) |
| `./save_all_modes.sh 0 2-1 2-4` | 保存指定modes |

### 对比输出

| 命令 | 说明 |
|------|------|
| `./comparewithfp <mode>` | 对比单个mode与FP32 |
| `./compare_all_modes.sh` | 批量对比所有modes，生成汇总表 |

### Python直接调用

```bash
# 保存FP32
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir <path> --mode fp32

# 保存量化mode
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir <path> --mode 2-1 --quantize

# 对比
python3 compare_with_fp.py 2-1 --reference fp32
```

---

## 🎯 核心指标解读

| 指标 | 优秀 | 良好 | 一般 | 说明 |
|------|------|------|------|------|
| **MSE** | < 1e-4 | 1e-4 ~ 1e-3 | 1e-3 ~ 1e-2 | 均方误差，越小越好 |
| **Correlation** | > 0.9999 | 0.999 ~ 0.9999 | 0.99 ~ 0.999 | 相关系数，越接近1越好 |
| **Relative MAE** | < 0.1% | 0.1% ~ 1% | 1% ~ 5% | 相对误差百分比 |

---

## 📁 输出文件结构

```
layer_outputs/
├── mode_fp32_layer_0.npy           # FP32第1层输出
├── mode_fp32_layer_23.npy          # FP32最后一层输出
├── mode_fp32_stats.json            # FP32统计
├── mode_0_layer_0.npy              # Mode 0第1层
├── mode_0_layer_23.npy             # Mode 0最后一层
└── mode_0_stats.json               # Mode 0统计

comparisons/                         # 对比结果
├── mode_0_vs_fp32.json
├── mode_2-1_vs_fp32.json
└── ...
```

---

## 🔧 常见问题

**Q: 必须先保存FP32吗？**
A: 是的，FP32作为精度参考，必须先保存。

**Q: 如何加速批量保存？**
A: 减少 `--calib_data_num`（默认10），或使用 `essential` 只保存关键modes。

**Q: 可以用fp16做参考吗？**
A: 可以，使用 `--reference fp16`。

**Q: 为什么只保存2层？**
A: 第1层反映初始影响，最后一层反映累积误差。可修改代码保存更多层。

---

## 📞 更多信息

- 详细文档: `LAYER_COMPARISON_README.md`
- 使用总结: `LAYER_COMPARISON_SUMMARY.md`
- Mode说明: `SSM_MODE_GUIDE.md`

---

**版本**: 1.0 | **更新**: 2025-01-10
