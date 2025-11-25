# Float Simulation 修改总结

## ✅ 完成的操作

### 1. 修改已应用到主目录

所有Float Simulation相关的修改已经应用到：
```
/home/yz/myprojects/2025/logquamba/Quamba/
```

### 2. temp-originalquamba 已回滚到官方版本

```
/home/yz/myprojects/2025/logquamba/Quamba/temp-originalquamba/
```
现在是干净的官方版本（除了pretrained_models和logs目录）

---

## 📁 主目录中的修改文件

### 核心代码修改 (quamba/)

```
quamba/qConvLayer.py         ✏️ 修改 - 添加float simulation + 日志记录
quamba/qLinearLayer.py       ✏️ 修改 - 处理FP32输入
quamba/qSelectiveScan.py     ✏️ 修改 - 处理FP32输入
```

### 新增文件 (根目录)

```
test_check_float_sim.py      ✨ 测试脚本（生成日志文件）
test_float_sim.py            ✨ 简单测试脚本（验证一致性）
FLOAT_SIM_README.md          ✨ 完整文档
yzCheckFloatSim_FORMAT.md    ✨ 日志格式说明
FLOAT_SIM_CHANGES_SUMMARY.md ✨ 本文件
```

---

## 🔧 使用方法

### 在主目录运行测试

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba

# 运行测试（会生成日志文件）
python test_check_float_sim.py --quantize

# 查看生成的日志
ls -lh yzCheckFloatSim/
cat yzCheckFloatSim/int8_baseline.json
```

### 环境变量控制

```bash
# 启用float simulation
export FLOAT_SIM_ASIC=true

# 启用better scale
export FLOAT_SIM_BETTER_SCALE=true
export FLOAT_SIM_SCALE_FACTOR=2025

# 启用日志记录
export YZ_CHECK_FLOAT_SIM=true
```

---

## 📊 目录对比

### 主目录 (/home/yz/myprojects/2025/logquamba/Quamba/)

```
Quamba/
├── quamba/
│   ├── qConvLayer.py          ✏️ 有float sim修改
│   ├── qLinearLayer.py        ✏️ 有float sim修改
│   ├── qSelectiveScan.py      ✏️ 有float sim修改
│   └── ...
├── test_check_float_sim.py    ✨ 新增
├── test_float_sim.py          ✨ 新增
├── FLOAT_SIM_README.md        ✨ 新增
└── yzCheckFloatSim_FORMAT.md  ✨ 新增
```

### temp-originalquamba (官方版本)

```
temp-originalquamba/
├── quamba/
│   ├── qConvLayer.py          ✅ 官方原版（无修改）
│   ├── qLinearLayer.py        ✅ 官方原版（无修改）
│   ├── qSelectiveScan.py      ✅ 官方原版（无修改）
│   └── ...
├── pretrained_models/         (保留，不影响)
└── logs/                      (保留，不影响)
```

---

## 🎯 快速验证

### 验证主目录有修改

```bash
# 应该看到 import os, import json, _CONV1D_LAYER_COUNTER
head -15 /home/yz/myprojects/2025/logquamba/Quamba/quamba/qConvLayer.py
```

### 验证temp-originalquamba是官方版本

```bash
# 应该看不到 import os, import json
head -15 /home/yz/myprojects/2025/logquamba/Quamba/temp-originalquamba/quamba/qConvLayer.py
```

### 运行测试

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba
python test_check_float_sim.py --quantize --seq-len 32
```

预期生成：
- `yzCheckFloatSim/int8_baseline.json`
- `yzCheckFloatSim/floatsim_samescale.json`

---

## 📖 文档

- **完整说明**: `FLOAT_SIM_README.md`
- **日志格式**: `yzCheckFloatSim_FORMAT.md`
- **本总结**: `FLOAT_SIM_CHANGES_SUMMARY.md`

---

## ✅ 检查清单

- [x] 所有修改已复制到主目录
- [x] temp-originalquamba已回滚到官方版本
- [x] 测试脚本已复制
- [x] 文档已复制
- [x] 验证主目录文件有修改
- [x] 验证temp-originalquamba文件是原版

---

## 🔄 如果需要回滚主目录

如果需要将主目录也恢复到官方版本：

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba
git restore quamba/qConvLayer.py quamba/qLinearLayer.py quamba/qSelectiveScan.py
rm -f test_check_float_sim.py test_float_sim.py FLOAT_SIM_README.md yzCheckFloatSim_FORMAT.md
```

---

生成时间: 2025-11-07
