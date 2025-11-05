# Percentile实验日志系统使用说明

## 📋 功能概述

这个系统自动记录每次量化实验中percentile裁剪和reorder对激活范围的影响，包括：

1. **实验元数据**：时间、命令行、配置参数
2. **Percentile裁剪前后**：每层的min/max/range变化
3. **Reorder效果**：重排序对范围的改善
4. **最终结果**：accuracy和perplexity

所有实验记录到**同一个文件**：`logs/percentile_experiments.jsonl`

---

## 🚀 快速开始

### 1. 运行实验（自动记录）

```bash
# 正常运行量化实验，日志会自动记录
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-370m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --percentile_alpha 1.0
```

**实验结束后会自动：**
- ✅ 记录配置和命令
- ✅ 收集前3层的激活统计
- ✅ 记录最终accuracy/perplexity
- ✅ 追加到 `logs/percentile_experiments.jsonl`
- ✅ 打印摘要到终端

---

### 2. 查看日志

#### 查看所有实验
```bash
python3 view_percentile_logs.py
```

#### 查看最近5次实验
```bash
python3 view_percentile_logs.py --last 5
```

#### 筛选特定模型
```bash
python3 view_percentile_logs.py --filter "mamba-370m"
```

#### 对比两个实验
```bash
# 对比第0个和第1个实验
python3 view_percentile_logs.py --compare 0,1
```

---

## 📊 日志格式说明

### JSON结构
```json
{
  "timestamp": "2025-11-05 15:30:00",
  "command": "python3 main.py ...",
  "config": {
    "model": "mamba-370m",
    "w_bits": 8,
    "a_bits": 8,
    "percentile_alpha": 1.0,
    "group_heads": false,
    "apply_gptq": true
  },
  "activation_stats": {
    "layer_0.x_proj:input": {
      "before_percentile": {
        "min": -127.35,
        "max": 156.82,
        "range": 284.17
      },
      "after_percentile": {
        "min": -127.35,
        "max": 156.82,
        "range": 284.17
      },
      "percentile_alpha": 1.0,
      "clipped_ratio": 0.0,
      "range_reduction": 0.0
    }
  },
  "reorder_summary": {
    "enabled": false,
    "avg_range_reduction": null,
    "total_layers": 0
  },
  "results": {
    "accuracy": 0.5205,
    "perplexity": 9.621
  }
}
```

---

## 🔍 关键指标解释

### 1. Percentile裁剪效果

**before_percentile**：真实激活的min/max
```
min: -127.35
max: 156.82
range: 284.17  (max - min)
```

**after_percentile**：percentile裁剪后的min/max
```
min: -120.50
max: 145.30
range: 265.80  (缩小了6.4%)
```

**range_reduction**：范围缩小比例
```
range_reduction = (284.17 - 265.80) / 284.17 = 0.064 (6.4%)
```

---

### 2. Percentile_alpha影响

| alpha值 | 含义 | 裁剪比例 |
|---------|------|---------|
| 0.9995 | 99.95%百分位 | 裁剪top 0.05% |
| 0.99999 | 99.999%百分位 | 裁剪top 0.001% |
| 1.0 | 100%百分位 | 不裁剪 (0%) |

**clipped_ratio = 1.0 - percentile_alpha**

---

### 3. Reorder效果（仅Mamba2）

**before_reorder**：聚类前的激活范围
**after_reorder**：聚类后的激活范围

```json
"reorder_summary": {
  "enabled": true,
  "avg_range_reduction": 15.6,  // 平均缩小15.6%
  "total_layers": 32
}
```

---

## 📈 实际使用案例

### 案例1：对比pa=1.0和默认值

```bash
# 运行两次实验
python3 main.py mamba-370m --quantize --w_bits 8 --a_bits 8 ...
# (自动记录为实验0)

python3 main.py mamba-370m --quantize --w_bits 8 --a_bits 8 --percentile_alpha 1.0 ...
# (自动记录为实验1)

# 对比结果
python3 view_percentile_logs.py --compare 0,1
```

**输出示例**：
```
📊 对比: 实验#0 vs 实验#1
================================================================================

配置对比:
  项目                      实验#0                    实验#1
  ---------------------------------------------------------------------------
  模型                      mamba-370m               mamba-370m
  量化                      W8A8                     W8A8
  Percentile Alpha          default                  1.0 ⚠️
  Group Heads               False                    False
  GPTQ                      True                     True

🎯 结果对比:
  Accuracy: 49.39% vs 52.05% (差异: +2.66%)
  Perplexity: 10.693 vs 9.621 (差异: -1.072)

📊 激活范围对比（第一层）:

  裁剪前范围:
    实验#0: 284.17
    实验#1: 284.17

  裁剪后范围:
    实验#0: 265.80  (裁剪了6.4%)
    实验#1: 284.17  (不裁剪)
```

---

### 案例2：分析Reorder效果

```bash
# Mamba2实验（有reorder）
python3 main.py state-spaces/mamba2-2.7b \
  --quantize --group_heads \
  --w_bits 8 --a_bits 8 ...

# 查看最近一次实验
python3 view_percentile_logs.py --last 1
```

**输出示例**：
```
📊 激活统计 (前3层):

  layer_0.x_conv_out:input_reordered:
    裁剪前: [-89.32, 102.45] 范围=191.77
    裁剪后: [-89.32, 102.45] 范围=191.77
    裁剪比例: 0.0001%

🔄 Reorder效果:
  影响层数: 32
  平均范围缩小: 15.6%
```

---

### 案例3：筛选特定模型的所有实验

```bash
# 查看所有mamba-370m的实验
python3 view_percentile_logs.py --filter "mamba-370m"
```

---

## 🎯 如何回答你的问题

### Q1: pa=1.0的min/max是多少？
**查看日志**：
```bash
python3 view_percentile_logs.py --filter "percentile_alpha: 1.0"
```

查看 `activation_stats` 中的 `after_percentile` 字段。

---

### Q2: 默认percentile的min/max是多少？
**查看日志**：
```bash
python3 view_percentile_logs.py --filter "percentile_alpha: null"
```

或者直接打开 `logs/percentile_experiments.jsonl`，搜索没有设置 `percentile_alpha` 的实验。

---

### Q3: Reorder改善了多少？
**查看日志**：
```bash
python3 view_percentile_logs.py --filter "group_heads: true"
```

查看 `reorder_summary.avg_range_reduction` 字段。

**或者对比**：
```bash
# 有reorder vs 无reorder
python3 view_percentile_logs.py --compare 0,1
```

---

### Q4: 最终accuracy是多少？
**查看日志**：
```bash
python3 view_percentile_logs.py --last 5
```

查看 `results.accuracy` 字段。

---

## 🔧 自定义和扩展

### 只记录特定层
修改 `quamba/modelutils_mamba.py` 的收集代码：
```python
# 只记录前3层
for i in range(min(3, len(layers))):
    ...

# 改为记录所有层
for i in range(len(layers)):
    ...
```

### 添加更多统计信息
修改 `quamba/observer.py` 的 `get_stats()` 方法：
```python
def get_stats(self):
    return {
        "before_percentile": ...,
        "after_percentile": ...,
        # 添加新字段
        "median": torch.median(self.w_max).item(),
        "std": torch.std(self.w_max).item(),
    }
```

---

## 📂 文件位置

```
Quamba/
├── logs/
│   └── percentile_experiments.jsonl  # 统一的实验日志
├── quamba/
│   ├── percentile_logger.py           # Logger实现
│   ├── observer.py                    # 修改：收集统计
│   └── modelutils_mamba.py            # 修改：调用logger
├── main.py                            # 修改：初始化logger
├── view_percentile_logs.py            # 查看工具
└── PERCENTILE_LOGGING.md              # 本文档
```

---

## 🚨 注意事项

1. **日志文件是追加模式**：每次实验都会追加到 `percentile_experiments.jsonl`，不会覆盖
2. **只记录前3层**：为了减少日志大小，默认只记录前3层的激活统计
3. **JSONL格式**：每行是一个独立的JSON对象，可以逐行解析
4. **异常处理**：如果记录失败，只会警告，不会中断实验

---

## 💡 最佳实践

### 实验命名规范
在命令中添加描述性参数：
```bash
python3 main.py mamba-370m \
  --quantize \
  --percentile_alpha 1.0 \
  --output_subdir "exp_pa1.0_no_gptq" \
  ...
```

### 定期备份日志
```bash
cp logs/percentile_experiments.jsonl logs/percentile_experiments_backup_$(date +%Y%m%d).jsonl
```

### 批量实验
```bash
# 运行多组对比实验
for pa in 0.9995 0.99999 1.0; do
  python3 main.py mamba-370m \
    --quantize --percentile_alpha $pa ...
done

# 一次性对比
python3 view_percentile_logs.py --last 3
```

---

## 🎓 示例输出

### 终端摘要输出
```
================================================================================
📊 Percentile实验摘要
================================================================================

🔧 配置:
  模型: mamba-370m
  量化: W8A8
  Percentile Alpha: 1.0
  Group Heads: False

📈 激活统计 (共64层):

  layer_0.x_proj:input:
    Percentile裁剪前: [-127.35, 156.82] 范围=284.17
    Percentile裁剪后: [-127.35, 156.82] 范围=284.17
    被裁剪比例: 0.0000%

  前3层平均:
    裁剪前平均范围: 265.45
    裁剪后平均范围: 265.45
    范围缩小: 0.00%
    平均裁剪比例: 0.0000%

🎯 最终结果:
  Accuracy: 52.05%
  Perplexity: 9.621

================================================================================

✅ Percentile实验日志已保存到: logs/percentile_experiments.jsonl
```

---

**祝实验顺利！🎉**
