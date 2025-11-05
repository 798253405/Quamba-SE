"""
Percentile Logger - 记录percentile和reorder对激活范围的影响
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

class PercentileLogger:
    """记录percentile实验的详细统计信息"""

    def __init__(self, log_file="percentile_experiments.jsonl", save_activations=False):
        self.log_file = Path(log_file)
        self.save_activations = save_activations
        self.activation_samples = {}  # 用于保存激活值样本

        self.current_experiment = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
            "config": {},
            "activation_stats": {},
            "reorder_summary": {
                "enabled": False,
                "avg_range_reduction": None,
                "total_layers": 0
            },
            "results": {},
            "activation_data_file": None  # 如果保存了激活值，记录文件路径
        }

    def log_config(self, args):
        """记录实验配置"""
        self.current_experiment["config"] = {
            "model": args.model,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "percentile_alpha": args.percentile_alpha,
            "group_heads": args.group_heads,
            "apply_gptq": args.apply_gptq,
            "quantize_embedding": args.quantize_embedding,
            "quantize_lm_head": args.quantize_lm_head,
            "calib_data_num": args.calib_data_num,
            "calib_seqlen": args.calib_seqlen
        }

    def log_activation_stats(self, layer_name, stats, activation_values=None):
        """
        记录激活统计信息

        Args:
            layer_name: 层名称
            stats: 包含以下字段的字典
                - before_percentile: {"min", "max", "range"}
                - after_percentile: {"min", "max", "range"}
                - percentile_alpha: float
                - clipped_ratio: float (被裁剪的比例)
            activation_values: (可选) 实际的激活值，用于复现
        """
        self.current_experiment["activation_stats"][layer_name] = stats

        # 如果需要保存激活值
        if self.save_activations and activation_values is not None:
            # 只保存前100个样本，避免文件过大
            if isinstance(activation_values, np.ndarray):
                self.activation_samples[layer_name] = activation_values.flatten()[:10000].tolist()
            else:
                # PyTorch tensor
                self.activation_samples[layer_name] = activation_values.flatten()[:10000].cpu().numpy().tolist()

    def log_reorder_impact(self, layer_name, before_reorder, after_reorder):
        """
        记录reorder对激活范围的影响

        Args:
            layer_name: 层名称
            before_reorder: {"min", "max", "range"}
            after_reorder: {"min", "max", "range"}
        """
        if layer_name in self.current_experiment["activation_stats"]:
            self.current_experiment["activation_stats"][layer_name]["before_reorder"] = before_reorder
            self.current_experiment["activation_stats"][layer_name]["after_reorder"] = after_reorder
            self.current_experiment["activation_stats"][layer_name]["range_reduction_pct"] = (
                (before_reorder["range"] - after_reorder["range"]) / before_reorder["range"] * 100
            )

        self.current_experiment["reorder_summary"]["enabled"] = True

    def compute_reorder_summary(self):
        """计算reorder的总体效果"""
        if not self.current_experiment["reorder_summary"]["enabled"]:
            return

        reductions = []
        total_layers = 0

        for layer_name, stats in self.current_experiment["activation_stats"].items():
            if "range_reduction_pct" in stats:
                reductions.append(stats["range_reduction_pct"])
                total_layers += 1

        if reductions:
            self.current_experiment["reorder_summary"]["avg_range_reduction"] = sum(reductions) / len(reductions)
            self.current_experiment["reorder_summary"]["total_layers"] = total_layers

    def log_results(self, accuracy, perplexity):
        """记录最终结果"""
        self.current_experiment["results"] = {
            "accuracy": float(accuracy),
            "perplexity": float(perplexity)
        }

    def save(self):
        """保存到JSONL文件（追加模式）"""
        self.compute_reorder_summary()

        # 创建目录
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果有激活值样本，保存到单独的.npz文件
        if self.activation_samples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.current_experiment["config"]["model"].split('/')[-1]
            pa = self.current_experiment["config"].get("percentile_alpha", "default")
            pa_str = f"pa{pa}" if pa != "default" else "default"

            activation_file = self.log_file.parent / f"activations_{model_name}_{pa_str}_{timestamp}.npz"
            np.savez_compressed(activation_file, **self.activation_samples)
            self.current_experiment["activation_data_file"] = str(activation_file)
            logging.info(f"✅ 激活值已保存到: {activation_file}")

        # 追加写入JSON日志
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(self.current_experiment, indent=None) + "\n")

        logging.info(f"✅ Percentile实验日志已保存到: {self.log_file}")

    def print_summary(self):
        """打印当前实验的摘要"""
        print("\n" + "="*80)
        print("📊 Percentile实验摘要")
        print("="*80)

        # 配置信息
        config = self.current_experiment["config"]
        print(f"\n🔧 配置:")
        print(f"  模型: {config['model']}")
        print(f"  量化: W{config['w_bits']}A{config['a_bits']}")
        print(f"  Percentile Alpha: {config['percentile_alpha']}")
        print(f"  Group Heads: {config['group_heads']}")

        # 激活统计摘要
        if self.current_experiment["activation_stats"]:
            print(f"\n📈 激活统计 (共{len(self.current_experiment['activation_stats'])}层):")

            # 计算平均值
            avg_range_before = 0
            avg_range_after = 0
            avg_clip_ratio = 0
            count = 0

            for layer_name, stats in list(self.current_experiment["activation_stats"].items())[:3]:
                if "before_percentile" in stats:
                    print(f"\n  {layer_name}:")
                    print(f"    Percentile裁剪前: [{stats['before_percentile']['min']:.2f}, {stats['before_percentile']['max']:.2f}] "
                          f"范围={stats['before_percentile']['range']:.2f}")
                    print(f"    Percentile裁剪后: [{stats['after_percentile']['min']:.2f}, {stats['after_percentile']['max']:.2f}] "
                          f"范围={stats['after_percentile']['range']:.2f}")

                    if "clipped_ratio" in stats:
                        print(f"    被裁剪比例: {stats['clipped_ratio']*100:.4f}%")

                    avg_range_before += stats['before_percentile']['range']
                    avg_range_after += stats['after_percentile']['range']
                    if "clipped_ratio" in stats:
                        avg_clip_ratio += stats['clipped_ratio']
                    count += 1

            if count > 0:
                print(f"\n  前3层平均:")
                print(f"    裁剪前平均范围: {avg_range_before/count:.2f}")
                print(f"    裁剪后平均范围: {avg_range_after/count:.2f}")
                print(f"    范围缩小: {(avg_range_before-avg_range_after)/avg_range_before*100:.2f}%")
                if avg_clip_ratio > 0:
                    print(f"    平均裁剪比例: {avg_clip_ratio/count*100:.4f}%")

        # Reorder效果
        reorder = self.current_experiment["reorder_summary"]
        if reorder["enabled"]:
            print(f"\n🔄 Reorder效果:")
            print(f"  影响层数: {reorder['total_layers']}")
            if reorder["avg_range_reduction"] is not None:
                print(f"  平均范围缩小: {reorder['avg_range_reduction']:.2f}%")

        # 最终结果
        if self.current_experiment["results"]:
            results = self.current_experiment["results"]
            print(f"\n🎯 最终结果:")
            print(f"  Accuracy: {results['accuracy']*100:.2f}%")
            print(f"  Perplexity: {results['perplexity']:.3f}")

        print("\n" + "="*80 + "\n")


# 全局单例
_global_logger = None

def get_percentile_logger(log_file="percentileRangeResults/experiments.jsonl", save_activations=False):
    """获取全局percentile logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PercentileLogger(log_file, save_activations=save_activations)
    return _global_logger

def reset_percentile_logger(log_file="percentileRangeResults/experiments.jsonl", save_activations=False):
    """重置全局logger（用于新实验）"""
    global _global_logger
    _global_logger = PercentileLogger(log_file, save_activations=save_activations)
    return _global_logger
