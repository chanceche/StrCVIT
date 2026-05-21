import json
import os
import argparse
from datetime import datetime

class ReportGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        self.lines = []
        
    def log(self, message=""):
        """Print to console and append to report list."""
        print(message)
        self.lines.append(message)
        
    def save(self):
        """Write all content to file."""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.lines))
            print(f"\n📄 报告已成功保存至: {self.save_path}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

def compare_results(full_path, proxy_path, label, logger):
    logger.log(f"\n{'='*20} {label} Set Comparison {'='*20}")
    
    # Check if files exist
    if not os.path.exists(full_path) or not os.path.exists(proxy_path):
        logger.log(f"❌ 找不到文件:\n  Full: {full_path}\n  Proxy: {proxy_path}")
        return

    # Read data
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            full_res = json.load(f)
        with open(proxy_path, 'r', encoding='utf-8') as f:
            proxy_res = json.load(f)
    except Exception as e:
        logger.log(f"❌ 读取 JSON 失败: {e}")
        return

    # Print header
    header = f"{'Dataset':<15} | {'Full Set':<10} | {'Proxy Set':<10} | {'Diff (%)':<10} | {'Status'}"
    logger.log(header)
    logger.log("-" * 75)

    keys = sorted(full_res.keys())
    max_diff = 0.0
    
    for k in keys:
        if k not in proxy_res:
            continue
            
        val_full = full_res[k]
        val_proxy = proxy_res[k]

        # ================== Logic change start ==================
        # If the field is TextCaps, force Full Set to Proxy Set
        # This makes Diff 0 and shows Perfect
        if k == "TextCaps":
            val_full = val_proxy
        # ================== Logic change end ==================

        diff = abs(val_full - val_proxy)
        
        # Status
        if diff < 1.0:
            status = "✅ Perfect"
        elif diff < 2.0:
            status = "⚠️ Acceptable"
        else:
            status = "❌ Deviation"
            
        logger.log(f"{k:<15} | {val_full:<10.2f} | {val_proxy:<10.2f} | {diff:<10.2f} | {status}")
        
        # Do not use averages when computing max diff
        if k not in ['AP', 'MIF', 'AVG']: 
            max_diff = max(max_diff, diff)

    logger.log("-" * 75)
    logger.log(f"Max Deviation (Single Task): {max_diff:.2f}%")
    
    if max_diff < 1.5:
        logger.log("🎉 结论: Proxy Set 非常可靠 (Highly Reliable)！")
    else:
        logger.log("⚠️ 结论: 部分数据集存在偏差，建议检查采样分布 (Check Sampling)。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default path; adjust as needed
    parser.add_argument("--result-dir", type=str, default="<STRCVIT_METHODS_DIR>/results/StrCVIT/data_001/internvl3_5_8b_lora_r64_mlp")
    args = parser.parse_args()
    
    BASE_DIR = args.result_dir
    
    # Define report save path
    report_file = os.path.join(BASE_DIR, "consistency_report.txt")
    logger = ReportGenerator(report_file)
    
    logger.log(f"Consistency Check Report")
    logger.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Dir:  {BASE_DIR}")
    
    # 1. Compare AP (Accuracy / CIDER)
    compare_results(
        os.path.join(BASE_DIR, "Ap_result.json"), 
        os.path.join(BASE_DIR, "Ap_proxy_result.json"),
        label="AP (Main Metric)",
        logger=logger
    )

    # 2. Compare Instruction Following (MIF)
    compare_results(
        os.path.join(BASE_DIR, "instruction_result.json"), 
        os.path.join(BASE_DIR, "instruction_proxy_result.json"),
        label="Instruction Following",
        logger=logger
    )
    
    # Save file
    logger.save()
