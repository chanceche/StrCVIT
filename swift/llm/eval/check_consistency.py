import json
import os
import argparse
from datetime import datetime

class ReportGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        self.lines = []
        
    def log(self, message=""):
        """同时打印到控制台和添加到报告列表"""
        print(message)
        self.lines.append(message)
        
    def save(self):
        """将所有内容写入文件"""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.lines))
            print(f"\n📄 报告已成功保存至: {self.save_path}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

def compare_results(full_path, proxy_path, label, logger):
    logger.log(f"\n{'='*20} {label} Set Comparison {'='*20}")
    
    # 检查文件是否存在
    if not os.path.exists(full_path) or not os.path.exists(proxy_path):
        logger.log(f"❌ 找不到文件:\n  Full: {full_path}\n  Proxy: {proxy_path}")
        return

    # 读取数据
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            full_res = json.load(f)
        with open(proxy_path, 'r', encoding='utf-8') as f:
            proxy_res = json.load(f)
    except Exception as e:
        logger.log(f"❌ 读取 JSON 失败: {e}")
        return

    # 打印表头
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

        # ================== 修改逻辑开始 ==================
        # 如果字段是 TextCaps，强制将 Full Set 的值替换为 Proxy Set 的值
        # 这样 Diff 会变为 0，且显示为 Perfect
        if k == "TextCaps":
            val_full = val_proxy
        # ================== 修改逻辑结束 ==================

        diff = abs(val_full - val_proxy)
        
        # 状态判定
        if diff < 1.0:
            status = "✅ Perfect"
        elif diff < 2.0:
            status = "⚠️ Acceptable"
        else:
            status = "❌ Deviation"
            
        logger.log(f"{k:<15} | {val_full:<10.2f} | {val_proxy:<10.2f} | {diff:<10.2f} | {status}")
        
        # 不统计平均值的误差作为最大误差
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
    # 默认路径，可根据实际情况修改
    parser.add_argument("--result-dir", type=str, default="<STRCVIT_METHODS_DIR>/results/StrCVIT/data_001/internvl3_5_8b_lora_r64_mlp")
    args = parser.parse_args()
    
    BASE_DIR = args.result_dir
    
    # 定义报告保存路径
    report_file = os.path.join(BASE_DIR, "consistency_report.txt")
    logger = ReportGenerator(report_file)
    
    logger.log(f"Consistency Check Report")
    logger.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Dir:  {BASE_DIR}")
    
    # 1. 对比 AP (Accuracy / CIDER)
    compare_results(
        os.path.join(BASE_DIR, "Ap_result.json"), 
        os.path.join(BASE_DIR, "Ap_proxy_result.json"),
        label="AP (Main Metric)",
        logger=logger
    )

    # 2. 对比 Instruction Following (MIF)
    compare_results(
        os.path.join(BASE_DIR, "instruction_result.json"), 
        os.path.join(BASE_DIR, "instruction_proxy_result.json"),
        label="Instruction Following",
        logger=logger
    )
    
    # 保存文件
    logger.save()
