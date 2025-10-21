#!/usr/bin/env python3
"""
适配urban目录结构的UrbanVideo-Bench结果分析脚本
分析原始视频、雨雾版本、遮挡版本的评估结果
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict
import re

class UrbanResultAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.test_types = ['original', 'rain_fog', 'occlusion']
        self.results = {}
        self.summaries = {}
        
    def load_urban_results(self):
        """加载所有urban测试结果"""
        print("加载urban测试结果...")
        
        for test_type in self.test_types:
            # 加载详细结果（使用urban前缀）
            result_file = self.results_dir / f"urban_{test_type}_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    self.results[test_type] = json.load(f)
                print(f"✅ 加载 urban_{test_type} 结果: {len(self.results[test_type])} 个问题")
            else:
                print(f"⚠️  未找到 urban_{test_type} 结果文件")
                self.results[test_type] = []
            
            # 加载总结
            summary_file = self.results_dir / f"urban_{test_type}_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.summaries[test_type] = json.load(f)
            else:
                print(f"⚠️  未找到 urban_{test_type} 总结文件")
    
    def extract_answer(self, model_response):
        """从模型响应中提取答案"""
        if not model_response:
            return None
        
        # 寻找答案格式：<answer>A</answer> 或直接的A、B、C、D
        answer_match = re.search(r'<answer>([ABCDE])</answer>', model_response)
        if answer_match:
            return answer_match.group(1)
        
        # 寻找常见的答案格式
        patterns = [
            r'答案是?\s*([ABCDE])',
            r'选择\s*([ABCDE])',
            r'选项\s*([ABCDE])',
            r'答案：\s*([ABCDE])',
            r'答案:\s*([ABCDE])',
            r'([ABCDE])(?=\s*[\.。])',
            r'选择答案\s*([ABCDE])',
            r'正确答案是\s*([ABCDE])',
            r'选项是\s*([ABCDE])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 最后尝试找单个字母
        letter_match = re.search(r'[ABCDE]', model_response)
        if letter_match:
            return letter_match.group(0).upper()
        
        return None
    
    def calculate_urban_accuracy(self):
        """计算详细的准确率统计"""
        print("计算urban详细准确率统计...")
        
        detailed_stats = {}
        
        for test_type in self.test_types:
            if not self.results[test_type]:
                continue
                
            stats = {
                'total': 0,
                'correct': 0,
                'no_answer_extracted': 0,
                'by_category': defaultdict(lambda: {'total': 0, 'correct': 0}),
                'by_video': defaultdict(lambda: {'total': 0, 'correct': 0}),
                'answer_distribution': defaultdict(int),
                'ground_truth_distribution': defaultdict(int),
                'correct_answers': [],
                'incorrect_answers': []
            }
            
            for result in self.results[test_type]:
                ground_truth = result.get('answer', '').strip().upper()
                model_response = result.get('content', '')
                predicted = self.extract_answer(model_response)
                
                stats['total'] += 1
                stats['ground_truth_distribution'][ground_truth] += 1
                
                if predicted is None:
                    stats['no_answer_extracted'] += 1
                    stats['incorrect_answers'].append({
                        'question_id': result.get('Question_id'),
                        'ground_truth': ground_truth,
                        'predicted': 'NO_ANSWER',
                        'question': result.get('question', '')[:100] + '...',
                        'response': model_response[:200] + '...'
                    })
                    continue
                
                stats['answer_distribution'][predicted] += 1
                
                is_correct = (predicted == ground_truth)
                if is_correct:
                    stats['correct'] += 1
                    stats['correct_answers'].append(result)
                else:
                    stats['incorrect_answers'].append({
                        'question_id': result.get('Question_id'),
                        'ground_truth': ground_truth,
                        'predicted': predicted,
                        'question': result.get('question', '')[:100] + '...',
                        'response': model_response[:200] + '...'
                    })
                
                # 按类别统计
                category = result.get('question_category', 'unknown')
                stats['by_category'][category]['total'] += 1
                if is_correct:
                    stats['by_category'][category]['correct'] += 1
                
                # 按视频统计
                video_id = result.get('original_video_id', result.get('video_id', 'unknown'))
                if video_id.startswith('urban_'):
                    video_id = video_id[6:]  # 去掉urban_前缀
                stats['by_video'][video_id]['total'] += 1
                if is_correct:
                    stats['by_video'][video_id]['correct'] += 1
            
            # 计算准确率
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            stats['answer_extraction_rate'] = (stats['total'] - stats['no_answer_extracted']) / stats['total'] if stats['total'] > 0 else 0
            
            detailed_stats[test_type] = stats
        
        return detailed_stats
    
    def create_urban_comparison_table(self, detailed_stats):
        """创建urban对比表格"""
        print("创建urban对比表格...")
        
        # 总体准确率对比
        comparison_data = []
        
        for test_type in self.test_types:
            if test_type in detailed_stats:
                stats = detailed_stats[test_type]
                comparison_data.append({
                    '测试类型': f"urban_{test_type}",
                    '总问题数': stats['total'],
                    '正确答案数': stats['correct'],
                    '准确率': f"{stats['accuracy']:.4f}",
                    '答案提取率': f"{stats['answer_extraction_rate']:.4f}",
                    '未提取答案数': stats['no_answer_extracted']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # 按类别对比
        categories = set()
        for stats in detailed_stats.values():
            categories.update(stats['by_category'].keys())
        
        category_data = []
        for category in sorted(categories):
            row = {'问题类别': category}
            for test_type in self.test_types:
                if test_type in detailed_stats and category in detailed_stats[test_type]['by_category']:
                    cat_stats = detailed_stats[test_type]['by_category'][category]
                    accuracy = cat_stats['correct'] / cat_stats['total'] if cat_stats['total'] > 0 else 0
                    row[f'urban_{test_type}_准确率'] = f"{accuracy:.4f}"
                    row[f'urban_{test_type}_问题数'] = cat_stats['total']
                else:
                    row[f'urban_{test_type}_准确率'] = "N/A"
                    row[f'urban_{test_type}_问题数'] = 0
            category_data.append(row)
        
        df_category = pd.DataFrame(category_data)
        
        return df_comparison, df_category
    
    def create_urban_visualizations(self, detailed_stats):
        """创建urban可视化图表"""
        print("创建urban可视化图表...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Urban UrbanVideo-Bench 评估结果分析', fontsize=16, fontweight='bold')
        
        # 1. 总体准确率对比
        test_types = []
        accuracies = []
        for test_type in self.test_types:
            if test_type in detailed_stats:
                test_types.append(f"urban_{test_type}")
                accuracies.append(detailed_stats[test_type]['accuracy'])
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = axes[0, 0].bar(test_types, accuracies, color=colors[:len(test_types)])
        axes[0, 0].set_title('Urban总体准确率对比')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.4f}', ha='center', va='bottom')
        
        # 2. 按类别准确率对比
        if 'original' in detailed_stats:
            categories = list(detailed_stats['original']['by_category'].keys())
            
            x = np.arange(len(categories))
            width = 0.25
            
            for i, test_type in enumerate(self.test_types):
                if test_type in detailed_stats:
                    cat_accs = []
                    for cat in categories:
                        if cat in detailed_stats[test_type]['by_category']:
                            cat_stats = detailed_stats[test_type]['by_category'][cat]
                            acc = cat_stats['correct'] / cat_stats['total'] if cat_stats['total'] > 0 else 0
                            cat_accs.append(acc)
                        else:
                            cat_accs.append(0)
                    
                    axes[0, 1].bar(x + i * width, cat_accs, width, 
                                  label=f"urban_{test_type}", color=colors[i])
            
            axes[0, 1].set_title('Urban按问题类别准确率对比')
            axes[0, 1].set_ylabel('准确率')
            axes[0, 1].set_xlabel('问题类别')
            axes[0, 1].set_xticks(x + width)
            axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
            axes[0, 1].legend()
        
        # 3. 性能下降对比
        if 'original' in detailed_stats:
            baseline_acc = detailed_stats['original']['accuracy']
            drops = []
            drop_labels = []
            
            for test_type in ['rain_fog', 'occlusion']:
                if test_type in detailed_stats:
                    test_acc = detailed_stats[test_type]['accuracy']
                    drop = baseline_acc - test_acc
                    relative_drop = (drop / baseline_acc * 100) if baseline_acc > 0 else 0
                    drops.append(drop)
                    drop_labels.append(f"urban_{test_type}\n(-{relative_drop:.1f}%)")
            
            if drops:
                bars = axes[1, 0].bar(drop_labels, drops, color=['#A23B72', '#F18F01'])
                axes[1, 0].set_title('Urban相对原始视频的性能下降')
                axes[1, 0].set_ylabel('准确率下降')
                
                # 添加数值标签
                for bar, drop in zip(bars, drops):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                   f'{drop:.4f}', ha='center', va='bottom')
        
        # 4. 答案分布对比
        if detailed_stats:
            answer_labels = ['A', 'B', 'C', 'D', 'E']
            x = np.arange(len(answer_labels))
            width = 0.25
            
            for i, test_type in enumerate(self.test_types):
                if test_type in detailed_stats:
                    answer_counts = []
                    total_answers = sum(detailed_stats[test_type]['answer_distribution'].values())
                    
                    for label in answer_labels:
                        count = detailed_stats[test_type]['answer_distribution'].get(label, 0)
                        percentage = count / total_answers if total_answers > 0 else 0
                        answer_counts.append(percentage)
                    
                    axes[1, 1].bar(x + i * width, answer_counts, width,
                                  label=f"urban_{test_type}", color=colors[i])
            
            axes[1, 1].set_title('Urban模型答案分布对比')
            axes[1, 1].set_ylabel('答案比例')
            axes[1, 1].set_xlabel('答案选项')
            axes[1, 1].set_xticks(x + width)
            axes[1, 1].set_xticklabels(answer_labels)
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def generate_urban_report(self, output_file=None):
        """生成完整的urban分析报告"""
        print("生成urban分析报告...")
        
        # 计算详细统计
        detailed_stats = self.calculate_urban_accuracy()
        
        # 创建对比表格
        df_comparison, df_category = self.create_urban_comparison_table(detailed_stats)
        
        # 创建可视化
        fig = self.create_urban_visualizations(detailed_stats)
        
        # 生成文本报告
        report = []
        report.append("# Urban UrbanVideo-Bench 评估结果分析报告\n")
        
        # 实验概述
        report.append("## 实验概述")
        report.append("本实验在UrbanVideo-Bench数据集上评估了视觉-语言模型在不同视觉条件下的鲁棒性。")
        report.append("使用urban前缀的目录结构，测试了三种条件：原始视频、雨雾效果视频、遮挡效果视频。")
        report.append("这是一个手动下载设置的评估实验。\n")
        
        # 总体结果
        report.append("## 总体结果")
        for test_type in self.test_types:
            if test_type in detailed_stats:
                stats = detailed_stats[test_type]
                report.append(f"- **urban_{test_type}测试**:")
                report.append(f"  - 准确率: {stats['accuracy']:.4f}")
                report.append(f"  - 问题总数: {stats['total']}")
                report.append(f"  - 正确答案: {stats['correct']}")
                report.append(f"  - 答案提取率: {stats['answer_extraction_rate']:.4f}")
        report.append("")
        
        # 性能对比
        if 'original' in detailed_stats:
            baseline_acc = detailed_stats['original']['accuracy']
            report.append("## 性能对比（相对于原始视频）")
            
            for test_type in ['rain_fog', 'occlusion']:
                if test_type in detailed_stats:
                    test_acc = detailed_stats[test_type]['accuracy']
                    abs_drop = baseline_acc - test_acc
                    rel_drop = (abs_drop / baseline_acc * 100) if baseline_acc > 0 else 0
                    
                    report.append(f"- **urban_{test_type}**:")
                    report.append(f"  - 绝对下降: {abs_drop:.4f}")
                    report.append(f"  - 相对下降: {rel_drop:.2f}%")
            report.append("")
        
        # 按类别分析
        report.append("## 按问题类别分析")
        if 'original' in detailed_stats:
            categories = detailed_stats['original']['by_category'].keys()
            for category in sorted(categories):
                report.append(f"### {category}")
                for test_type in self.test_types:
                    if test_type in detailed_stats and category in detailed_stats[test_type]['by_category']:
                        cat_stats = detailed_stats[test_type]['by_category'][category]
                        acc = cat_stats['correct'] / cat_stats['total'] if cat_stats['total'] > 0 else 0
                        report.append(f"- urban_{test_type}: {acc:.4f} ({cat_stats['correct']}/{cat_stats['total']})")
                report.append("")
        
        # 视频级别分析
        report.append("## 按视频分析")
        if 'original' in detailed_stats:
            videos = detailed_stats['original']['by_video'].keys()
            for video in sorted(videos):
                report.append(f"### {video}")
                for test_type in self.test_types:
                    if test_type in detailed_stats and video in detailed_stats[test_type]['by_video']:
                        vid_stats = detailed_stats[test_type]['by_video'][video]
                        acc = vid_stats['correct'] / vid_stats['total'] if vid_stats['total'] > 0 else 0
                        report.append(f"- urban_{test_type}: {acc:.4f} ({vid_stats['correct']}/{vid_stats['total']})")
                report.append("")
        
        # 错误分析示例
        report.append("## 错误分析示例")
        for test_type in self.test_types:
            if test_type in detailed_stats and detailed_stats[test_type]['incorrect_answers']:
                report.append(f"### urban_{test_type} 错误示例（前3个）")
                for i, error in enumerate(detailed_stats[test_type]['incorrect_answers'][:3]):
                    report.append(f"**错误 {i+1}:**")
                    report.append(f"- 问题ID: {error['question_id']}")
                    report.append(f"- 正确答案: {error['ground_truth']}")
                    report.append(f"- 模型答案: {error['predicted']}")
                    report.append(f"- 问题: {error['question']}")
                    report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.results_dir / "urban_analysis_report.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 保存图表
        fig_path = output_path.parent / "urban_analysis_charts.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # 保存表格
        table_path = output_path.parent / "urban_comparison_tables.xlsx"
        with pd.ExcelWriter(table_path) as writer:
            df_comparison.to_excel(writer, sheet_name='Urban总体对比', index=False)
            df_category.to_excel(writer, sheet_name='Urban按类别对比', index=False)
        
        # 保存详细统计
        stats_path = output_path.parent / "urban_detailed_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            # 转换不可序列化的对象
            serializable_stats = {}
            for test_type, stats in detailed_stats.items():
                serializable_stats[f"urban_{test_type}"] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': stats['accuracy'],
                    'answer_extraction_rate': stats['answer_extraction_rate'],
                    'no_answer_extracted': stats['no_answer_extracted'],
                    'by_category': dict(stats['by_category']),
                    'by_video': dict(stats['by_video']),
                    'answer_distribution': dict(stats['answer_distribution']),
                    'ground_truth_distribution': dict(stats['ground_truth_distribution'])
                }
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Urban分析报告生成完成:")
        print(f"  - 报告文件: {output_path}")
        print(f"  - 图表文件: {fig_path}")
        print(f"  - 表格文件: {table_path}")
        print(f"  - 统计文件: {stats_path}")
        
        return report_text, fig, df_comparison, df_category

def main():
    parser = argparse.ArgumentParser(description='分析Urban UrbanVideo-Bench评估结果')
    parser.add_argument('--results_dir', type=str, default='./urban_results',
                       help='urban结果目录路径 (默认: ./urban_results)')
    parser.add_argument('--output_file', type=str,
                       help='输出报告文件路径（可选）')
    
    args = parser.parse_args()
    
    print("=== Urban UrbanVideo-Bench 结果分析 ===")
    print(f"结果目录: {args.results_dir}")
    
    # 检查目录是否存在
    if not Path(args.results_dir).exists():
        print(f"错误: 结果目录不存在: {args.results_dir}")
        print("请确保已运行urban评估脚本并生成了结果")
        return
    
    # 创建分析器
    analyzer = UrbanResultAnalyzer(args.results_dir)
    
    # 加载结果
    analyzer.load_urban_results()
    
    # 检查是否有有效结果
    if not any(analyzer.results.values()):
        print("错误: 没有找到有效的评估结果")
        print("请检查结果文件是否存在且格式正确")
        return
    
    # 生成报告
    report_text, fig, df_comparison, df_category = analyzer.generate_urban_report(args.output_file)
    
    # 显示总结
    print("\n=== Urban分析总结 ===")
    if not df_comparison.empty:
        print(df_comparison.to_string(index=False))
    else:
        print("无可显示的对比数据")
    
    print("\n🎉 Urban结果分析完成！")

if __name__ == "__main__":
    main()