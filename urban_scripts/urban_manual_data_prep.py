#!/usr/bin/env python3
"""
UrbanVideo-Bench手动设置版本的数据准备脚本
适用于手动下载的5个视频和MCQ.parquet文件
使用urban前缀的目录结构
"""

import json
import pandas as pd
import os
import shutil
from pathlib import Path
import argparse

def load_manual_data(mcq_path, videos_dir):
    """
    加载手动下载的UrbanVideo-Bench数据
    """
    print("正在加载手动下载的UrbanVideo-Bench数据...")
    
    # 加载MCQ数据
    if not os.path.exists(mcq_path):
        raise FileNotFoundError(f"MCQ文件不存在: {mcq_path}")
    
    df = pd.read_parquet(mcq_path)
    print(f"加载了 {len(df)} 个问题")
    
    # 转换为字典格式
    data = []
    for _, row in df.iterrows():
        data.append({
            "Question_id": int(row["Question_id"]),
            "video_id": row["video_id"],
            "question_category": row["question_category"],
            "question": row["question"],
            "answer": row["answer"]
        })
    
    # 检查手动下载的视频文件
    video_files = []
    if os.path.exists(videos_dir):
        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
        print(f"视频目录中找到 {len(video_files)} 个视频文件:")
        for video in sorted(video_files):
            print(f"  - {video}")
    else:
        raise FileNotFoundError(f"视频目录不存在: {videos_dir}")
    
    return data, video_files

def extract_questions_for_videos(data, video_files):
    """
    提取手动下载视频对应的所有问题
    """
    print("提取对应视频的问题...")
    
    # 统计每个视频的问题数量
    video_question_count = {}
    for item in data:
        video_id = item["video_id"]
        video_question_count[video_id] = video_question_count.get(video_id, 0) + 1
    
    print(f"数据集包含 {len(video_question_count)} 个不同的视频")
    
    # 找出手动下载的视频对应的问题
    selected_questions = []
    selected_videos = []
    
    for video_file in video_files:
        if video_file in video_question_count:
            selected_videos.append(video_file)
            questions_count = video_question_count[video_file]
            print(f"  {video_file}: {questions_count} 个问题")
            
            # 提取该视频的所有问题
            for item in data:
                if item["video_id"] == video_file:
                    selected_questions.append(item)
        else:
            print(f"  警告: {video_file} 在MCQ数据中未找到对应问题")
    
    print(f"\n选择的视频:")
    for i, video in enumerate(selected_videos, 1):
        questions_for_video = sum(1 for q in selected_questions if q["video_id"] == video)
        print(f"  {i}. {video} ({questions_for_video} 个问题)")
    
    print(f"\n总共提取了 {len(selected_questions)} 个问题")
    
    return selected_videos, selected_questions

def create_urban_structure(work_dir, selected_videos, videos_dir):
    """
    创建urban前缀的目录结构并验证视频文件
    """
    work_path = Path(work_dir)
    
    print("创建urban前缀的目录结构...")
    
    # 创建目录结构
    directories = [
        "urban_videos/original",
        "urban_videos/rain_fog", 
        "urban_videos/occlusion",
        "urban_annotations",
        "urban_results/original",
        "urban_results/rain_fog",
        "urban_results/occlusion"
    ]
    
    for dir_name in directories:
        (work_path / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"  创建目录: {dir_name}")
    
    # 验证原始视频是否存在于正确位置
    videos_source = Path(videos_dir)
    videos_dest = work_path / "urban_videos" / "original"
    
    verified_videos = []
    for video_id in selected_videos:
        source_file = videos_source / video_id
        dest_file = videos_dest / video_id
        
        if source_file.exists():
            if not dest_file.exists():
                # 如果目标不存在，复制文件
                shutil.copy2(source_file, dest_file)
                print(f"  复制视频: {video_id}")
            else:
                print(f"  视频已存在: {video_id}")
            verified_videos.append(video_id)
        else:
            print(f"  错误: 视频文件不存在: {source_file}")
    
    print(f"\n验证完成，共 {len(verified_videos)} 个视频准备就绪")
    return verified_videos

def save_urban_annotations(questions, output_file):
    """
    保存问题数据到urban_annotations目录
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"保存 {len(questions)} 个问题到: {output_file}")

def create_urban_summary(work_dir, selected_videos, questions):
    """
    创建实验总结文件
    """
    summary = {
        "experiment_info": {
            "dataset": "UrbanVideo-Bench",
            "setup_method": "manual_download",
            "directory_prefix": "urban_",
            "selected_videos": len(selected_videos),
            "total_questions": len(questions),
            "created_at": pd.Timestamp.now().isoformat()
        },
        "videos": selected_videos,
        "question_categories": {},
        "questions_per_video": {},
        "directory_structure": {
            "videos": "urban_videos/",
            "annotations": "urban_annotations/",
            "results": "urban_results/"
        }
    }
    
    # 统计问题类别
    for q in questions:
        category = q["question_category"]
        summary["question_categories"][category] = summary["question_categories"].get(category, 0) + 1
    
    # 统计每个视频的问题数
    for q in questions:
        video_id = q["video_id"]
        summary["questions_per_video"][video_id] = summary["questions_per_video"].get(video_id, 0) + 1
    
    summary_file = os.path.join(work_dir, "urban_experiment_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"实验总结保存到: {summary_file}")
    return summary

def create_video_mapping_file(work_dir, selected_videos):
    """
    创建视频映射文件，用于后续脚本使用
    """
    mapping = {
        "original_videos": selected_videos,
        "rain_fog_videos": [video.replace('.mp4', '_rain_fog.mp4') for video in selected_videos],
        "occlusion_videos": [video.replace('.mp4', '_occlusion.mp4') for video in selected_videos]
    }
    
    mapping_file = os.path.join(work_dir, "urban_video_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"视频映射文件保存到: {mapping_file}")
    return mapping

def main():
    parser = argparse.ArgumentParser(description='手动设置UrbanVideo-Bench评估数据')
    parser.add_argument('--mcq_path', type=str, default='./urban_data/MCQ.parquet',
                       help='MCQ.parquet文件路径 (默认: ./urban_data/MCQ.parquet)')
    parser.add_argument('--videos_dir', type=str, default='./urban_videos/original',
                       help='手动下载的视频目录 (默认: ./urban_videos/original)')
    parser.add_argument('--work_dir', type=str, default='./',
                       help='工作目录路径 (默认: ./)')
    
    args = parser.parse_args()
    
    print("=== UrbanVideo-Bench手动设置版数据准备 ===")
    print(f"MCQ文件: {args.mcq_path}")
    print(f"视频目录: {args.videos_dir}")
    print(f"工作目录: {args.work_dir}")
    print()
    
    try:
        # 加载数据
        data, video_files = load_manual_data(args.mcq_path, args.videos_dir)
        
        # 提取对应的问题
        selected_videos, selected_questions = extract_questions_for_videos(data, video_files)
        
        if not selected_videos:
            print("错误: 没有找到有效的视频和问题对应关系")
            return
        
        # 创建目录结构
        verified_videos = create_urban_structure(args.work_dir, selected_videos, args.videos_dir)
        
        # 只保留验证成功的视频的问题
        final_questions = [q for q in selected_questions if q["video_id"] in verified_videos]
        
        # 保存问题数据
        annotations_file = os.path.join(args.work_dir, "urban_annotations", "test_data.json")
        save_urban_annotations(final_questions, annotations_file)
        
        # 创建总结
        summary = create_urban_summary(args.work_dir, verified_videos, final_questions)
        
        # 创建视频映射
        mapping = create_video_mapping_file(args.work_dir, verified_videos)
        
        print("\n=== 手动设置完成 ===")
        print(f"成功准备了 {len(verified_videos)} 个视频")
        print(f"总共 {len(final_questions)} 个问题")
        print(f"问题类别分布:")
        for category, count in summary["question_categories"].items():
            print(f"  {category}: {count}")
        
        print(f"\n目录结构:")
        print(f"  - 原始视频: urban_videos/original/ ({len(verified_videos)} 个文件)")
        print(f"  - 问题数据: urban_annotations/test_data.json")
        print(f"  - 实验总结: urban_experiment_summary.json")
        
        print(f"\n下一步:")
        print(f"1. 运行视频效果生成脚本: python urban_video_effects.py")
        print(f"2. 运行评估脚本: ./urban_video_evaluation.sh")
        print(f"3. 分析结果: python urban_video_analysis.py")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查文件路径和权限")

if __name__ == "__main__":
    main()