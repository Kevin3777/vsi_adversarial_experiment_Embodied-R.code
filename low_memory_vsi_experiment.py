#!/usr/bin/env python3
"""
32GB显存适配的VSI-Bench对抗实验
使用API服务或小模型替代72B本地部署
"""

import os
import shutil
import json
import random
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class LowMemoryVSIExperiment:
    def __init__(self, work_dir="./vsi_adversarial_experiment"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 设置目录结构
        self.dirs = {
            'original_videos': self.work_dir / 'original_videos',
            'rain_fog_videos': self.work_dir / 'rain_fog_videos', 
            'occlusion_videos': self.work_dir / 'occlusion_videos',
            'annotations': self.work_dir / 'annotations',
            'results': self.work_dir / 'results',
            'embodiedr': self.work_dir / 'Embodied-R.code'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def setup_vsi_bench_data(self, video_source_dir, num_videos=50):
        """设置VSI-Bench数据"""
        print("设置VSI-Bench数据...")
        
        # 1. 加载VSI-Bench标注数据
        try:
            vsi_dataset = load_dataset("nyu-visionx/VSI-Bench")
            df = vsi_dataset['test'].to_pandas()
            print(f"成功加载{len(df)}条标注数据")
        except Exception as e:
            print(f"加载失败: {e}")
            return False
        
        # 2. 筛选arkitscenes数据
        arkitscenes_data = df[df['dataset'] == 'arkitscenes'].copy()
        print(f"ARKitScenes数据: {len(arkitscenes_data)}条")
        
        # 3. 选择有视频文件的场景
        available_scenes = []
        video_source_path = Path(video_source_dir)
        
        for scene_name in arkitscenes_data['scene_name'].unique():
            possible_files = [
                video_source_path / f"{scene_name}.mp4",
                video_source_path / f"{scene_name}.avi",
                video_source_path / f"{scene_name}.mov"
            ]
            
            for video_file in possible_files:
                if video_file.exists():
                    available_scenes.append(scene_name)
                    break
        
        print(f"找到{len(available_scenes)}个有视频文件的场景")
        
        if len(available_scenes) < num_videos:
            print(f"可用视频不足，使用全部{len(available_scenes)}个")
            num_videos = len(available_scenes)
        
        # 4. 随机选择场景
        selected_scenes = random.sample(available_scenes, num_videos)
        
        # 5. 复制视频文件
        print(f"复制{len(selected_scenes)}个视频文件...")
        for scene_name in selected_scenes:
            for ext in ['.mp4', '.avi', '.mov']:
                src_file = video_source_path / f"{scene_name}{ext}"
                if src_file.exists():
                    dst_file = self.dirs['original_videos'] / f"{scene_name}.mp4"
                    shutil.copy2(src_file, dst_file)
                    print(f"复制: {scene_name}.mp4")
                    break
        
        # 6. 生成标注数据
        selected_data = arkitscenes_data[arkitscenes_data['scene_name'].isin(selected_scenes)]
        
        embodiedr_format = []
        for _, row in selected_data.iterrows():
            embodiedr_format.append({
                "Question_id": int(row['id']),
                "video_id": f"{row['scene_name']}.mp4",
                "question_category": row['question_type'],
                "question": row['question'],
                "answer": str(row['ground_truth']),
                "scene_name": row['scene_name']
            })
        
        # 保存标注文件
        annotations_file = self.dirs['annotations'] / 'test_data.json'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(embodiedr_format, f, indent=2, ensure_ascii=False)
        
        print(f"保存标注数据: {annotations_file}")
        print(f"总问题数: {len(embodiedr_format)}")
        
        return True
    
    def process_adversarial_videos(self):
        """处理对抗性视频"""
        print("生成对抗性视频...")
        
        import subprocess
        
        cmd_rain_fog = [
            "python", "video_editor.py",
            "--input_dir", str(self.dirs['original_videos']),
            "--output_dir", str(self.work_dir),
            "--rain_intensity", "0.6",
            "--fog_intensity", "0.4"
        ]
        
        try:
            subprocess.run(cmd_rain_fog, check=True)
            
            # 移动生成的视频到正确位置
            rain_fog_source = self.work_dir / "edited_videos" / "rain_fog"
            if rain_fog_source.exists():
                for video in rain_fog_source.iterdir():
                    shutil.move(str(video), str(self.dirs['rain_fog_videos']))
            
            occlusion_source = self.work_dir / "edited_videos" / "occlusion"
            if occlusion_source.exists():
                for video in occlusion_source.iterdir():
                    shutil.move(str(video), str(self.dirs['occlusion_videos']))
                
            print("对抗性视频生成完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"视频处理失败: {e}")
            return False
    
    def setup_embodiedr(self):
        """设置EmbodiedR环境"""
        print("设置EmbodiedR环境...")
        
        if not self.dirs['embodiedr'].exists():
            import subprocess
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/EmbodiedCity/Embodied-R.code.git",
                    str(self.dirs['embodiedr'])
                ], check=True)
                print("成功克隆EmbodiedR")
            except subprocess.CalledProcessError as e:
                print(f"克隆失败: {e}")
                return False
        
        # 设置数据
        embodiedr_dataset_dir = self.dirs['embodiedr'] / 'dataset' / 'complete'
        embodiedr_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(
            self.dirs['annotations'] / 'test_data.json',
            embodiedr_dataset_dir / 'test_data.json'
        )
        
        print("EmbodiedR环境设置完成")
        return True
    
    def create_test_scripts(self, use_api=True, api_key=None, small_model=None):
        """创建32GB显存适配的测试脚本"""
        print("创建32GB显存适配的测试脚本...")
        
        if use_api:
            # 使用API版本
            vision_cmd = f'''python train/conver_format/VLM_perception_API.py \\
  --data_paths dataset/complete/test_data.json \\
  --api_key {api_key or "YOUR_API_KEY"} \\
  --base_url https://dashscope.aliyuncs.com/compatible-mode/v1'''
        else:
            # 使用小模型版本
            model_path = small_model or "Qwen/Qwen2-VL-7B-Instruct"
            vision_cmd = f'''python train/conver_format/VLM_perception_local.py \\
  --model_path {model_path} \\
  --data_paths dataset/complete/test_data.json'''
        
        test_script = f'''#!/bin/bash

# 32GB显存适配的VSI-Bench对抗性测试脚本

WORK_DIR="{self.work_dir}"
EMBODIEDR_DIR="$WORK_DIR/Embodied-R.code"

echo "开始32GB显存适配的VSI-Bench测试..."

cd "$EMBODIEDR_DIR"

# 测试原始视频
echo "测试原始视频..."
{vision_cmd} \\
  --folder_path "$WORK_DIR/original_videos" \\
  --save_path "$WORK_DIR/results/original"

cd infer  
bash run_batch_inference.sh \\
  --model "Qwen/Qwen2.5-VL-3B-Instruct" \\
  --input_file "$WORK_DIR/results/original/test_data.json" \\
  --output_file "$WORK_DIR/results/original_results.json"

cd ..

# 测试雨雾视频
echo "测试雨雾视频..."
{vision_cmd} \\
  --folder_path "$WORK_DIR/rain_fog_videos" \\
  --save_path "$WORK_DIR/results/rain_fog"

cd infer
bash run_batch_inference.sh \\
  --model "Qwen/Qwen2.5-VL-3B-Instruct" \\
  --input_file "$WORK_DIR/results/rain_fog/test_data.json" \\
  --output_file "$WORK_DIR/results/rain_fog_results.json"

cd ..

# 测试遮挡视频  
echo "测试遮挡视频..."
{vision_cmd} \\
  --folder_path "$WORK_DIR/occlusion_videos" \\
  --save_path "$WORK_DIR/results/occlusion"

cd infer
bash run_batch_inference.sh \\
  --model "Qwen/Qwen2.5-VL-3B-Instruct" \\
  --input_file "$WORK_DIR/results/occlusion/test_data.json" \\
  --output_file "$WORK_DIR/results/occlusion_results.json"

echo "所有测试完成！"
echo "结果文件："
echo "  - 原始: $WORK_DIR/results/original_results.json"
echo "  - 雨雾: $WORK_DIR/results/rain_fog_results.json"  
echo "  - 遮挡: $WORK_DIR/results/occlusion_results.json"
'''
        
        script_file = self.work_dir / 'run_low_memory_test.sh'
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        os.chmod(script_file, 0o755)
        print(f"测试脚本保存到: {script_file}")
        
        if use_api and not api_key:
            print("\n重要提醒: 请在脚本中设置你的API密钥!")
    
    def run_complete_setup(self, video_source_dir, num_videos=50, use_api=True, api_key=None, small_model=None):
        """运行完整的32GB显存适配实验设置"""
        print("开始32GB显存适配的VSI-Bench对抗性实验设置")
        print("=" * 60)
        
        if not self.setup_vsi_bench_data(video_source_dir, num_videos):
            return False
        
        if not self.process_adversarial_videos():
            return False
        
        if not self.setup_embodiedr():
            return False
        
        self.create_test_scripts(use_api, api_key, small_model)
        
        print("\n" + "=" * 60)
        print("32GB显存适配实验环境设置完成！")
        print(f"工作目录: {self.work_dir}")
        
        print("\n下一步操作:")
        if use_api:
            print("1. 在阿里云百炼平台申请API密钥")
            print("2. 在脚本中设置API密钥")
            print("3. 只需下载3B推理模型: Qwen/Qwen2.5-VL-3B-Instruct")
        else:
            print("1. 下载小型视觉模型 (如 Qwen/Qwen2-VL-7B-Instruct)")
            print("2. 下载3B推理模型: Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("4. 运行测试: ./run_low_memory_test.sh")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='32GB显存适配的VSI-Bench对抗性实验')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='ARKitScenes视频文件目录')
    parser.add_argument('--work_dir', type=str, default='./vsi_adversarial_experiment',
                       help='工作目录')
    parser.add_argument('--num_videos', type=int, default=50,
                       help='选择的视频数量')
    parser.add_argument('--use_api', action='store_true', default=False,
                       help='使用API而非本地大模型')
    parser.add_argument('--api_key', type=str, 
                       help='API密钥（如果使用API）')
    parser.add_argument('--small_model', type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help='小型视觉模型路径（如果不使用API）')
    
    args = parser.parse_args()
    
    experiment = LowMemoryVSIExperiment(args.work_dir)
    
    success = experiment.run_complete_setup(
        args.video_dir, 
        args.num_videos, 
        args.use_api,
        args.api_key,
        args.small_model
    )
    
    if success:
        print("\n实验设置成功完成！")
    else:
        print("\n实验设置失败！")

if __name__ == "__main__":
    main()