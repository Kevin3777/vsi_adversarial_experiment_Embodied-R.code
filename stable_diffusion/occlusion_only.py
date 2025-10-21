#!/usr/bin/env python3
"""
Stable Diffusion 视频批量编辑 - 仅遮挡效果
批处理优化版 - 适用于32GB显存
"""

import os
import glob
import subprocess
from pathlib import Path
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import sys
import random

# ==================== 配置区域 ====================
WORK_DIR = "/root/autodl-tmp/sida"
SD_DIR = f"{WORK_DIR}/stable_diffusion"
MODEL_DIR = f"{SD_DIR}/models"
VIDEO_INPUT_DIR = f"{WORK_DIR}/urban_videos/original"
VIDEO_OUTPUT_DIR = f"{WORK_DIR}/urban_videos/processed"
TEMP_DIR = f"{SD_DIR}/temp"

# 多样化的遮挡Prompt列表
OCCLUSION_PROMPTS = [
    "close-up car in foreground partially blocking view, vehicle silhouette in front, car passing by camera, automobile obstruction, natural street perspective",
    "tree branches and leaves in foreground, natural foliage partially blocking view, tree trunk in front of camera, vegetation obstruction, outdoor perspective",
    "building corner in foreground, architectural element blocking view, wall edge obscuring perspective, structure in front, urban obstruction",
    "person walking in front of camera, pedestrian silhouette in foreground, human figure partially blocking view, passerby obstruction",
    "traffic sign in foreground, street pole blocking view, urban signage obstruction, road infrastructure in front",
    "foreground objects partially blocking view, natural urban obstruction, street elements in front of camera, realistic perspective with obstacles",
    "fence or railing in foreground, barrier partially blocking view, guardrail obstruction, metal structure in front",
    "bus stop shelter in foreground, urban furniture blocking view, street infrastructure obstruction, city elements in front of camera"
]

NEGATIVE_PROMPT = "blurry, distorted, low quality, bad anatomy, deformed, ugly, artificial, unrealistic placement"

# 参数配置
FPS = 8
STRENGTH = 0.35
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
BATCH_SIZE = 128  # 32GB显存可以设置为4-8，根据实际情况调整

# ==================== 核心函数 ====================

def load_model():
    """加载Stable Diffusion模型"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ['HF_HOME'] = MODEL_DIR
    os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR
    
    model_path = os.path.join(MODEL_DIR, "stable-diffusion-v1-5")
    
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"从本地加载模型: {model_path}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
    else:
        print(f"下载模型到: {model_path}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=MODEL_DIR
        )
        pipe.save_pretrained(model_path)
    
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print(f"模型加载完成，批处理大小: {BATCH_SIZE}\n")
    return pipe

def extract_frames(video_path, output_dir, fps=8):
    """提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(','))
        
        if width > height:
            scale = f"512:-1"
        else:
            scale = f"-1:512"
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps},scale={scale}',
            '-q:v', '2',
            f'{output_dir}/frame_%06d.png',
            '-y'
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        frames = sorted(glob.glob(f'{output_dir}/frame_*.png'))
        return frames, (width, height)
    
    except Exception as e:
        print(f"      提取帧失败: {e}")
        return [], None

def process_frames_batch(pipe, frame_paths, prompt, output_paths):
    """批处理多帧"""
    try:
        images = [Image.open(fp).convert('RGB') for fp in frame_paths]
        
        results = pipe(
            prompt=[prompt] * len(images),
            negative_prompt=[NEGATIVE_PROMPT] * len(images),
            image=images,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS
        ).images
        
        for result, output_path in zip(results, output_paths):
            result.save(output_path)
        
        return len(results)
    
    except Exception as e:
        print(f"批处理失败: {e}")
        return 0

def frames_to_video(frames_dir, output_path, fps=8, original_size=None):
    """合成视频"""
    pattern = f"{frames_dir}/frame_%06d.png"
    
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', pattern,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p'
    ]
    
    if original_size:
        width, height = original_size
        cmd.extend(['-vf', f'scale={width}:{height}'])
    
    cmd.extend([output_path, '-y'])
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def process_video(pipe, video_path):
    """处理单个视频"""
    video_name = Path(video_path).stem
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_name}_occlusion.mp4")
    
    print(f"    处理: {video_name}")
    
    temp_video_dir = os.path.join(TEMP_DIR, f"{video_name}_occlusion")
    input_frames_dir = os.path.join(temp_video_dir, "input")
    output_frames_dir = os.path.join(temp_video_dir, "output")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    print(f"       [1/3] 提取视频帧...")
    frames, original_size = extract_frames(video_path, input_frames_dir, fps=FPS)
    
    if not frames:
        print(f"    提取帧失败")
        return False
    
    print(f"       完成，共 {len(frames)} 帧")
    
    prompt = random.choice(OCCLUSION_PROMPTS)
    print(f"       遮挡类型: {prompt[:60]}...")
    
    print(f"       [2/3] 批处理每一帧 (batch_size={BATCH_SIZE})...")
    success_count = 0
    
    # 批处理帧
    with tqdm(total=len(frames), desc="       进度", ncols=80) as pbar:
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i:i+BATCH_SIZE]
            batch_output_paths = [
                os.path.join(output_frames_dir, Path(fp).name) 
                for fp in batch_frames
            ]
            
            count = process_frames_batch(pipe, batch_frames, prompt, batch_output_paths)
            success_count += count
            pbar.update(len(batch_frames))
    
    print(f"       成功 {success_count}/{len(frames)} 帧")
    
    if success_count > 0:
        print(f"       [3/3] 合成视频...")
        if frames_to_video(output_frames_dir, output_path, fps=FPS, original_size=original_size):
            print(f"    完成: {Path(output_path).name}\n")
            subprocess.run(['rm', '-rf', temp_video_dir])
            return True
        else:
            print(f"    合成视频失败\n")
            return False
    else:
        print(f"    没有成功处理的帧\n")
        return False

# ==================== 主程序 ====================

def main():
    print("\n" + "="*70)
    print("Stable Diffusion 视频遮挡效果处理 (批处理优化)")
    print("="*70 + "\n")
    
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    if not os.path.exists(VIDEO_INPUT_DIR):
        print(f"输入目录不存在: {VIDEO_INPUT_DIR}")
        sys.exit(1)
    
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(VIDEO_INPUT_DIR, ext)))
    
    if not video_files:
        print(f"未找到视频文件: {VIDEO_INPUT_DIR}")
        sys.exit(1)
    
    print(f"找到 {len(video_files)} 个视频")
    print(f"输入: {VIDEO_INPUT_DIR}")
    print(f"输出: {VIDEO_OUTPUT_DIR}")
    print(f"批处理大小: {BATCH_SIZE} (32GB显存优化)\n")
    
    pipe = load_model()
    
    success_count = 0
    for idx, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        
        print(f"{'='*70}")
        print(f"[{idx}/{len(video_files)}] {video_name}")
        print(f"{'='*70}")
        
        if process_video(pipe, video_path):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"全部完成")
    print(f"成功: {success_count}/{len(video_files)}")
    print(f"输出: {VIDEO_OUTPUT_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()