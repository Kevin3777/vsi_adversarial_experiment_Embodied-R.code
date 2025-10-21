"""
CogVideoX1.5 视频遮挡添加工具 - 性能优化版
使用最新的 CogVideoX1.5-5B-I2V 模型，效果更好，速度更快

使用方法:
    python video_occlusion_optimized.py

作者: AI Assistant
日期: 2025-10-05
"""

import os
import random
import torch
import cv2
from pathlib import Path
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import warnings
import time
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
CONFIG = {
    # 路径配置
    'video_input_dir': "/root/autodl-tmp/sida/urban_videos/original",
    'model_dir': "/root/autodl-tmp/sida/occlusion_new/models",
    'output_dir': "/root/autodl-tmp/sida/urban_videos/with_occlusion",
    
    # 模型配置 - 使用最新的1.5版本
    'model_id': "THUDM/CogVideoX1.5-5B-I2V",
    
    # 生成参数（优化后的推荐值）
    'num_inference_steps': 50,      # 推理步数
    'num_frames': 81,               # 1.5版本支持81帧（更长）
    'guidance_scale': 6.0,          # 引导系数
    'fps': 16,                      # 1.5版本推荐16fps
    'seed': 42,                     # 随机种子
    
    # 内存优化（32GB显存建议配置）
    'use_cpu_offload': False,       # 32GB显存可以不用CPU offload
    'use_vae_slicing': True,        # 开启VAE切片
    'use_vae_tiling': True,         # 开启VAE平铺
    'dtype': torch.bfloat16,        # 使用BF16获得更好效果
    
    # 处理选项
    'random_occlusion': True,       # 随机选择遮挡
    'max_size': 1360,               # 图像最大边
    'min_size': 768,                # 图像最小边
    'batch_size': 1,                # 批处理大小
}

# 遮挡提示词（经过优化的prompts）
OCCLUSION_PROMPTS = [
    "A close-up car in the foreground partially blocking the view, with a vehicle silhouette passing by the camera, creating a natural street perspective with realistic automobile obstruction",
    "Tree branches and leaves in the foreground, natural foliage partially blocking the camera view, with a tree trunk visible in front, creating an outdoor perspective with vegetation obstruction",
    "A building corner in the foreground, architectural elements blocking part of the view, wall edge obscuring the perspective, urban structure creating a natural city obstruction",
    "A person walking directly in front of the camera, pedestrian silhouette in the foreground, human figure partially blocking the street view, realistic passerby obstruction",
    "Traffic sign pole in the foreground, street signage blocking part of the view, urban infrastructure obstruction with road signs clearly visible in front",
    "Various foreground objects partially blocking the view, natural urban elements obstructing the camera, street furniture creating realistic perspective with obstacles",
    "Fence or metal railing in the foreground, barrier partially blocking the street view, guardrail obstruction creating depth in the urban scene",
    "Bus stop shelter in the foreground, urban furniture blocking part of the view, street infrastructure with city elements clearly visible in front of camera"
]


class VideoOcclusionProcessor:
    """视频遮挡处理器 - 优化版"""
    
    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processing_stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0
        }
        
    def print_system_info(self):
        """打印系统信息"""
        print("\n" + "="*80)
        print("系统信息".center(80))
        print("="*80)
        
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ 显存: {total_memory:.2f} GB")
            print(f"✓ CUDA版本: {torch.version.cuda}")
        else:
            print("✗ 警告: 未检测到GPU，将使用CPU (非常慢)")
            
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ 设备: {self.device}")
        print("="*80 + "\n")
        
    def setup_model(self):
        """加载模型 - 增强断点续传版"""
        print(f"正在加载模型: {self.config['model_id']}")
        print(f"模型保存路径: {self.config['model_dir']}\n")
        
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # 设置环境变量以优化下载
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像（国内用户）
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用高速传输
        
        start_time = time.time()
        
        print("📥 下载/加载模型中（支持断点续传）...")
        
        # 导入必要的库
        from huggingface_hub import snapshot_download
        import time as time_module
        
        # 重试配置
        max_retries = 5
        retry_delay = 10
        
        # 先使用 snapshot_download 下载模型（支持断点续传）
        model_downloaded = False
        for attempt in range(max_retries):
            try:
                print(f"\n尝试下载模型文件 (第 {attempt + 1}/{max_retries} 次)...")
                snapshot_download(
                    repo_id=self.config['model_id'],
                    cache_dir=self.config['model_dir'],
                    resume_download=True,  # 关键：启用断点续传
                    max_workers=4,  # 并行下载
                    local_files_only=False,
                    ignore_patterns=["*.md", "*.txt"]  # 跳过不必要的文件
                )
                print("✓ 模型文件下载完成")
                model_downloaded = True
                break
                
            except KeyboardInterrupt:
                print("\n⚠️  下载被用户中断")
                print("💡 提示：再次运行脚本将从断点继续下载")
                raise
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  下载遇到错误: {str(e)[:100]}")
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    time_module.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 120)  # 指数退避，最多2分钟
                else:
                    print("❌ 多次重试后仍然失败")
                    print("💡 尝试使用已下载的部分文件继续...")
        
        # 加载 pipeline
        try:
            print("\n📦 正在加载模型到内存...")
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.config['model_id'],
                torch_dtype=self.config['dtype'],
                cache_dir=self.config['model_dir'],
                local_files_only=model_downloaded,  # 如果下载完成，只使用本地文件
                resume_download=True
            )
            
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            print("\n可能的原因:")
            print("1. 模型文件未完全下载")
            print("2. 显存不足")
            print("3. 依赖库版本不兼容")
            print("\n建议操作:")
            print("- 再次运行脚本继续下载")
            print("- 检查磁盘空间是否充足（需要约20GB）")
            print(f"- 检查模型目录: {self.config['model_dir']}")
            raise
        
        # 根据显存情况应用优化
        if self.config['use_cpu_offload']:
            print("✓ 启用CPU offload优化")
            self.pipe.enable_model_cpu_offload()
        else:
            print("✓ 模型加载到GPU")
            self.pipe.to(self.device)
            
        if self.config['use_vae_slicing']:
            print("✓ 启用VAE切片优化")
            self.pipe.vae.enable_slicing()
            
        if self.config['use_vae_tiling']:
            print("✓ 启用VAE平铺优化")
            self.pipe.vae.enable_tiling()
        
        load_time = time.time() - start_time
        print(f"\n✓ 模型加载完成! 总耗时: {load_time:.2f}秒\n")
        
    def extract_first_frame(self, video_path):
        """提取视频第一帧"""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise ValueError(f"无法读取视频: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), fps, total_frames
    
    def resize_image(self, image):
        """调整图像尺寸 - 优化版"""
        w, h = image.size
        
        # 计算新尺寸
        min_dim = min(w, h)
        max_dim = max(w, h)
        
        if min_dim < self.config['min_size']:
            scale = self.config['min_size'] / min_dim
            new_w, new_h = int(w * scale), int(h * scale)
        elif max_dim > self.config['max_size']:
            scale = self.config['max_size'] / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = w, h
        
        # 确保尺寸能被16整除（重要！）
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        
        # 使用高质量重采样
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def generate_video(self, video_path, output_path, occlusion_prompt):
        """生成带遮挡的视频 - 优化版"""
        print(f"\n{'='*80}")
        print(f"处理视频: {os.path.basename(video_path)}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 提取第一帧和视频信息
            first_frame, original_fps, total_frames = self.extract_first_frame(video_path)
            original_size = first_frame.size
            
            # 调整图像尺寸
            first_frame = self.resize_image(first_frame)
            print(f"📐 图像尺寸: {original_size} -> {first_frame.size}")
            print(f"🎬 原始视频: {original_fps}fps, {total_frames}帧")
            
            # 构建优化的prompt
            full_prompt = (
                f"A high-quality urban street scene with realistic motion and natural lighting. "
                f"{occlusion_prompt}. "
                f"The scene maintains consistent dynamics and smooth motion. "
                f"Cinematic quality, detailed foreground and background, natural perspective."
            )
            
            print(f"📝 遮挡效果: {occlusion_prompt[:70]}...")
            print(f"\n⚙️  生成参数:")
            print(f"   - 推理步数: {self.config['num_inference_steps']}")
            print(f"   - 生成帧数: {self.config['num_frames']}")
            print(f"   - 输出帧率: {self.config['fps']}fps")
            print(f"   - 引导系数: {self.config['guidance_scale']}")
            
            # 生成视频
            print(f"\n🎨 正在生成视频...")
            
            generator = torch.Generator(device=self.device)
            if self.config['seed'] is not None:
                generator.manual_seed(self.config['seed'])
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            video_frames = self.pipe(
                prompt=full_prompt,
                image=first_frame,
                num_videos_per_prompt=1,
                num_inference_steps=self.config['num_inference_steps'],
                num_frames=self.config['num_frames'],
                guidance_scale=self.config['guidance_scale'],
                generator=generator,
            ).frames[0]
            
            # 保存视频
            print(f"💾 保存视频...")
            export_to_video(video_frames, output_path, fps=self.config['fps'])
            
            generation_time = time.time() - start_time
            print(f"\n✓ 完成! 耗时: {generation_time:.2f}秒")
            print(f"✓ 保存至: {output_path}\n")
            
            return True, generation_time
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"\n✗ 错误: {str(e)}")
            print(f"✗ 失败耗时: {error_time:.2f}秒\n")
            return False, error_time
    
    def process_all_videos(self):
        """批量处理所有视频"""
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 获取视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(self.config['video_input_dir']).glob(f'*{ext}'))
        
        if not video_files:
            print(f"❌ 错误: 在 {self.config['video_input_dir']} 中未找到视频文件!")
            return
        
        self.processing_stats['total_videos'] = len(video_files)
        print(f"\n📁 找到 {len(video_files)} 个视频文件\n")
        
        # 显示将要处理的文件
        print("待处理视频:")
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {vf.name}")
        print()
        
        # 处理每个视频
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'#'*80}")
            print(f"进度: [{i}/{len(video_files)}]")
            print(f"{'#'*80}")
            
            # 选择遮挡效果
            if self.config['random_occlusion']:
                occlusion_prompt = random.choice(OCCLUSION_PROMPTS)
            else:
                occlusion_prompt = OCCLUSION_PROMPTS[i % len(OCCLUSION_PROMPTS)]
            
            # 生成输出文件名
            output_filename = f"{video_path.stem}_occluded.mp4"
            output_path = os.path.join(self.config['output_dir'], output_filename)
            
            # 生成视频
            success, duration = self.generate_video(
                str(video_path),
                output_path,
                occlusion_prompt
            )
            
            if success:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
            
            self.processing_stats['total_time'] += duration
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 打印统计信息
        self.print_summary()
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "="*80)
        print("处理完成统计".center(80))
        print("="*80)
        print(f"总视频数: {self.processing_stats['total_videos']}")
        print(f"成功: {self.processing_stats['successful']} ✓")
        print(f"失败: {self.processing_stats['failed']} ✗")
        print(f"总耗时: {self.processing_stats['total_time']:.2f}秒")
        
        if self.processing_stats['successful'] > 0:
            avg_time = self.processing_stats['total_time'] / self.processing_stats['successful']
            print(f"平均每个视频: {avg_time:.2f}秒")
        
        print(f"\n输出目录: {self.config['output_dir']}")
        print("="*80 + "\n")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*15 + "CogVideoX1.5 视频遮挡添加工具 - 优化版" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    
    # 初始化处理器
    processor = VideoOcclusionProcessor(CONFIG)
    
    # 显示系统信息
    processor.print_system_info()
    
    # 显示配置
    print("当前配置:")
    print(f"  📂 输入目录: {CONFIG['video_input_dir']}")
    print(f"  📂 输出目录: {CONFIG['output_dir']}")
    print(f"  📂 模型目录: {CONFIG['model_dir']}")
    print(f"  🤖 模型: {CONFIG['model_id']}")
    print(f"  🎲 遮挡方式: {'随机' if CONFIG['random_occlusion'] else '顺序'}")
    print(f"  🎬 输出规格: {CONFIG['num_frames']}帧 @ {CONFIG['fps']}fps")
    print(f"  💾 数据类型: {CONFIG['dtype']}")
    
    # 确认开始
    print("\n" + "="*80)
    response = input("确认配置无误，按Enter键开始处理 (或输入 q 退出): ")
    if response.lower() == 'q':
        print("已取消")
        return
    
    # 加载模型
    processor.setup_model()
    
    # 处理视频
    processor.process_all_videos()
    
    print("🎉 程序结束!\n")


if __name__ == "__main__":
    main()