"""
CogVideoX1.5 视频遮挡添加工具 - 修复版
修复了依赖库兼容性问题

使用方法:
    python video_occlusion_fixed.py

作者: AI Assistant
日期: 2025-10-06
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
    
    # 模型配置
    'model_id': "THUDM/CogVideoX1.5-5B-I2V",
    
    # 生成参数
    'num_inference_steps': 50,
    'num_frames': 81,
    'guidance_scale': 6.0,
    'fps': 16,
    'seed': 42,
    
    # 内存优化
    'use_cpu_offload': False,
    'use_vae_slicing': True,
    'use_vae_tiling': True,
    'dtype': torch.bfloat16,
    
    # 处理选项
    'random_occlusion': True,
    'max_size': 1360,
    'min_size': 768,
    'batch_size': 1,
}

# 遮挡提示词
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
    """视频遮挡处理器 - 修复版"""
    
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
        
        # 打印关键依赖版本
        try:
            import transformers
            import diffusers
            import accelerate
            print(f"✓ Transformers版本: {transformers.__version__}")
            print(f"✓ Diffusers版本: {diffusers.__version__}")
            print(f"✓ Accelerate版本: {accelerate.__version__}")
        except ImportError as e:
            print(f"⚠️  警告: 无法获取某些依赖版本 - {e}")
            
        print(f"✓ 设备: {self.device}")
        print("="*80 + "\n")
        
    def setup_model(self):
        """加载模型 - 修复版"""
        print(f"正在加载模型: {self.config['model_id']}")
        print(f"模型保存路径: {self.config['model_dir']}\n")
        
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # 设置环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        
        start_time = time.time()
        
        try:
            print("📦 正在加载模型到内存...")
            
            # 修复：不传递不兼容的参数
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.config['model_id'],
                torch_dtype=self.config['dtype'],
                cache_dir=self.config['model_dir'],
                # 移除了 resume_download 和 local_files_only 参数
            )
            
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            print("\n🔧 尝试使用备用加载方式...")
            
            try:
                # 备用方案：分步加载
                from transformers import T5EncoderModel, T5Tokenizer
                
                print("   1. 加载文本编码器...")
                text_encoder = T5EncoderModel.from_pretrained(
                    self.config['model_id'],
                    subfolder="text_encoder",
                    torch_dtype=self.config['dtype'],
                    cache_dir=self.config['model_dir'],
                )
                
                print("   2. 加载完整pipeline...")
                self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                    self.config['model_id'],
                    text_encoder=text_encoder,
                    torch_dtype=self.config['dtype'],
                    cache_dir=self.config['model_dir'],
                )
                print("✓ 备用加载方式成功!")
                
            except Exception as e2:
                print(f"\n❌ 备用加载也失败: {e2}")
                print("\n💡 建议:")
                print("1. 更新依赖库:")
                print("   pip install --upgrade transformers diffusers accelerate")
                print("2. 或安装指定版本:")
                print("   pip install transformers==4.46.0 diffusers==0.31.0")
                raise
        
        # 应用优化
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
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), fps, total_frames
    
    def resize_image(self, image):
        """调整图像尺寸"""
        w, h = image.size
        
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
        
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def generate_video(self, video_path, output_path, occlusion_prompt):
        """生成带遮挡的视频"""
        print(f"\n{'='*80}")
        print(f"处理视频: {os.path.basename(video_path)}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            first_frame, original_fps, total_frames = self.extract_first_frame(video_path)
            original_size = first_frame.size
            
            first_frame = self.resize_image(first_frame)
            print(f"📐 图像尺寸: {original_size} -> {first_frame.size}")
            print(f"🎬 原始视频: {original_fps}fps, {total_frames}帧")
            
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
            
            print(f"\n🎨 正在生成视频...")
            
            generator = torch.Generator(device=self.device)
            if self.config['seed'] is not None:
                generator.manual_seed(self.config['seed'])
            
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
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(self.config['video_input_dir']).glob(f'*{ext}'))
        
        if not video_files:
            print(f"❌ 错误: 在 {self.config['video_input_dir']} 中未找到视频文件!")
            return
        
        self.processing_stats['total_videos'] = len(video_files)
        print(f"\n📁 找到 {len(video_files)} 个视频文件\n")
        
        print("待处理视频:")
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {vf.name}")
        print()
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'#'*80}")
            print(f"进度: [{i}/{len(video_files)}]")
            print(f"{'#'*80}")
            
            if self.config['random_occlusion']:
                occlusion_prompt = random.choice(OCCLUSION_PROMPTS)
            else:
                occlusion_prompt = OCCLUSION_PROMPTS[i % len(OCCLUSION_PROMPTS)]
            
            output_filename = f"{video_path.stem}_occluded.mp4"
            output_path = os.path.join(self.config['output_dir'], output_filename)
            
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
    print("║" + " "*15 + "CogVideoX1.5 视频遮挡添加工具 - 修复版" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    
    processor = VideoOcclusionProcessor(CONFIG)
    processor.print_system_info()
    
    print("当前配置:")
    print(f"  📂 输入目录: {CONFIG['video_input_dir']}")
    print(f"  📂 输出目录: {CONFIG['output_dir']}")
    print(f"  📂 模型目录: {CONFIG['model_dir']}")
    print(f"  🤖 模型: {CONFIG['model_id']}")
    print(f"  🎲 遮挡方式: {'随机' if CONFIG['random_occlusion'] else '顺序'}")
    print(f"  🎬 输出规格: {CONFIG['num_frames']}帧 @ {CONFIG['fps']}fps")
    print(f"  💾 数据类型: {CONFIG['dtype']}")
    
    print("\n" + "="*80)
    response = input("确认配置无误，按Enter键开始处理 (或输入 q 退出): ")
    if response.lower() == 'q':
        print("已取消")
        return
    
    processor.setup_model()
    processor.process_all_videos()
    
    print("🎉 程序结束!\n")


if __name__ == "__main__":
    main()