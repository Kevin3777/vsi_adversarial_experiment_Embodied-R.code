import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import argparse

class VideoEditor:
    def __init__(self):
        self.rain_intensity = 0.6      # 增加雨的强度
        self.fog_intensity = 0.4       # 增加雾的强度
        self.weather_darkness = 0.3    # 添加阴雨天的暗度
        
    def add_rain_effect(self, frame, frame_num):
        """添加密集雨的效果"""
        h, w = frame.shape[:2]
        
        # 创建多层雨效果
        rain_layer = np.zeros((h, w), dtype=np.uint8)
        
        # 使用frame_num确保雨滴的连续性，但更新更频繁
        np.random.seed((frame_num // 1) % 1000)  # 每帧都更新，更动态
        
        # 生成密集的雨线 - 大幅增加数量
        num_raindrops = random.randint(300, 500)  # 增加到300-500条
        for _ in range(num_raindrops):
            x = random.randint(0, w-1)
            y_start = random.randint(-50, h//4)  # 从更高处开始
            length = random.randint(40, 80)      # 更长的雨线
            y_end = min(y_start + length, h+20)  # 可以延伸到屏幕外
            
            # 更明显的倾斜角度模拟强风
            angle_offset = random.randint(-8, 8)
            x_end = min(max(x + angle_offset, -10), w+10)
            
            # 不同粗细的雨线
            thickness = random.choice([1, 1, 1, 2])  # 大部分是1像素，少数是2像素
            cv2.line(rain_layer, (x, y_start), (x_end, y_end), 255, thickness)
        
        # 添加雨滴效果（小圆点模拟远处的雨）
        for _ in range(200):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.choice([1, 2])
            cv2.circle(rain_layer, (x, y), radius, 180, -1)
        
        # 添加运动模糊效果
        kernel = cv2.getRotationMatrix2D((0, 0), 15, 1)  # 15度角的运动模糊
        rain_layer = cv2.warpAffine(rain_layer, kernel[:2], (w, h))
        rain_layer = cv2.GaussianBlur(rain_layer, (1, 5), 0)  # 垂直方向模糊
        
        # 转换为3通道并应用更强的效果
        rain_overlay = cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(frame, 1-self.rain_intensity, rain_overlay, self.rain_intensity, 0)
        
        return result
    
    def add_fog_effect(self, frame, frame_num):
        """添加全屏浓雾和水汽效果"""
        h, w = frame.shape[:2]
        
        # 基础雾强度，动态变化
        fog_base_intensity = self.fog_intensity + 0.15 * np.sin(frame_num * 0.03)
        fog_base_intensity = max(0.25, min(0.6, fog_base_intensity))
        
        # 创建多层雾效果
        # 第一层：基础雾层（全屏覆盖）
        base_fog = np.full((h, w, 3), 215, dtype=np.uint8)  # 浅灰白色
        
        # 第二层：水汽效果（更细腻的噪声）
        # 创建多尺度噪声来模拟水汽
        noise_large = np.random.normal(0, 25, (h//4, w//4, 3))
        noise_large = cv2.resize(noise_large, (w, h))
        
        noise_medium = np.random.normal(0, 15, (h//2, w//2, 3))
        noise_medium = cv2.resize(noise_medium, (w, h))
        
        noise_fine = np.random.normal(0, 8, (h, w, 3))
        
        # 组合噪声
        combined_noise = noise_large + noise_medium + noise_fine
        
        # 应用噪声到雾层
        fog_with_noise = np.clip(base_fog + combined_noise, 150, 255).astype(np.uint8)
        
        # 添加动态流动效果
        time_factor = frame_num * 0.1
        flow_x = int(10 * np.sin(time_factor * 0.2))
        flow_y = int(5 * np.cos(time_factor * 0.15))
        
        # 创建变换矩阵来模拟雾的流动
        M = np.float32([[1, 0, flow_x], [0, 1, flow_y]])
        fog_with_noise = cv2.warpAffine(fog_with_noise, M, (w, h), borderMode=cv2.BORDER_WRAP)
        
        # 多次高斯模糊创建更柔和的效果
        fog_layer = cv2.GaussianBlur(fog_with_noise, (31, 31), 0)
        fog_layer = cv2.GaussianBlur(fog_layer, (21, 21), 0)
        
        # 创建不均匀的雾密度（近处薄，远处厚）
        # 创建渐变mask
        gradient_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            # 上半部分雾更浓（模拟远景）
            intensity = 0.7 + 0.3 * (i / h)
            gradient_mask[i, :] = intensity
        
        gradient_mask = np.stack([gradient_mask] * 3, axis=2)
        
        # 应用渐变
        fog_layer = fog_layer.astype(np.float32) * gradient_mask
        fog_layer = np.clip(fog_layer, 0, 255).astype(np.uint8)
        
        # 混合雾效果到原图像
        result = cv2.addWeighted(frame, 1-fog_base_intensity, fog_layer, fog_base_intensity, 0)
        
        # 添加整体的湿润感（降低饱和度，增加蓝色调）
        result = result.astype(np.float32)
        # 轻微的蓝色调
        result[:,:,0] *= 1.05  # 增加蓝色通道
        result[:,:,1] *= 0.98  # 轻微降低绿色
        result[:,:,2] *= 0.95  # 降低红色
        
        # 降低整体亮度模拟阴天
        result *= (1 - self.weather_darkness)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def add_occlusion_effect(self, frame, frame_num):
        """添加完全不透明的动态遮挡效果"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # 时间因子
        time_factor = frame_num * 0.08
        
        # 遮挡物1：从左到右移动的大型不透明块
        block1_width = w // 4
        block1_height = h // 2
        block1_x = int((w + block1_width) * (0.5 + 0.5 * np.sin(time_factor * 0.4))) - block1_width
        block1_y = h // 4
        
        if -block1_width < block1_x < w:
            x1 = max(0, block1_x)
            x2 = min(w, block1_x + block1_width)
            y1 = max(0, block1_y)
            y2 = min(h, block1_y + block1_height)
            # 完全不透明的黑色块
            result[y1:y2, x1:x2] = [20, 20, 20]
        
        # 遮挡物2：垂直移动的横条（模拟经过的物体）
        bar_height = 60
        bar_y = int(h * (0.5 + 0.4 * np.sin(time_factor * 0.6)))
        bar_y = max(0, min(h - bar_height, bar_y))
        # 完全不透明的深灰色条
        result[bar_y:bar_y + bar_height, :] = [35, 35, 35]
        
        # 遮挡物3：大型圆形遮挡（模拟头部或大型物体）
        if frame_num % 40 < 25:  # 每40帧中显示25帧
            center_x = w//2 + int(w//4 * np.sin(time_factor * 0.8))
            center_y = h//3 + int(h//4 * np.cos(time_factor * 0.5))
            radius = int(min(w, h) * 0.15)  # 更大的半径
            # 完全不透明的圆形
            cv2.circle(result, (center_x, center_y), radius, (25, 25, 25), -1)
        
        # 遮挡物4：边缘的不规则遮挡（模拟手部或其他物体）
        # 左边缘遮挡
        edge_width = int(w * 0.12 * (1 + 0.3 * np.sin(time_factor * 1.2)))
        result[:, :edge_width] = [15, 15, 15]
        
        # 右边缘遮挡
        if frame_num % 60 < 30:  # 间歇性出现
            right_edge_width = int(w * 0.08 * (1 + 0.5 * np.cos(time_factor * 0.9)))
            result[:, -right_edge_width:] = [18, 18, 18]
        
        # 遮挡物5：顶部下拉的遮挡（模拟帽檐或其他）
        top_height = int(h * 0.15 * (1 + 0.2 * np.sin(time_factor * 0.7)))
        result[:top_height, :] = [22, 22, 22]
        
        # 遮挡物6：随机位置的小型遮挡块（模拟飞行物体或碎片）
        np.random.seed(frame_num // 5)  # 每5帧更新一次位置
        for i in range(8):  # 增加小遮挡物的数量
            small_size = random.randint(30, 80)
            pos_x = random.randint(0, w - small_size)
            pos_y = random.randint(0, h - small_size)
            
            # 随机形状的遮挡
            if random.choice([True, False]):
                # 矩形遮挡
                result[pos_y:pos_y+small_size, pos_x:pos_x+small_size] = [30, 30, 30]
            else:
                # 圆形遮挡
                cv2.circle(result, (pos_x + small_size//2, pos_y + small_size//2), 
                          small_size//2, (28, 28, 28), -1)
        
        return result
    
    def process_video(self, input_path, output_path, effect_type="rain_fog"):
        """处理单个视频"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return False
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用对应的效果
            if effect_type == "rain_fog":
                frame = self.add_rain_effect(frame, frame_count)
                frame = self.add_fog_effect(frame, frame_count)
            elif effect_type == "occlusion":
                frame = self.add_occlusion_effect(frame, frame_count)
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"Saved: {output_path}")
        return True
    
    def batch_process(self, input_dir, output_dir, num_videos=50):
        """批量处理视频"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rain_fog"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "occlusion"), exist_ok=True)
        
        # 获取视频文件列表
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if len(video_files) < num_videos:
            print(f"Warning: Only found {len(video_files)} videos, less than requested {num_videos}")
            num_videos = len(video_files)
        
        # 随机选择视频
        selected_videos = random.sample(video_files, num_videos)
        
        print(f"Processing {num_videos} videos...")
        
        for i, video_file in enumerate(selected_videos):
            print(f"\n=== Processing video {i+1}/{num_videos}: {video_file} ===")
            
            input_path = os.path.join(input_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            
            # 生成雨雾版本
            rain_fog_output = os.path.join(output_dir, "rain_fog", f"{base_name}_rain_fog.mp4")
            success1 = self.process_video(input_path, rain_fog_output, "rain_fog")
            
            # 生成遮挡版本
            occlusion_output = os.path.join(output_dir, "occlusion", f"{base_name}_occlusion.mp4")
            success2 = self.process_video(input_path, occlusion_output, "occlusion")
            
            if success1 and success2:
                print(f"✅ Successfully processed {video_file}")
            else:
                print(f"❌ Failed to process {video_file}")

def main():
    parser = argparse.ArgumentParser(description='Add weather effects and occlusions to videos')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed videos')
    parser.add_argument('--num_videos', type=int, default=50,
                       help='Number of videos to process (default: 50)')
    parser.add_argument('--rain_intensity', type=float, default=0.6,
                       help='Rain intensity (0.0-1.0, default: 0.6)')
    parser.add_argument('--fog_intensity', type=float, default=0.4,
                       help='Fog intensity (0.0-1.0, default: 0.4)')
    parser.add_argument('--weather_darkness', type=float, default=0.3,
                       help='Weather darkness factor (0.0-1.0, default: 0.3)')
    
    args = parser.parse_args()
    
    # 创建编辑器
    editor = VideoEditor()
    editor.rain_intensity = args.rain_intensity
    editor.fog_intensity = args.fog_intensity
    editor.weather_darkness = args.weather_darkness
    
    # 批量处理
    editor.batch_process(args.input_dir, args.output_dir, args.num_videos)
    
    print(f"\n🎉 All done! Processed videos saved to {args.output_dir}")
    print(f"📁 Rain/fog videos: {args.output_dir}/rain_fog/")
    print(f"📁 Occlusion videos: {args.output_dir}/occlusion/")

if __name__ == "__main__":
    main()