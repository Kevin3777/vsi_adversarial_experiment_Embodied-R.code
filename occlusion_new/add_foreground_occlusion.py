"""
原视频前景遮挡添加工具
在保持原视频内容的基础上，添加前景遮挡效果

使用方法:
    python add_foreground_occlusion.py

作者: AI Assistant
日期: 2025-10-06
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random

# ==================== 配置区域 ====================
CONFIG = {
    'video_input_dir': "/root/autodl-tmp/sida/urban_videos/original",
    'output_dir': "/root/autodl-tmp/sida/urban_videos/with_occlusion",
    'occlusion_assets_dir': "/root/autodl-tmp/sida/occlusion_assets",  # 遮挡物素材目录
    
    # 遮挡效果配置
    'occlusion_types': [
        'tree_branch',      # 树枝
        'building_edge',    # 建筑边缘
        'pole',            # 柱子/杆
        'fence',           # 栅栏
        'motion_blur',     # 运动模糊遮挡
    ],
    
    # 遮挡参数
    'occlusion_opacity': 0.7,        # 遮挡物透明度 (0-1)
    'occlusion_position': 'random',  # 位置: 'left', 'right', 'top', 'random'
    'occlusion_size_ratio': 0.3,     # 遮挡物占画面比例 (0-1)
    'add_motion': True,              # 是否添加运动效果
    'random_occlusion': True,        # 每个视频随机选择遮挡类型
}


class ForegroundOcclusionAdder:
    """前景遮挡添加器"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def create_tree_branch_mask(self, frame_shape, position='left'):
        """创建树枝遮挡效果"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 随机树枝参数
        num_branches = random.randint(3, 6)
        branch_color = (25, 40, 20)  # 深绿色
        
        for _ in range(num_branches):
            if position == 'left':
                x_start = random.randint(0, w//4)
                y_start = random.randint(0, h)
                x_end = random.randint(w//6, w//3)
                y_end = random.randint(0, h)
            elif position == 'right':
                x_start = random.randint(3*w//4, w)
                y_start = random.randint(0, h)
                x_end = random.randint(2*w//3, 5*w//6)
                y_end = random.randint(0, h)
            else:  # top
                x_start = random.randint(0, w)
                y_start = random.randint(0, h//4)
                x_end = random.randint(0, w)
                y_end = random.randint(h//6, h//3)
            
            thickness = random.randint(5, 15)
            cv2.line(overlay, (x_start, y_start), (x_end, y_end), 
                    branch_color, thickness)
            cv2.line(mask, (x_start, y_start), (x_end, y_end), 
                    255, thickness)
            
            # 添加树叶效果
            num_leaves = random.randint(2, 5)
            for _ in range(num_leaves):
                leaf_x = random.randint(min(x_start, x_end), max(x_start, x_end))
                leaf_y = random.randint(min(y_start, y_end), max(y_start, y_end))
                leaf_size = random.randint(10, 30)
                cv2.circle(overlay, (leaf_x, leaf_y), leaf_size, 
                          (30, 50, 25), -1)
                cv2.circle(mask, (leaf_x, leaf_y), leaf_size, 255, -1)
        
        # 添加模糊效果使其更自然
        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return overlay, mask
    
    def create_building_edge_mask(self, frame_shape, position='left'):
        """创建建筑边缘遮挡"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        building_color = (60, 60, 65)  # 灰色建筑
        
        if position == 'left':
            width = random.randint(w//6, w//4)
            cv2.rectangle(overlay, (0, 0), (width, h), building_color, -1)
            cv2.rectangle(mask, (0, 0), (width, h), 255, -1)
        elif position == 'right':
            width = random.randint(w//6, w//4)
            cv2.rectangle(overlay, (w-width, 0), (w, h), building_color, -1)
            cv2.rectangle(mask, (w-width, 0), (w, h), 255, -1)
        else:  # top
            height = random.randint(h//6, h//4)
            cv2.rectangle(overlay, (0, 0), (w, height), building_color, -1)
            cv2.rectangle(mask, (0, 0), (w, height), 255, -1)
        
        # 添加边缘渐变效果
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return overlay, mask
    
    def create_pole_mask(self, frame_shape, position='left'):
        """创建柱子/杆遮挡"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        pole_color = (40, 40, 50)
        
        if position == 'left':
            x = random.randint(w//10, w//5)
        elif position == 'right':
            x = random.randint(4*w//5, 9*w//10)
        else:
            x = random.randint(2*w//5, 3*w//5)
        
        width = random.randint(15, 40)
        cv2.rectangle(overlay, (x-width//2, 0), (x+width//2, h), 
                     pole_color, -1)
        cv2.rectangle(mask, (x-width//2, 0), (x+width//2, h), 
                     255, -1)
        
        # 添加模糊边缘
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return overlay, mask
    
    def create_fence_mask(self, frame_shape):
        """创建栅栏遮挡"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        fence_color = (45, 45, 50)
        
        # 垂直栅栏条
        num_bars = random.randint(5, 10)
        for i in range(num_bars):
            x = int(w * i / num_bars) + random.randint(-20, 20)
            width = random.randint(8, 15)
            cv2.rectangle(overlay, (x-width//2, 0), (x+width//2, h), 
                         fence_color, -1)
            cv2.rectangle(mask, (x-width//2, 0), (x+width//2, h), 
                         255, -1)
        
        # 水平横杆
        for y in [h//3, 2*h//3]:
            cv2.rectangle(overlay, (0, y-10), (w, y+10), fence_color, -1)
            cv2.rectangle(mask, (0, y-10), (w, y+10), 255, -1)
        
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        return overlay, mask
    
    def create_motion_blur_mask(self, frame_shape, position='left'):
        """创建运动模糊遮挡效果（模拟快速移动的物体）"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        blur_color = (30, 30, 35)
        
        if position == 'left':
            x1, y1 = 0, random.randint(h//3, 2*h//3)
            x2, y2 = w//3, y1 + random.randint(-h//4, h//4)
        else:  # right
            x1, y1 = w, random.randint(h//3, 2*h//3)
            x2, y2 = 2*w//3, y1 + random.randint(-h//4, h//4)
        
        thickness = random.randint(40, 80)
        cv2.line(overlay, (x1, y1), (x2, y2), blur_color, thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
        # 强烈的模糊效果
        overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        return overlay, mask
    
    def apply_occlusion_to_frame(self, frame, occlusion_type, frame_num=0):
        """将遮挡效果应用到帧上"""
        position = self.config['occlusion_position']
        if position == 'random':
            position = random.choice(['left', 'right', 'top'])
        
        # 为运动效果添加时间偏移
        if self.config['add_motion']:
            position_offset = int(10 * np.sin(frame_num * 0.05))
        else:
            position_offset = 0
        
        # 根据类型创建遮挡
        if occlusion_type == 'tree_branch':
            overlay, mask = self.create_tree_branch_mask(frame.shape, position)
        elif occlusion_type == 'building_edge':
            overlay, mask = self.create_building_edge_mask(frame.shape, position)
        elif occlusion_type == 'pole':
            overlay, mask = self.create_pole_mask(frame.shape, position)
        elif occlusion_type == 'fence':
            overlay, mask = self.create_fence_mask(frame.shape)
        elif occlusion_type == 'motion_blur':
            overlay, mask = self.create_motion_blur_mask(frame.shape, position)
        else:
            return frame
        
        # 应用轻微运动
        if position_offset != 0 and occlusion_type != 'fence':
            M = np.float32([[1, 0, position_offset], [0, 1, 0]])
            overlay = cv2.warpAffine(overlay, M, (frame.shape[1], frame.shape[0]))
            mask = cv2.warpAffine(mask, M, (frame.shape[1], frame.shape[0]))
        
        # 混合
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        alpha = mask_3channel * self.config['occlusion_opacity']
        
        result = frame.astype(float) * (1 - alpha) + overlay.astype(float) * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def process_video(self, video_path, output_path, occlusion_type):
        """处理单个视频"""
        print(f"\n处理: {os.path.basename(video_path)}")
        print(f"遮挡类型: {occlusion_type}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        pbar = tqdm(total=total_frames, desc="处理帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用遮挡
            processed_frame = self.apply_occlusion_to_frame(
                frame, occlusion_type, frame_num
            )
            
            out.write(processed_frame)
            frame_num += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"完成: {output_path}")
    
    def process_all_videos(self):
        """批量处理所有视频"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(self.config['video_input_dir']).glob(f'*{ext}'))
            video_files.extend(Path(self.config['video_input_dir']).glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"未找到视频文件: {self.config['video_input_dir']}")
            return
        
        print(f"\n找到 {len(video_files)} 个视频")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}]")
            
            if self.config['random_occlusion']:
                occlusion_type = random.choice(self.config['occlusion_types'])
            else:
                occlusion_type = self.config['occlusion_types'][i % len(self.config['occlusion_types'])]
            
            output_filename = f"{video_path.stem}_occluded.mp4"
            output_path = os.path.join(self.config['output_dir'], output_filename)
            
            self.process_video(str(video_path), output_path, occlusion_type)
        
        print(f"\n所有视频处理完成!")
        print(f"输出目录: {self.config['output_dir']}")


def main():
    print("\n" + "="*80)
    print("原视频前景遮挡添加工具".center(80))
    print("="*80)
    
    processor = ForegroundOcclusionAdder(CONFIG)
    
    print("\n配置:")
    print(f"  输入目录: {CONFIG['video_input_dir']}")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print(f"  遮挡类型: {CONFIG['occlusion_types']}")
    print(f"  透明度: {CONFIG['occlusion_opacity']}")
    print(f"  添加运动: {CONFIG['add_motion']}")
    
    print("\n" + "="*80)
    input("按Enter开始处理...")
    
    processor.process_all_videos()


if __name__ == "__main__":
    main()