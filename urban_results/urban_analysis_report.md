# Urban UrbanVideo-Bench 评估结果分析报告

## 实验概述
本实验在UrbanVideo-Bench数据集上评估了视觉-语言模型在不同视觉条件下的鲁棒性。
使用urban前缀的目录结构，测试了三种条件：原始视频、雨雾效果视频、遮挡效果视频。
这是一个手动下载设置的评估实验。

## 总体结果
- **urban_original测试**:
  - 准确率: 0.3571
  - 问题总数: 14
  - 正确答案: 5
  - 答案提取率: 1.0000
- **urban_rain_fog测试**:
  - 准确率: 0.2857
  - 问题总数: 14
  - 正确答案: 4
  - 答案提取率: 0.7857
- **urban_occlusion测试**:
  - 准确率: 0.5000
  - 问题总数: 14
  - 正确答案: 7
  - 答案提取率: 1.0000

## 性能对比（相对于原始视频）
- **urban_rain_fog**:
  - 绝对下降: 0.0714
  - 相对下降: 20.00%
- **urban_occlusion**:
  - 绝对下降: -0.1429
  - 相对下降: -40.00%

## 按问题类别分析
### Action Generation
- urban_original: 0.5000 (2/4)
- urban_rain_fog: 0.3333 (1/3)
- urban_occlusion: 0.5000 (2/4)

### Counterfactual
- urban_original: 0.0000 (0/1)
- urban_rain_fog: 0.0000 (0/1)
- urban_occlusion: 0.0000 (0/1)

### Landmark Position
- urban_original: 0.5000 (2/4)
- urban_rain_fog: 0.0000 (0/2)
- urban_occlusion: 0.5000 (2/4)

### Object Recall
- urban_original: 0.0000 (0/1)
- urban_rain_fog: 0.0000 (0/1)
- urban_occlusion: 0.0000 (0/1)

### Progress Evaluation
- urban_original: 0.2500 (1/4)
- urban_rain_fog: 0.7500 (3/4)
- urban_occlusion: 0.7500 (3/4)

## 按视频分析
### AerialVLN_0_30BUDKLTXMTIKYKFIPPOXCMISXQE5S.mp4
- urban_original: 0.0000 (0/2)
- urban_rain_fog: 0.0000 (0/2)
- urban_occlusion: 0.0000 (0/2)

### AerialVLN_0_35H6S234SJYE7JR0C76QLOU436G65H_0.mp4
- urban_original: 0.3333 (1/3)
- urban_rain_fog: 0.6667 (2/3)
- urban_occlusion: 0.6667 (2/3)

### AerialVLN_0_35H6S234SJYE7JR0C76QLOU436G65H_1.mp4
- urban_original: 0.3333 (1/3)
- urban_rain_fog: 0.3333 (1/3)
- urban_occlusion: 0.3333 (1/3)

### AerialVLN_0_35H6S234SJYE7JR0C76QLOU436G65H_2.mp4
- urban_original: 0.3333 (1/3)
- urban_rain_fog: 1.0000 (1/1)
- urban_occlusion: 0.6667 (2/3)

### AerialVLN_0_35H6S234SJYE7JR0C76QLOU436G65H_3.mp4
- urban_original: 0.6667 (2/3)
- urban_rain_fog: 0.0000 (0/2)
- urban_occlusion: 0.6667 (2/3)

## 错误分析示例
### urban_original 错误示例（前3个）
**错误 1:**
- 问题ID: 2104
- 正确答案: E
- 模型答案: A
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 2:**
- 问题ID: 2105
- 正确答案: A
- 模型答案: B
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 3:**
- 问题ID: 2107
- 正确答案: C
- 模型答案: E
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

### urban_rain_fog 错误示例（前3个）
**错误 1:**
- 问题ID: 2105
- 正确答案: A
- 模型答案: NO_ANSWER
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 2:**
- 问题ID: 2106
- 正确答案: E
- 模型答案: NO_ANSWER
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 3:**
- 问题ID: 2107
- 正确答案: C
- 模型答案: A
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

### urban_occlusion 错误示例（前3个）
**错误 1:**
- 问题ID: 2105
- 正确答案: A
- 模型答案: B
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 2:**
- 问题ID: 2107
- 正确答案: C
- 模型答案: A
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...

**错误 3:**
- 问题ID: 171
- 正确答案: B
- 模型答案: D
- 问题: Please assume the role of an agent. The video represents your egocentric observations from the past ...
