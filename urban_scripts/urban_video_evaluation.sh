#!/bin/bash
# 适配urban目录结构的UrbanVideo-Bench完整评估脚本
# 对原始视频、雨雾版本、遮挡版本进行完整的视觉理解和推理评估

set -e  # 遇到错误立即退出

# 配置参数 - 使用urban前缀的目录结构
WORK_DIR="${1:-/path/to/your/urban_experiment}"
EMBODIEDR_DIR="${2:-/path/to/Embodied-R.code}"
MODEL_7B="${3:-/path/to/Qwen2-VL-7B-Instruct}"
MODEL_3B="${4:-/path/to/Qwen2.5-VL-3B-Instruct}"

# 检查参数
if [ "$#" -lt 4 ]; then
    echo "用法: $0 <工作目录> <Embodied-R代码目录> <7B模型路径> <3B模型路径>"
    echo "示例: $0 /root/urban_experiment /root/Embodied-R.code /root/models/Qwen2-VL-7B /root/models/Qwen2.5-VL-3B"
    exit 1
fi

echo "=== Urban结构UrbanVideo-Bench完整评估 ==="
echo "工作目录: $WORK_DIR"
echo "Embodied-R目录: $EMBODIEDR_DIR" 
echo "7B模型: $MODEL_7B"
echo "3B模型: $MODEL_3B"

# 检查必要目录和文件
check_urban_requirements() {
    echo "检查urban目录结构和必要文件..."
    
    local missing=0
    
    if [ ! -d "$WORK_DIR" ]; then
        echo "❌ 工作目录不存在: $WORK_DIR"
        missing=1
    fi
    
    if [ ! -d "$EMBODIEDR_DIR" ]; then
        echo "❌ Embodied-R代码目录不存在: $EMBODIEDR_DIR"
        missing=1
    fi
    
    if [ ! -d "$MODEL_7B" ]; then
        echo "❌ 7B模型目录不存在: $MODEL_7B"
        missing=1
    fi
    
    if [ ! -d "$MODEL_3B" ]; then
        echo "❌ 3B模型目录不存在: $MODEL_3B"
        missing=1
    fi
    
    # 检查urban目录结构
    if [ ! -d "$WORK_DIR/urban_videos/original" ]; then
        echo "❌ urban原始视频目录不存在: $WORK_DIR/urban_videos/original"
        missing=1
    fi
    
    if [ ! -f "$WORK_DIR/urban_annotations/test_data.json" ]; then
        echo "❌ urban标注文件不存在: $WORK_DIR/urban_annotations/test_data.json"
        echo "请先运行urban数据准备脚本"
        missing=1
    fi
    
    # 检查实验总结文件
    if [ ! -f "$WORK_DIR/urban_experiment_summary.json" ]; then
        echo "⚠️  实验总结文件不存在，但可以继续"
    fi
    
    if [ $missing -eq 1 ]; then
        echo "请确保所有必要文件存在后重新运行"
        exit 1
    fi
    
    echo "✅ urban目录结构检查通过"
}

# 过滤标注文件，适配urban结构
filter_urban_annotations() {
    local video_folder=$1
    local output_file=$2
    local test_type=$3
    
    echo "过滤urban标注文件，生成${test_type}测试数据..."
    
    python3 -c "
import json
import os
from pathlib import Path

# 读取urban标注文件
with open('$WORK_DIR/urban_annotations/test_data.json', 'r') as f:
    annotations = json.load(f)

print(f'urban标注文件包含 {len(annotations)} 个问题')

# 检查urban视频文件夹中实际存在的视频
video_folder = Path('$video_folder')
test_type = '$test_type'

if not video_folder.exists():
    print(f'错误: urban视频文件夹不存在: {video_folder}')
    exit(1)

# 获取视频文件列表
video_files = [f.name for f in video_folder.glob('*.mp4')]
print(f'urban视频文件夹中找到 {len(video_files)} 个视频文件')

# 建立原始视频ID到实际文件名的映射
id_to_filename_map = {}

for video_file in video_files:
    if test_type == 'original':
        # 原始视频：直接映射
        original_name = video_file
        id_to_filename_map[original_name] = video_file
    else:
        # 效果视频：去掉后缀映射到原始名称
        if video_file.endswith(f'_{test_type}.mp4'):
            original_name = video_file.replace(f'_{test_type}.mp4', '.mp4')
            id_to_filename_map[original_name] = video_file

print(f'建立了 {len(id_to_filename_map)} 个urban视频映射')

# 过滤标注，只保留存在的视频的问题，并更新video_id
filtered_annotations = []
for ann in annotations:
    original_video_id = ann.get('video_id', '')
    
    if original_video_id in id_to_filename_map:
        # 创建新的标注记录，更新video_id为实际文件名
        new_ann = ann.copy()
        new_ann['video_id'] = id_to_filename_map[original_video_id]  # 更新为实际文件名
        new_ann['original_video_id'] = original_video_id   # 保留原始video_id
        filtered_annotations.append(new_ann)

print(f'过滤后保留 {len(filtered_annotations)} 个问题')

# 统计每个视频的问题数量
video_question_count = {}
for ann in filtered_annotations:
    original_video_id = ann.get('original_video_id', ann.get('video_id', ''))
    video_question_count[original_video_id] = video_question_count.get(original_video_id, 0) + 1

print(f'urban {test_type}测试每个视频的问题数量:')
for video, count in sorted(video_question_count.items()):
    print(f'  {video}: {count} 个问题')

# 保存过滤后的标注
os.makedirs(os.path.dirname('$output_file'), exist_ok=True)
with open('$output_file', 'w') as f:
    json.dump(filtered_annotations, f, indent=2)

print(f'过滤后的urban标注保存到: $output_file')
"
}

# 运行单个urban测试
run_urban_test() {
    local test_type=$1
    local video_folder="$WORK_DIR/urban_videos/$test_type"
    local result_folder="$WORK_DIR/urban_results/$test_type"
    
    echo ""
    echo "=== 开始urban ${test_type}测试 ==="
    echo "视频目录: $video_folder"
    echo "结果目录: $result_folder"
    
    # 检查视频目录
    if [ ! -d "$video_folder" ] || [ -z "$(ls -A "$video_folder" 2>/dev/null)" ]; then
        echo "⚠️  ${test_type}视频目录不存在或为空，跳过测试"
        return 1
    fi
    
    # 统计视频数量
    video_count=$(ls "$video_folder"/*.mp4 2>/dev/null | wc -l)
    echo "找到 $video_count 个视频文件"
    
    if [ $video_count -eq 0 ]; then
        echo "⚠️  ${test_type}目录中没有找到视频文件，跳过测试"
        return 1
    fi
    
    # 显示视频文件示例
    echo "视频文件示例:"
    ls "$video_folder"/*.mp4 | head -3
    
    # 创建结果目录
    mkdir -p "$result_folder"
    
    # 过滤标注文件
    local filtered_annotation="$result_folder/filtered_test_data.json"
    filter_urban_annotations "$video_folder" "$filtered_annotation" "$test_type"
    
    if [ ! -f "$filtered_annotation" ]; then
        echo "❌ 无法创建过滤后的标注文件"
        return 1
    fi
    
    # 验证过滤结果
    question_count=$(python3 -c "
import json
try:
    with open('$filtered_annotation', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
")
    
    if [ "$question_count" = "0" ]; then
        echo "❌ 过滤后没有有效问题"
        return 1
    fi
    
    echo "将处理 $question_count 个问题"
    
    # 切换到Embodied-R目录
    cd "$EMBODIEDR_DIR"
    
    # 第一阶段：视觉理解
    echo "--- 阶段1: 视觉理解 ---"
    python train/conver_format/VLM_perception_local.py \
      --model_path "$MODEL_7B" \
      --data_paths "$filtered_annotation" \
      --folder_path "$video_folder" \
      --save_path "$result_folder"
    
    if [ $? -ne 0 ]; then
        echo "❌ ${test_type}视觉处理失败"
        return 1
    fi
    
    # 检查生成的文件并标准化名称
    echo "--- 检查生成的文件 ---"
    
    # 寻找生成的视觉理解结果文件
    visual_result_file=""
    for possible_file in "$result_folder/test_data.json" "$result_folder"/*.json; do
        if [ -f "$possible_file" ] && [ "$(basename "$possible_file")" != "filtered_test_data.json" ]; then
            visual_result_file="$possible_file"
            break
        fi
    done
    
    # 如果找到的文件不是标准名称，重命名它
    if [ -n "$visual_result_file" ] && [ "$visual_result_file" != "$result_folder/test_data.json" ]; then
        mv "$visual_result_file" "$result_folder/test_data.json"
        echo "重命名文件为: test_data.json"
    fi
    
    if [ ! -f "$result_folder/test_data.json" ]; then
        echo "❌ ${test_type}视觉处理没有生成预期的结果文件"
        echo "目录内容:"
        ls -la "$result_folder/"
        return 1
    fi
    
    # 验证生成的视觉理解文件
    processed_count=$(python3 -c "
import json
try:
    with open('$result_folder/test_data.json', 'r') as f:
        data = json.load(f)
    print(len(data))
except Exception as e:
    print(0)
" 2>/dev/null)
    
    echo "视觉理解处理的问题数量: $processed_count"
    
    if [ "$processed_count" = "0" ]; then
        echo "❌ ${test_type}生成的视觉理解文件为空或格式错误"
        return 1
    fi
    
    # 第二阶段：推理
    echo "--- 阶段2: 推理 ---"
    cd infer
    
    python batch_inference.py \
      --model "$MODEL_3B" \
      --input_file "$result_folder/test_data.json" \
      --output_file "$WORK_DIR/urban_results/urban_${test_type}_results.json" \
      --batch_size 1 \
      --max_tokens 3096
    
    local inference_result=$?
    cd ..
    
    if [ $inference_result -ne 0 ]; then
        echo "❌ ${test_type}推理失败"
        return 1
    fi
    
    # 验证最终结果
    if [ -f "$WORK_DIR/urban_results/urban_${test_type}_results.json" ]; then
        result_count=$(python3 -c "
import json
try:
    with open('$WORK_DIR/urban_results/urban_${test_type}_results.json', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
" 2>/dev/null)
        echo "✅ ${test_type}测试完成，生成 $result_count 个结果"
        
        # 创建测试总结
        echo "创建${test_type}测试总结..."
        python3 -c "
import json
from datetime import datetime
import re

# 读取结果
with open('$WORK_DIR/urban_results/urban_${test_type}_results.json', 'r') as f:
    results = json.load(f)

# 统计准确率
correct = 0
total = len(results)
category_stats = {}

def extract_answer(response):
    '''提取模型答案'''
    if not response:
        return None
    
    # 寻找答案格式
    answer_match = re.search(r'<answer>([ABCDE])</answer>', response)
    if answer_match:
        return answer_match.group(1)
    
    # 寻找常见格式
    patterns = [
        r'答案是?\\\s*([ABCDE])',
        r'选择\\\s*([ABCDE])',
        r'答案：\\\s*([ABCDE])',
        r'([ABCDE])(?=\\\s*[\.。])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # 最后尝试找单个字母
    letter_match = re.search(r'[ABCDE]', response)
    if letter_match:
        return letter_match.group(0)
    
    return None

for result in results:
    ground_truth = result.get('answer', '').strip()
    model_response = result.get('content', '')
    predicted = extract_answer(model_response)
    
    is_correct = (predicted == ground_truth) if predicted else False
    if is_correct:
        correct += 1
    
    # 按类别统计
    category = result.get('question_category', 'unknown')
    if category not in category_stats:
        category_stats[category] = {'correct': 0, 'total': 0}
    category_stats[category]['total'] += 1
    if is_correct:
        category_stats[category]['correct'] += 1

accuracy = correct / total if total > 0 else 0

summary = {
    'test_type': 'urban_$test_type',
    'total_questions': total,
    'correct_answers': correct,
    'accuracy': round(accuracy, 4),
    'category_breakdown': {},
    'timestamp': datetime.now().isoformat()
}

for category, stats in category_stats.items():
    cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    summary['category_breakdown'][category] = {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': round(cat_accuracy, 4)
    }

# 保存总结
with open('$WORK_DIR/urban_results/urban_${test_type}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'urban ${test_type}测试总结:')
print(f'  总问题数: {total}')
print(f'  正确答案: {correct}')
print(f'  准确率: {accuracy:.4f}')
print(f'总结保存到: $WORK_DIR/urban_results/urban_${test_type}_summary.json')
"
        return 0
    else
        echo "❌ ${test_type}测试失败，未生成结果文件"
        return 1
    fi
}

# 主程序开始
echo "开始Urban结构UrbanVideo-Bench评估..."

# 检查必要条件
check_urban_requirements

# 显示实验信息
if [ -f "$WORK_DIR/urban_experiment_summary.json" ]; then
    echo ""
    echo "Urban实验信息:"
    python3 -c "
import json
with open('$WORK_DIR/urban_experiment_summary.json', 'r') as f:
    summary = json.load(f)
print(f\"数据集: {summary['experiment_info']['dataset']}\")
print(f\"设置方法: {summary['experiment_info']['setup_method']}\")
print(f\"选择视频数: {summary['experiment_info']['selected_videos']}\")
print(f\"总问题数: {summary['experiment_info']['total_questions']}\")
print('视频列表:')
for i, video in enumerate(summary['videos'], 1):
    print(f'  {i}. {video}')
"
fi

# 创建总结果目录
mkdir -p "$WORK_DIR/urban_results"

# 运行三种测试
test_results=()

echo ""
echo "开始三种条件的urban评估测试..."

# 测试1: 原始视频
if run_urban_test "original"; then
    echo "✅ 原始视频测试成功"
    test_results+=("original:success")
else
    echo "❌ 原始视频测试失败"
    test_results+=("original:failed")
fi

# 测试2: 雨雾视频
if run_urban_test "rain_fog"; then
    echo "✅ 雨雾视频测试成功"
    test_results+=("rain_fog:success")
else
    echo "❌ 雨雾视频测试失败"
    test_results+=("rain_fog:failed")
fi

# 测试3: 遮挡视频
if run_urban_test "occlusion"; then
    echo "✅ 遮挡视频测试成功"
    test_results+=("occlusion:success")
else
    echo "❌ 遮挡视频测试失败"
    test_results+=("occlusion:failed")
fi

# 生成最终报告
echo ""
echo "=== 生成最终Urban评估报告 ==="

python3 -c "
import json
import os
from datetime import datetime

# 收集所有测试结果
test_types = ['original', 'rain_fog', 'occlusion']
final_report = {
    'experiment_info': {
        'dataset': 'UrbanVideo-Bench',
        'directory_structure': 'urban_prefixed',
        'test_types': test_types,
        'timestamp': datetime.now().isoformat()
    },
    'results': {},
    'comparison': {}
}

# 读取每个测试的总结
for test_type in test_types:
    summary_file = f'$WORK_DIR/urban_results/urban_{test_type}_summary.json'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        final_report['results'][test_type] = summary
    else:
        final_report['results'][test_type] = {'status': 'failed'}

# 计算性能对比
if 'original' in final_report['results'] and 'accuracy' in final_report['results']['original']:
    baseline_acc = final_report['results']['original']['accuracy']
    
    for test_type in ['rain_fog', 'occlusion']:
        if test_type in final_report['results'] and 'accuracy' in final_report['results'][test_type]:
            test_acc = final_report['results'][test_type]['accuracy']
            performance_drop = baseline_acc - test_acc
            relative_drop = (performance_drop / baseline_acc * 100) if baseline_acc > 0 else 0
            
            final_report['comparison'][test_type] = {
                'absolute_drop': round(performance_drop, 4),
                'relative_drop_percent': round(relative_drop, 2)
            }

# 保存最终报告
with open('$WORK_DIR/urban_results/urban_final_evaluation_report.json', 'w') as f:
    json.dump(final_report, f, indent=2)

# 打印总结
print('=== Urban UrbanVideo-Bench评估完成 ===')
print()

for test_type in test_types:
    if test_type in final_report['results']:
        result = final_report['results'][test_type]
        if 'accuracy' in result:
            print(f'URBAN {test_type.upper()}测试:')
            print(f'  准确率: {result[\"accuracy\"]:.4f}')
            print(f'  问题数: {result[\"total_questions\"]}')
        else:
            print(f'URBAN {test_type.upper()}测试: 失败')
    print()

if 'original' in final_report['results'] and 'accuracy' in final_report['results']['original']:
    print('性能对比 (相对于原始视频):')
    for test_type, comparison in final_report['comparison'].items():
        print(f'  {test_type}: -{comparison[\"absolute_drop\"]:.4f} ({comparison[\"relative_drop_percent\"]:.2f}% 下降)')

print(f'\\n详细报告保存到: $WORK_DIR/urban_results/urban_final_evaluation_report.json')
"

echo ""
echo "=== Urban测试完成总结 ==="
echo "测试结果:"
for result in "${test_results[@]}"; do
    IFS=':' read -r test_type status <<< "$result"
    if [ "$status" = "success" ]; then
        echo "  ✅ urban_$test_type"
    else
        echo "  ❌ urban_$test_type"
    fi
done

echo ""
echo "结果文件位置:"
echo "  - 详细结果: $WORK_DIR/urban_results/"
echo "  - 最终报告: $WORK_DIR/urban_results/urban_final_evaluation_report.json"

if [ -f "$WORK_DIR/urban_results/urban_final_evaluation_report.json" ]; then
    echo ""
    echo "🎉 Urban UrbanVideo-Bench评估成功完成！"
    echo "现在可以运行结果分析脚本查看视觉-语言模型在不同视觉条件下的鲁棒性。"
else
    echo ""
    echo "⚠️  评估过程中出现问题，请检查日志获取详细信息。"
fi