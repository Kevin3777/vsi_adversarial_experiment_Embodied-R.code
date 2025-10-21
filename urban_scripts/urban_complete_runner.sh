#!/bin/bash
# Urban UrbanVideo-Bench 完整实验执行脚本
# 一键运行整个评估流程：数据准备 -> 效果生成 -> 评估 -> 分析

set -e  # 遇到错误立即退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(pwd)"

# 默认路径配置（请根据实际情况修改）
DEFAULT_MCQ_PATH="$WORK_DIR/urban_data/MCQ.parquet"
DEFAULT_VIDEOS_DIR="$WORK_DIR/urban_videos/original"
DEFAULT_EMBODIEDR_DIR="/root/autodl-tmp/sida/Embodied-R.code"
DEFAULT_MODEL_7B="/root/autodl-tmp/sida/model/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
DEFAULT_MODEL_3B="/root/autodl-tmp/sida/model/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Urban UrbanVideo-Bench 完整实验执行脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    --mcq-path PATH         MCQ.parquet文件路径 (默认: $DEFAULT_MCQ_PATH)
    --videos-dir PATH       手动下载的视频目录 (默认: $DEFAULT_VIDEOS_DIR)
    --embodiedr-dir PATH    Embodied-R代码目录 (默认: $DEFAULT_EMBODIEDR_DIR)
    --model-7b PATH         7B模型路径 (默认: $DEFAULT_MODEL_7B)
    --model-3b PATH         3B模型路径 (默认: $DEFAULT_MODEL_3B)
    --skip-data-prep        跳过数据准备步骤
    --skip-effects          跳过视频效果生成步骤
    --skip-evaluation       跳过评估步骤
    --skip-analysis         跳过结果分析步骤
    --dry-run              仅显示将要执行的命令，不实际执行

示例:
    # 完整运行（需要先配置路径）
    $0

    # 指定路径运行
    $0 --embodiedr-dir /root/Embodied-R.code --model-7b /root/models/Qwen2-VL-7B

    # 跳过数据准备，从效果生成开始
    $0 --skip-data-prep

    # 仅运行分析步骤
    $0 --skip-data-prep --skip-effects --skip-evaluation

注意：
    1. 确保已手动下载MCQ.parquet和5个视频文件到指定位置
    2. 首次运行前请修改脚本中的默认路径配置
    3. 整个流程大约需要1.5-2.5小时
EOF
}

# 解析命令行参数
MCQ_PATH="$DEFAULT_MCQ_PATH"
VIDEOS_DIR="$DEFAULT_VIDEOS_DIR"
EMBODIEDR_DIR="$DEFAULT_EMBODIEDR_DIR"
MODEL_7B="$DEFAULT_MODEL_7B"
MODEL_3B="$DEFAULT_MODEL_3B"
SKIP_DATA_PREP=false
SKIP_EFFECTS=false
SKIP_EVALUATION=false
SKIP_ANALYSIS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --mcq-path)
            MCQ_PATH="$2"
            shift 2
            ;;
        --videos-dir)
            VIDEOS_DIR="$2"
            shift 2
            ;;
        --embodiedr-dir)
            EMBODIEDR_DIR="$2"
            shift 2
            ;;
        --model-7b)
            MODEL_7B="$2"
            shift 2
            ;;
        --model-3b)
            MODEL_3B="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip-effects)
            SKIP_EFFECTS=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行命令函数
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    log_info "$description"
    echo "执行命令: $cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "[DRY RUN] 将执行: $cmd"
        return 0
    fi
    
    if eval "$cmd"; then
        log_success "$description 完成"
        return 0
    else
        log_error "$description 失败"
        return 1
    fi
}

# 检查必要文件和目录
check_prerequisites() {
    log_info "检查必要条件..."
    
    local missing=0
    
    # 检查Python环境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        missing=1
    fi
    
    # 检查必要的Python包
    for package in pandas opencv-python tqdm matplotlib seaborn; do
        if ! python3 -c "import ${package%%-*}" &> /dev/null; then
            log_warning "Python包 $package 可能未安装"
        fi
    done
    
    if [ "$SKIP_DATA_PREP" = false ]; then
        if [ ! -f "$MCQ_PATH" ]; then
            log_error "MCQ文件不存在: $MCQ_PATH"
            missing=1
        fi
        
        if [ ! -d "$VIDEOS_DIR" ]; then
            log_error "视频目录不存在: $VIDEOS_DIR"
            missing=1
        fi
    fi
    
    if [ "$SKIP_EVALUATION" = false ]; then
        if [ ! -d "$EMBODIEDR_DIR" ]; then
            log_error "Embodied-R目录不存在: $EMBODIEDR_DIR"
            missing=1
        fi
        
        if [ ! -d "$MODEL_7B" ]; then
            log_error "7B模型目录不存在: $MODEL_7B"
            missing=1
        fi
        
        if [ ! -d "$MODEL_3B" ]; then
            log_error "3B模型目录不存在: $MODEL_3B"
            missing=1
        fi
    fi
    
    if [ $missing -eq 1 ]; then
        log_error "必要条件检查失败，请修复后重新运行"
        exit 1
    fi
    
    log_success "必要条件检查通过"
}

# 创建脚本文件
create_scripts() {
    local scripts_dir="$WORK_DIR/urban_scripts"
    mkdir -p "$scripts_dir"
    
    log_info "检查脚本文件..."
    
    # 检查脚本文件是否存在，如果不存在则提示用户
    local required_scripts=(
        "urban_manual_data_prep.py"
        "urban_video_effects.py"
        "urban_video_evaluation.sh"
        "urban_video_analysis.py"
    )
    
    local missing_scripts=0
    for script in "${required_scripts[@]}"; do
        if [ ! -f "$scripts_dir/$script" ]; then
            log_warning "脚本文件不存在: $scripts_dir/$script"
            missing_scripts=1
        fi
    done
    
    if [ $missing_scripts -eq 1 ]; then
        log_warning "部分脚本文件缺失，请确保已保存所有必要的脚本到 $scripts_dir 目录"
        log_info "需要的脚本文件："
        for script in "${required_scripts[@]}"; do
            echo "  - $script"
        done
        
        read -p "是否继续？(y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 主执行流程
main() {
    echo "=========================================="
    echo "Urban UrbanVideo-Bench 完整实验执行"
    echo "=========================================="
    echo "工作目录: $WORK_DIR"
    echo "MCQ文件: $MCQ_PATH"
    echo "视频目录: $VIDEOS_DIR"
    echo "Embodied-R目录: $EMBODIEDR_DIR"
    echo "7B模型: $MODEL_7B"
    echo "3B模型: $MODEL_3B"
    echo "=========================================="
    
    # 检查必要条件
    check_prerequisites
    
    # 创建脚本目录
    create_scripts
    
    local start_time=$(date +%s)
    
    # 步骤1: 数据准备
    if [ "$SKIP_DATA_PREP" = false ]; then
        log_info "步骤1/4: 数据准备"
        execute_cmd "python3 urban_scripts/urban_manual_data_prep.py --mcq_path '$MCQ_PATH' --videos_dir '$VIDEOS_DIR' --work_dir '$WORK_DIR'" "数据准备"
    else
        log_warning "跳过数据准备步骤"
    fi
    
    # 步骤2: 视频效果生成
    if [ "$SKIP_EFFECTS" = false ]; then
        log_info "步骤2/4: 视频效果生成"
        execute_cmd "python3 urban_scripts/urban_video_effects.py --work_dir '$WORK_DIR'" "视频效果生成"
    else
        log_warning "跳过视频效果生成步骤"
    fi
    
    # 步骤3: 模型评估
    if [ "$SKIP_EVALUATION" = false ]; then
        log_info "步骤3/4: 模型评估"
        # 给脚本添加执行权限
        chmod +x urban_scripts/urban_video_evaluation.sh
        execute_cmd "urban_scripts/urban_video_evaluation.sh '$WORK_DIR' '$EMBODIEDR_DIR' '$MODEL_7B' '$MODEL_3B'" "模型评估"
    else
        log_warning "跳过模型评估步骤"
    fi
    
    # 步骤4: 结果分析
    if [ "$SKIP_ANALYSIS" = false ]; then
        log_info "步骤4/4: 结果分析"
        execute_cmd "python3 urban_scripts/urban_video_analysis.py --results_dir '$WORK_DIR/urban_results'" "结果分析"
    else
        log_warning "跳过结果分析步骤"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo "=========================================="
    log_success "Urban UrbanVideo-Bench 实验完成！"
    echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
    echo ""
    echo "结果文件位置："
    echo "  - 实验总结: $WORK_DIR/urban_experiment_summary.json"
    echo "  - 评估结果: $WORK_DIR/urban_results/"
    echo "  - 分析报告: $WORK_DIR/urban_results/urban_analysis_report.md"
    echo "  - 可视化图表: $WORK_DIR/urban_results/urban_analysis_charts.png"
    echo "=========================================="
}

# 脚本入口
if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN 模式 - 只显示命令，不实际执行"
fi

main

exit 0