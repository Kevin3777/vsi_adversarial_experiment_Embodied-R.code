# VSI Adversarial Experiment: Effects of Visual Interference on Video Understanding

This repository contains comprehensive experiments analyzing the effects of rain/fog and occlusion interference on various video understanding tasks, including object recognition, spatial reasoning, and navigation planning.

## 📋 Project Overview

This research investigates how different types of visual interference affect AI models' performance on video understanding tasks across indoor and outdoor environments. The study compares three visual conditions:

- **Original videos** (no processing)
- **Rain/fog processed videos** (added atmospheric effects)
- **Occlusion processed videos** (added random shadow occlusion)

## 🎯 Key Findings

### Task Complexity Determines Interference Vulnerability

| Task Type | Original | Rain/Fog | Occlusion | Key Insight |
|-----------|----------|-----------|-----------|-------------|
| **Easy Spatial Reasoning** | 100% ✅ | 100% ✅ | 100% ✅ | Universal robustness to interference |
| **Route Planning** | 100% ✅ | 100% ✅ | 100% ✅ | Sequential structure provides protection |
| **Medium Spatial Reasoning** | 100% ✅ | 0% ❌ | 0% ❌ | Reasoning synthesis failures occur |
| **Object Counting** | 100% ✅ | 100% ✅ | 75% ⚠️ | Occlusion causes detection failures |
| **Object Size Estimation** | 96.8% ✅ | 96.8% ✅ | 0% ❌ | Measurement requires higher visual clarity |
| **Progress Evaluation** | 100% ✅ | 0% ❌ | 0% ❌ | Temporal sequencing highly vulnerable |
| **Action Generation** | 100% ✅ | 0% ❌ | 0% ❌ | Vertical movement assessment fails |

### Interference Type Characteristics

#### 🌧️ Rain/Fog Effects

- **Primary Impact**: Global visual degradation
- **Failure Mode**: Systematic misinterpretation
- **Vulnerable Tasks**: Temporal reasoning, vertical spatial awareness
- **Key Finding**: Can cause "detection inversion" - correct observations lead to opposite conclusions

#### 🎭 Occlusion Effects

- **Primary Impact**: Local information loss
- **Failure Mode**: Reasoning abandonment or regression
- **Vulnerable Tasks**: Tasks requiring continuous observation
- **Key Finding**: 71% frame-level errors can still yield correct reasoning with proper evidence weighting

## 🏗️ Project Structure

```
sida/
├── Embodied-R.code/          # Core experiment framework
├── data_50/                  # Dataset (excluded from Git)
├── model/                    # Model files (excluded from Git)
├── occlusion_new/            # Occlusion processing scripts
├── stable_diffusion/         # AI-based video processing
├── urban_scripts/            # Urban scene analysis scripts
├── urban_results/            # Experimental results and analysis
├── urban_annotations/        # Data annotations
├── urban_data/               # Urban dataset (small files)
├── vsi_adversarial_experiment/ # Main experiment videos (excluded)
└── video_editor.py          # Video processing utilities
```

## 🔬 Experimental Tasks

### Indoor Tasks (OpenCV Processing)

1. **Object Counting** - Table quantity statistics
2. **Object Size Estimation** - Stove dimension measurement
3. **Absolute Distance Measurement** - Sofa to stove distance
4. **Relative Directional Reasoning** - Spatial relationship identification
5. **Route Planning** - Sequential navigation planning

### Outdoor Tasks (Aerial Navigation)

1. **Progress Evaluation** - Navigation instruction following
2. **Landmark Position Assessment** - Spatial relationship identification
3. **Action Generation** - Next step selection in navigation

## 📊 Major Insights

### 1. Cognitive Load Theory Application

- **High cognitive load** (complex tasks): Selective vulnerability to different interferences
- **Medium cognitive load**: Different vulnerability patterns than hard tasks
- **Low cognitive load** (easy tasks): Universal robustness to visual degradation

### 2. Detection vs Reasoning Dissociation

Rain/fog video achieved perfect detection (100%) but inverted reasoning (0%) in medium-complexity tasks, proving these are separate cognitive processes with differential vulnerability.

### 3. Task Structure Protection

**Sequential, landmark-based tasks** (route planning) are more robust to visual interference than **single-step complex reasoning** tasks requiring spatial transformations.

### 4. Error Recovery Mechanisms

- **Majority evidence weighting**: Systems can overcome frame-level errors through redundant observations
- **Geometric consistency checking**: Higher-level reasoning can override unreliable low-level detections
- **Early information anchoring**: Explicit initial information provides strong protection against interference

## 🛠️ Technical Implementation

### Video Processing Methods

- **OpenCV**: Traditional computer vision for rain/fog and occlusion effects
- **Stable Diffusion**: Image-to-image processing for natural occlusion effects
- **CogVideo**: Video generation for temporal consistency (experimental)

### Interference Types

```python
# Rain/Fog: Global atmospheric effects
# - Reduced contrast and visibility
# - Uniform visual degradation

# Occlusion: Local blocking elements
# - Random shadow patterns
# - Partial information loss
# - Realistic foreground objects
```

## 📈 Results Summary

### Indoor Environment Findings

- **Object recognition** maintains reasonable accuracy under both interference types
- **Spatial reasoning** shows complexity-dependent vulnerability
- **Metric measurements** (distance, size) are highly sensitive to visual quality
- **Simple binary choices** (left/right) remain robust even under severe degradation

### Outdoor Environment Findings

- **Temporal sequencing** (progress evaluation) is highly vulnerable to both interference types
- **Spatial positioning** shows differential vulnerability (rain/fog affects vertical awareness)
- **Action generation** suffers from both directional confusion and action fragmentation
- **Landmark recognition** maintains better accuracy than progress assessment

## 🎓 Theoretical Implications

### For Robust AI System Design

1. **Simplify spatial reasoning queries** when operating under degraded conditions
2. **Use sequential task decomposition** rather than single-step complex reasoning
3. **Provide multiple observation opportunities** for critical decisions
4. **Implement evidence weighting mechanisms** to handle conflicting detections
5. **Protect reasoning synthesis** mechanisms, not just improve detection quality

### For Visual Interference Research

- Must test across complexity spectrum - easy task success doesn't predict hard task performance
- Interference effects are task-complexity dependent, not fixed properties
- Different interference types create characteristic error patterns across tasks

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python
pip install torch
pip install transformers
```

### Basic Usage



## 📚 Citation



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for discussions.

## 📧 Contact



---

**Note**: Large model files and video datasets are excluded from this repository due to size constraints. Please contact the authors for access to the complete dataset.
```

