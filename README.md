# Exoskeleton-Assisted Human Locomotion Analysis

An extension of the Kinesis framework for studying exoskeleton assistance effects on human locomotion using reinforcement learning and musculoskeletal simulation.

## 🎯 Project Overview

This project builds upon the [Kinesis framework](https://github.com/amathislab/Kinesis) to analyze and optimize exoskeleton assistance for human walking. It provides comprehensive tools for:

- **EMG-style muscle activation analysis** comparing assisted vs unassisted locomotion
- **Quantitative assessment** of exoskeleton assistance effectiveness
- **Publication-ready visualizations** for academic research
- **Flexible exoskeleton configuration** with knee and hip joint assistance

## 🚀 Key Features

### Simulation Capabilities
- High-fidelity musculoskeletal human model (80 muscles, 20 DOF)
- Configurable exoskeleton with knee and hip assistance
- Real-time physics simulation using MuJoCo
- Integration with KIT motion capture dataset

### Analysis Tools
- **Muscle activation extraction** and time-series analysis
- **EMG-style comparison** between assisted/unassisted conditions  
- **Statistical evaluation** with publication-ready figures
- **Key metrics**: r_m (mean activation), work_proxy (metabolic cost)

### Visualization
- Professional 300 DPI figures suitable for journals
- Multi-panel layouts optimized for academic papers
- Comprehensive statistical annotations
- Both PNG and PDF output formats

## 📊 Research Results

Our analysis demonstrates significant assistance benefits:

| Metric | Improvement | Key Finding |
|--------|-------------|-------------|
| **Knee Flexors** | -23% | Largest reduction in muscle activation |
| **Ankle Plantarflexors** | -14% | Secondary benefit through kinetic chain |
| **Overall Activation (r_m)** | -1.8% | Consistent assistance across all muscles |
| **Work Proxy** | -1.7% | Reduced metabolic demand |

## 🛠️ Installation

### Prerequisites
- **Python 3.8+** with conda/miniconda
- **CUDA-capable GPU** (recommended)
- **MuJoCo 2.3+**

### Setup Instructions

1. **Clone and setup environment**:
   ```bash
   git clone [your-repo-url]
   cd kinesis-exo
   conda create -n kinesis python=3.8
   conda activate kinesis
   pip install -r requirements.txt
   ```

2. **Download mesh files** (Required):
   ```bash
   # Follow detailed instructions in MESH_DOWNLOAD_INSTRUCTIONS.md
   # Essential files from amathislab/Kinesis:
   git clone https://github.com/amathislab/Kinesis.git
   cp -r Kinesis/data/xml/*assets.xml data/xml/
   cp -r Kinesis/data/meshes/ data/meshes/
   ```

3. **Verify installation**:
   ```bash
   python -c "import mujoco; print('✓ MuJoCo ready')"
   python -c "import torch; print('✓ PyTorch ready')"
   ```

## 🎮 Quick Start

### Step 1: Extract Muscle Activation Data

**Exoskeleton OFF (baseline)**:
```bash
python tools/extract_muscle_activations.py \
  --exp_name "walk_knee_hip_pdangle_v4" \
  --xml_path "data/xml/myolegs_exo.xml" \
  --motion_file "data/kit_train_motion_dict.pkl" \
  --initial_pose_file "data/initial_pose/initial_pose_train.pkl" \
  --epoch 4600 \
  --use_exo_in_action false \
  --exo_zero true \
  --out_dir "results/exo_off"
```

**Exoskeleton ON (assisted)**:
```bash
python tools/extract_muscle_activations.py \
  --exp_name "walk_knee_hip_pdangle_v4" \
  --xml_path "data/xml/myolegs_exo.xml" \
  --motion_file "data/kit_train_motion_dict.pkl" \
  --initial_pose_file "data/initial_pose/initial_pose_train.pkl" \
  --epoch 4600 \
  --use_exo_in_action true \
  --exo_zero false \
  --out_dir "results/exo_on"
```

### Step 2: Perform EMG-Style Analysis

```bash
python tools/muscle_activation_analysis.py \
  --human_dir "results/exo_off" \
  --exo_dir "results/exo_on" \
  --out_dir "results/analysis" \
  --smooth_win 5
```

### Step 3: View Results

The analysis generates several publication-ready figures:
- `muscle_group_timeseries.png` - EMG-style activation patterns by muscle group
- `overall_activation_comparison.png` - Overall assistance effectiveness
- `assistance_metrics_analysis.png` - r_m and work_proxy temporal analysis
- `muscle_group_analysis_enhanced.png` - Comprehensive statistical comparison

## 📁 Project Structure

```
kinesis-exo/
├── data/                          # Data and models
│   ├── xml/myolegs_exo.xml       # Main exoskeleton model
│   ├── trained_models/           # RL model checkpoints
│   ├── kit_train_motion_dict.pkl # Motion data
│   └── initial_pose/             # Starting poses
├── tools/                         # Analysis scripts
│   ├── extract_muscle_activations.py  # Data extraction
│   ├── muscle_activation_analysis.py  # EMG analysis
│   ├── inspect_checkpoint.py          # Model utilities
│   └── MESH_DOWNLOAD_INSTRUCTIONS.md  # Asset setup guide
├── results/                       # Analysis outputs
│   ├── exo_off/                  # Baseline data
│   ├── exo_on/                   # Assisted condition data
│   └── analysis/                 # Final results
└── README.md                      # This file
```

## 🔧 Advanced Configuration

### Key Parameters

**Model Configuration**:
- `--exp_name`: Trained model identifier
- `--epoch`: Specific checkpoint (use 4600 for best performance)
- `--xml_path`: Musculoskeletal model file

**Exoskeleton Settings**:
- `--use_exo_in_action`: Include exoskeleton in control policy
- `--exo_zero`: Force zero exoskeleton output (baseline condition)
- `--exo_actuators`: Specify assisted joints ("exo_knee_r,exo_knee_l,exo_hip_r,exo_hip_l")

**Analysis Options**:
- `--smooth_win`: Smoothing window for visualization (5-10 recommended)
- `--trim_start/end`: Remove initial/final transient periods
- `--align_mode`: Time series alignment ("intersection" or "firstN")

### Checkpoint Management

```bash
# Inspect model checkpoint
python tools/inspect_checkpoint.py --checkpoint model.pth --action inspect

# Use specific epoch (if available)
python tools/extract_muscle_activations.py --epoch 4000  # Use 4000 epoch checkpoint
```

## 📊 Understanding the Results

### Key Metrics Explained

**r_m (Mean Muscle Activation)**:
- Average activation across all muscles
- Lower values indicate reduced muscular effort
- Target: 1-3% reduction with exoskeleton

**work_proxy (Sum of Squared Activations)**:
- Metabolic cost proxy based on muscle activation patterns
- Quadratic relationship emphasizes high activations
- Target: Similar reduction pattern to r_m

**Muscle Group Analysis**:
- **Knee flexors**: Expect largest benefit (10-25% reduction)
- **Hip extensors**: May show compensation (slight increase)
- **Ankle muscles**: Secondary benefits through kinetic coupling

### Expected Results Timeline

| Phase | Duration | Pattern |
|-------|----------|---------|
| **0-1s** | Initialization | High variability, ignore for analysis |
| **1-3s** | Steady walking | Clear cyclical patterns, main analysis window |
| **3-4s** | Potential fatigue | May show increased activation |

## 🎨 Visualization Features

- **High-resolution output**: 300 DPI for journal submission
- **Professional styling**: Academic color schemes and typography
- **Multi-panel layouts**: Optimized for single/double column figures
- **Statistical annotations**: Automatic percentage change calculations
- **Publication formats**: Both PNG (presentations) and PDF (print) outputs

## 🐛 Troubleshooting

### Common Issues

**Missing mesh files**:
```bash
# Error: "No such file or directory: myotorsorigid_assets.xml"
# Solution: Follow MESH_DOWNLOAD_INSTRUCTIONS.md
```

**CUDA memory errors**:
```bash
# Add --headless true to reduce memory usage
python tools/extract_muscle_activations.py --headless true [other args]
```

**Poor exoskeleton performance**:
- Verify `--epoch 4600` (avoid overfitted checkpoints)
- Check `--exo_zero true/false` matches intended condition
- Ensure `--use_exo_in_action` is configured correctly

## 📚 Citation

If you use this code for your research, please cite:

```bibtex
@misc{exo_kinesis_2025,
  title={Exoskeleton-Assisted Human Locomotion Analysis using Kinesis Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/kinesis-exo}
}

@article{simos2025kinesis,
  title={Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control},
  author={Simos, Merkourios and Chiappa, Alberto Silvio and Mathis, Alexander},
  journal={arXiv},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear documentation

## 📞 Support

- **Issues**: [GitHub Issues](your-repo-issues-link)
- **Email**: [your-email@university.edu]
- **Original Kinesis**: [amathislab/Kinesis](https://github.com/amathislab/Kinesis)

## 🙏 Acknowledgments

- **Mathis Lab (EPFL)** for the original Kinesis framework
- **MyoSuite** for musculoskeletal modeling tools
- **MuJoCo** physics simulation engine
- **KIT Motion Dataset** for human locomotion data

---

**⭐ Star this repository if it advances your research!**
