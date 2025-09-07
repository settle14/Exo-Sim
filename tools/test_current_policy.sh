#!/bin/bash
# Usage examples:
#   bash test_current_policy.sh --mode test --headless False --exo on
#   bash test_current_policy.sh --mode train --headless True  --exo off --cpu True

mode=test
headless=False
exo=on
cpu=False

while [[ $# -gt 0 ]]; do
  case $1 in
    --headless) headless=$2; shift; shift;;
    --mode) mode=$2; shift; shift;;
    --exo) exo=$2; shift; shift;;
    --cpu) cpu=$2; shift; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ $mode == "train" ]]; then
  motion_file="data/kit_train_motion_dict.pkl"
  initial_pose_file="data/initial_pose/initial_pose_train.pkl"
else
  motion_file="data/kit_test_motion_dict.pkl"
  initial_pose_file="data/initial_pose/initial_pose_test.pkl"
fi

EXO_ZERO=False; [[ "$exo" == "off" ]] && EXO_ZERO=True
GLVAR=$( [[ "$headless" == "True" ]] && echo "egl" || echo "glfw" )

echo "Visualizing latest policy | mode=${mode} headless=${headless} exo=${exo} cpu=${cpu}"

# ---- CPU 强制：隐藏所有 GPU ----
if [[ "$cpu" == "True" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
fi

# 渲染后端
export MUJOCO_GL=$GLVAR

python src/run.py \
  project="exo_pd_angle_walk" \
  exp_name="walk_knee_hip_pdangle_v4" \
  epoch=-1 env=env_im learning=im_mlp run=eval_run \
  learning.actor_type="gauss" \
  run.headless=${headless} \
  run.xml_path="data/xml/myolegs_exo.xml" \
  run.motion_file="${motion_file}" \
  run.initial_pose_file="${initial_pose_file}" \
  run.im_eval=True run.test=True \
  run.num_motions=1 run.fast_forward=$([[ "$headless" == "True" ]] && echo True || echo False) \
  run.use_exo_in_action=true \
  run.exo_actuators=[exo_knee_r,exo_knee_l,exo_hip_r,exo_hip_l] \
  +run.exo_zero=${EXO_ZERO} \
  env.control_frequency_inv=5
