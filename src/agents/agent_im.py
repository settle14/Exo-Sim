# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC_MJX (https://github.com/ZhengyiLuo/PHC_MJX)

import os
import torch
import numpy as np
import logging

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_humanoid import AgentHumanoid
from src.learning.learning_utils import to_test, to_cpu
from src.env.myolegs_im import MyoLegsIm
from typing import Tuple
logger = logging.getLogger(__name__)


class AgentIM(AgentHumanoid):
    """
    AgentIM is a specialized reinforcement learning agent for humanoid environments,
    extending AgentHumanoid with specific functionalities for the MyoLegsIm environment.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        """
        Initialize the AgentIM with configurations and set up necessary components.

        Args:
            cfg: Configuration object containing hyperparameters and settings.
            dtype: Data type for tensors (e.g., torch.float32).
            device: Device for computations (e.g., 'cuda' or 'cpu').
            training (bool, optional): Flag indicating if the agent is in training mode.
            checkpoint_epoch (int, optional): Epoch number from which to load the checkpoint.
        """
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def get_full_state_weights(self) -> dict:
        """
        Extends the state dictionary with termination history for checkpointing.

        Returns:
            dict: The state dictionary including termination history.
        """
        state = super().get_full_state_weights()
        return state
    
    def set_full_state_weights(self, state) -> None:
        """
        Loads the state dictionary.

        Args:
            state (dict): The state dictionary including termination history.
        """
        super().set_full_state_weights(state)
        
    
    def pre_epoch(self) -> None:
        """
        Performs operations before each training epoch, such as resampling motions.
        """
        if (self.epoch > 1) and self.epoch % self.cfg.env.resampling_interval == 1: # + 1 to evade the evaluations. 
            self.env.sample_motions()
        return super().pre_epoch()
    
    def setup_env(self):
        """
        Initializes the MyoLegsIm environment based on the configuration.
        """
        self.env = MyoLegsIm(self.cfg)
        logger.info("MyoLegsIm environment initialized.")

    def eval_policy(self, epoch: int = 0, dump: bool = False, runs=None) -> float:
        """
        Evaluate current policy. If cfg.run.exo_eval_both == True, run AB tests:
        - exo_on  (外骨骼力矩按策略输出)
        - exo_off (外骨骼力矩置零)
        Prints success/MPJPE/coverage and energy-style metrics:
        human_energy, exo_energy, exo_smooth, assist_align(+ pos/neg)
        Also logs to wandb if available.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting policy evaluation.")
        # 标记评估模式（环境内部可能会改变终止/渲染行为）
        self.env.start_eval(im_eval=True)

        def _one_pass(tag: str, exo_zero: bool):
            # 评估期切换外骨骼是否置零（需要 env.set_exo_zero 已实现）
            if hasattr(self.env, "set_exo_zero"):
                self.env.set_exo_zero(exo_zero)
            # === 新增：AB 切换前清理 env 里与 assist 相关的缓存 ===
            # 清历史 jerk 队列，避免 r_as 受上一次影响
            if hasattr(self.env, "_exo_hist") and self.env._exo_hist is not None:
                try:
                    self.env._exo_hist.clear()
                except Exception:
                    pass
            # 将上一时刻外骨骼力矩清零，避免第一步 Δu 异常放大
            if hasattr(self.env, "_exo_prev"):
                try:
                    import numpy as _np
                    self.env._exo_prev = _np.zeros_like(self.env._exo_prev)
                except Exception:
                    self.env._exo_prev = None
            # 清空本 episode 的 exo 统计量
            for _name in ("curr_exo_usage", "curr_exo_rate"):
                if hasattr(self.env, _name) and isinstance(getattr(self.env, _name), list):
                    getattr(self.env, _name).clear()
            # 明确标记 exo_zero 到 env（有些 reward 会读取该标记）
            setattr(self.env, "_exo_zero", bool(exo_zero))

                        # （可选）健壮性打印
            try:
                sdim = self.env.observation_space.shape[0]
                adim = self.env.action_space.shape[0]
                logger.debug(f"[{tag}] obs_dim={sdim}, act_dim={adim}, exo_zero={exo_zero}")
            except Exception:
                pass


            success_dict, mpjpe_dict, frame_coverage_dict = {}, {}, {}
            # New metric buffers that match myolegs_im.py reward_info
            rm_list, ras_list, rtau_list = [], [], []
            work_list, tausq_list = [], []
            upright_list, eh_list = [], []
            rbody_list, rrootxy_list = [], []


            # 用确定性动作评估；to_cpu 切回 CPU，避免多 GPU 环境的资源占用
            with to_cpu(*self.sample_modules), torch.no_grad():
                # 保守：确保策略处于 eval 模式
                try:
                    self.policy_net.eval()
                except Exception:
                    pass

                for run_idx in self.env.forward_motions():
                    # 单线程评估接口
                    result, mpjpe, frame_coverage = self.eval_single_thread()

                    # Pull new reward_info fields written in myolegs_im.py
                    # r_m, r_as, r_tau are already squashed in [0,1] form (exp(-sigma * cost))
                    info = getattr(self.env, "reward_info", {}) or {}
                    if "r_m" in info:
                        rm_list.append(float(info["r_m"]))
                    if "r_as" in info:
                        ras_list.append(float(info["r_as"]))
                    if "r_tau" in info:
                        rtau_list.append(float(info["r_tau"]))
                    if "work_val_proxy" in info:
                        work_list.append(float(info["work_val_proxy"]))
                    if "tau_sq" in info:
                        tausq_list.append(float(info["tau_sq"]))
                    if "upright_reward" in info:
                        upright_list.append(float(info["upright_reward"]))
                    if "energy_human" in info:
                        eh_list.append(float(info["energy_human"]))
                    if "r_body_pos" in info:
                        rbody_list.append(float(info["r_body_pos"]))
                    


                    success_dict[run_idx] = bool(result)
                    mpjpe_dict[run_idx] = float(mpjpe)
                    frame_coverage_dict[run_idx] = float(frame_coverage)

                    if runs is not None and len(success_dict) >= int(runs):
                        break

            # 汇总
            srate = np.mean(list(success_dict.values())) if success_dict else 0.0
            mpjpe_mean = np.mean(list(mpjpe_dict.values())) if mpjpe_dict else 0.0
            cov_mean = np.mean(list(frame_coverage_dict.values())) if frame_coverage_dict else 0.0
            rm_mean    = np.mean(rm_list)    if rm_list    else 0.0
            ras_mean   = np.mean(ras_list)   if ras_list   else 0.0
            rtau_mean  = np.mean(rtau_list)  if rtau_list  else 0.0
            work_mean  = np.mean(work_list)  if work_list  else 0.0
            tausq_mean = np.mean(tausq_list) if tausq_list else 0.0
            upr_mean   = np.mean(upright_list) if upright_list else 0.0
            eh_mean    = np.mean(eh_list)    if eh_list    else 0.0
            rbody_mean  = np.mean(rbody_list)  if rbody_list  else 0.0
            


            # 打印（mpjpe -> 毫米）
            print(
                f"[{tag}] success={srate*100:.2f}%, MPJPE={mpjpe_mean*1000:.3f}mm, "
                f"coverage={cov_mean*100:.2f}%"
            )
            print(
                f"[{tag}] r_m~{rm_mean:.3f}, r_as~{ras_mean:.3f}, r_tau~{rtau_mean:.3f}, "
                f"work_proxy~{work_mean:.4f}, tau_sq~{tausq_mean:.4f}, "
                f"human_energy~{eh_mean:.4f}, upright~{upr_mean:.3f}, "
                f"r_body_pos~{rbody_mean:.3f}"
            )



            # 可选：wandb 记录（若未初始化或无 wandb，不报错）
            try:
                import wandb  # noqa: F401
                wandb.log(
                    {
                        f"{tag}/success": srate,
                        f"{tag}/mpjpe_mm": mpjpe_mean * 1000.0,
                        f"{tag}/coverage": cov_mean,
                        f"{tag}/r_m": rm_mean,
                        f"{tag}/r_as": ras_mean,
                        f"{tag}/r_tau": rtau_mean,
                        f"{tag}/work_proxy": work_mean,
                        f"{tag}/tau_sq": tausq_mean,
                        f"{tag}/human_energy": eh_mean,
                        f"{tag}/upright": upr_mean,
                        f"{tag}/r_body_pos": rbody_mean,
                        

                    },
                    step=getattr(self, "epoch", epoch),
                )

            except Exception:
                pass

            return mpjpe_dict

        # AB 对照（默认读取 cfg.run.exo_eval_both；若为 False，只跑 exo_on）
        if getattr(self.cfg.run, "exo_eval_both", False):
            dict_on = _one_pass("exo_on", exo_zero=False)
            dict_off = _one_pass("exo_off", exo_zero=True)
            return dict_on, dict_off
        else:
            # 若用户通过 cfg.run.exo_zero 指定单次取值，则尊重该开关
            exo_zero_flag = bool(getattr(self.cfg.run, "exo_zero", False))
            tag = "exo_off" if exo_zero_flag else "exo_on"
            result = _one_pass(tag, exo_zero=exo_zero_flag)
            return result, (np.mean(list(result.values())) if result else 0.0)




    
    def eval_single_thread(self) -> Tuple[bool, float, float]:
        """
        Evaluates the policy in a single thread by running an episode.

        Returns:
            Tuple[bool, float, float]: (success_flag, mpjpe_value, frame_coverage)
        """
        with to_cpu(*self.sample_modules), torch.no_grad():
            obs_dict, info = self.env.reset()
            state = self.preprocess_obs(obs_dict)
            for t in range(10000):
                actions = self.policy_net.select_action(
                    torch.from_numpy(state).to(self.dtype), True
                )[0].numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(
                    self.preprocess_actions(actions)
                )
                next_state = self.preprocess_obs(next_obs)
                done = terminated or truncated

                if done:                      
                    return not terminated, self.env.mpjpe_value, self.env.frame_coverage
                state = next_state

        # If the loop exits without termination, consider it a failure
        return False, self.env.mpjpe_value, self.env.frame_coverage

            
            
    def run_policy(self, epoch: int = 0, dump: bool = False) -> dict:
        """
        Runs the trained policy in the environment.

        Args:
            epoch (int, optional): Current epoch number.
            dump (bool, optional): Flag indicating whether to dump run results.

        Returns:
            dict: Run metrics.
        """
        self.env.start_eval(im_eval = False)
        return super().run_policy(epoch, dump)

    def save_muscle_activation_data(self):
        """
        保存肌肉激活和生物力学数据，用于人体建模分析
        """
        import pickle
        from datetime import datetime
        
        print("开始保存肌肉激活数据...")
        
        # 创建导出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"data/muscle_exports/human_modeling_{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        # 收集肌肉数据
        try:
            muscle_controls = np.array(self.env.muscle_controls) if hasattr(self.env, 'muscle_controls') and self.env.muscle_controls else np.array([])
            muscle_forces = np.array(self.env.muscle_forces) if hasattr(self.env, 'muscle_forces') and self.env.muscle_forces else np.array([])
            
            print(f"肌肉控制数据形状: {muscle_controls.shape}")
            print(f"肌肉力量数据形状: {muscle_forces.shape}")
            
            if muscle_controls.size > 0:
                # Filter out NaN values for statistics
                valid_controls = muscle_controls[~np.isnan(muscle_controls)]
                if valid_controls.size > 0:
                    print(f"肌肉控制数值范围: [{np.min(valid_controls):.3f}, {np.max(valid_controls):.3f}]")
                else:
                    print("肌肉控制数据: 全部为NaN")
            
            if muscle_forces.size > 0:
                # Filter out NaN values for statistics
                valid_forces = muscle_forces[~np.isnan(muscle_forces)]
                if valid_forces.size > 0:
                    print(f"肌肉力量数值范围: [{np.min(valid_forces):.2f}, {np.max(valid_forces):.2f}] N")
                else:
                    print("肌肉力量数据: 全部为NaN")
            
            # 组织数据
            biomech_data = {
                # 核心肌肉数据
                'muscle_controls': muscle_controls,
                'muscle_forces': muscle_forces,
                
                # 运动学数据
                'joint_positions': self.env.joint_pos if hasattr(self.env, 'joint_pos') and self.env.joint_pos else [],
                'joint_velocities': self.env.joint_vel if hasattr(self.env, 'joint_vel') and self.env.joint_vel else [],
                'body_positions': self.env.body_pos if hasattr(self.env, 'body_pos') and self.env.body_pos else [],
                'body_rotations': self.env.body_rot if hasattr(self.env, 'body_rot') and self.env.body_rot else [],
                'body_velocities': self.env.body_vel if hasattr(self.env, 'body_vel') and self.env.body_vel else [],
                
                # 参考运动数据
                'reference_positions': self.env.ref_pos if hasattr(self.env, 'ref_pos') and self.env.ref_pos else [],
                'reference_rotations': self.env.ref_rot if hasattr(self.env, 'ref_rot') and self.env.ref_rot else [],
                'reference_velocities': self.env.ref_vel if hasattr(self.env, 'ref_vel') and self.env.ref_vel else [],
                
                # 接触和相位信息
                'feet_contacts': self.env.feet if hasattr(self.env, 'feet') and self.env.feet else [],
                'motion_ids': self.env.motion_id if hasattr(self.env, 'motion_id') and self.env.motion_id else [],
                'policy_outputs': self.env.policy_outputs if hasattr(self.env, 'policy_outputs') and self.env.policy_outputs else [],
                
                # 元数据
                'metadata': {
                    'model_name': 'kinesis-moe-imitation',
                    'export_time': timestamp,
                    'num_frames': len(muscle_controls) if muscle_controls.size > 0 else 0,
                    'sampling_frequency': 1/self.env.dt if hasattr(self.env, 'dt') else 30.0,
                    'muscle_count': muscle_controls.shape[1] if muscle_controls.ndim > 1 else 0,
                    'export_purpose': 'human_modeling_for_exoskeleton_analysis',
                }
            }
            
            # 保存主数据文件
            main_data_file = os.path.join(export_dir, "muscle_activation_data.pkl")
            with open(main_data_file, 'wb') as f:
                pickle.dump(biomech_data, f)
            
            # 保存CSV文件用于快速查看
            if muscle_controls.size > 0 and muscle_controls.ndim == 2:
                try:
                    import pandas as pd
                    
                    # 肌肉激活数据
                    muscle_df = pd.DataFrame(muscle_controls)
                    muscle_df.to_csv(os.path.join(export_dir, "muscle_activations.csv"), index=False)
                    
                    # 力量数据
                    if muscle_forces.size > 0 and muscle_forces.ndim == 2:
                        force_df = pd.DataFrame(muscle_forces)
                        force_df.to_csv(os.path.join(export_dir, "muscle_forces.csv"), index=False)
                    
                    print(f"✅ 肌肉数据已保存到: {export_dir}")
                    print(f"   - 主数据文件: muscle_activation_data.pkl")
                    print(f"   - 肌肉激活: muscle_activations.csv ({muscle_df.shape})")
                    if muscle_forces.size > 0 and muscle_forces.ndim == 2:
                        print(f"   - 肌肉力量: muscle_forces.csv ({force_df.shape})")
                    print(f"   - 总帧数: {biomech_data['metadata']['num_frames']}")
                    print(f"   - 肌肉数量: {biomech_data['metadata']['muscle_count']}")
                    
                except ImportError:
                    print("警告: pandas未安装，仅保存pickle文件")
                    print(f"✅ 肌肉数据已保存到: {export_dir}")
                    print(f"   - 主数据文件: muscle_activation_data.pkl")
            else:
                print("警告: 未收集到有效的肌肉数据")
            
            # 生成数据摘要报告
            self.generate_data_summary(export_dir, biomech_data)
            
            return export_dir
            
        except Exception as e:
            print(f"保存肌肉数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_data_summary(self, export_dir, data):
        """生成数据摘要报告"""
        
        # 获取力量数据统计信息
        force_stats = ""
        if isinstance(data['muscle_forces'], np.ndarray) and data['muscle_forces'].size > 0:
            force_stats = f"""
   - 力量范围: [{np.min(data['muscle_forces']):.2f}, {np.max(data['muscle_forces']):.2f}] N
   - 平均力量: {np.mean(data['muscle_forces']):.2f} N
"""
        else:
            force_stats = "\n   - 力量数据: 未收集或格式异常"
        
        summary = f"""
# 人体肌肉激活数据导出报告

## 导出信息
- 模型: kinesis-moe-imitation (MoE架构)
- 导出时间: {data['metadata']['export_time']}
- 数据帧数: {data['metadata']['num_frames']}
- 肌肉数量: {data['metadata']['muscle_count']}
- 采样频率: {data['metadata']['sampling_frequency']:.1f} Hz

## 数据内容
1. **肌肉激活数据** (muscle_controls): {len(data['muscle_controls'])} 帧
   - 数值范围: 0-1 (激活水平)
   - 用途: 分析肌肉激活模式，识别外骨骼辅助时机

2. **肌肉力量数据** (muscle_forces): {len(data['muscle_forces']) if hasattr(data['muscle_forces'], '__len__') else 'N/A'} 帧  
   - 单位: 牛顿 (N){force_stats}
   - 用途: 计算肌肉负载，设计外骨骼辅助力量

3. **运动学数据**: 关节位置、速度，身体姿态
   - 用途: 分析运动模式，设计外骨骼控制策略

4. **步态数据**: 足部接触，运动相位
   - 用途: 识别步态周期，优化辅助时机

## 外骨骼分析应用
此数据可用于:
- 识别需要辅助的关键肌群
- 分析肌肉负载时间模式  
- 设计外骨骼辅助策略
- 量化预期的减负效果

## 后续步骤
1. 运行 analyze_human_modeling.py 进行详细分析
2. 基于分析结果设计外骨骼控制算法
3. 与外骨骼辅助后的数据进行对比分析
"""
        
        summary_path = os.path.join(export_dir, "data_summary.md")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"📋 数据摘要报告已生成: {summary_path}")