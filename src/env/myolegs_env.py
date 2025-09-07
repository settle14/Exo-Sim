# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import os
import sys
from typing import List, Tuple
sys.path.append(os.getcwd())

import numpy as np
from collections import OrderedDict
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation as sRot

from src.env.myolegs_base_env import BaseEnv
import src.utils.np_transform_utils as npt_utils
# --- add: module logger ---
import logging
logger = logging.getLogger(__name__)


class MyoLegsEnv(BaseEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg=self.cfg)
        self.setup_configs(cfg)

        self.create_sim(
            cfg.run.xml_path
        )
        self.setup_myolegs_params()
        self.reward_info = {}
        
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(self.get_obs_size()),
            np.inf * np.ones(self.get_obs_size()),
            dtype=self.dtype,
        )
        
        nu = self.mj_model.nu
        self.action_space = gym.spaces.Box(
            low=-np.ones(nu, dtype=self.dtype),
            high=np.ones(nu, dtype=self.dtype),
            dtype=self.dtype,
        )
        # --- EXO 开关：初始化为 cfg.run.exo_zero，运行时可被 setter 覆盖 ---
        self._exo_zero = bool(getattr(self.cfg.run, "exo_zero", False))

    # add this helper to recompute observation_space after run-params are known
    def refresh_observation_space(self):
        from gym import spaces
        obs_dim = int(self.get_obs_size())
        self.observation_space = spaces.Box(
            -np.inf * np.ones(obs_dim, dtype=np.float32),
            +np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )


    def setup_configs(self, cfg) -> None:
        """
        Sets various configuration parameters.
        """
        self._kp_scale = cfg.env.kp_scale
        self._kd_scale = cfg.env.kd_scale
        # default to PD if not provided in run config
        self.control_mode = getattr(cfg.run, "control_mode", "PD")
        assert self.control_mode in ("PD", "direct"), \
            f"Unknown control_mode {self.control_mode}, expected 'PD' or 'direct'"
        self.max_episode_length = 300
        self.dtype = np.float32



    def setup_myolegs_params(self) -> None:
        """
        Sets up various parameters related to the MyoLeg environment.
        """
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):
            body_name = self.mj_model.body(i).name
            self.mj_body_names.append(body_name)
        
        self.body_names = self.mj_body_names[1:] # the first one is always world
            
        self.num_bodies = len(self.body_names)
        self.num_vel_limit = self.num_bodies * 3
        self.robot_body_idxes = [
            self.mj_body_names.index(name) for name in self.body_names
        ]
        self.robot_idx_start = self.robot_body_idxes[0]
        self.robot_idx_end = self.robot_body_idxes[-1] + 1

        self.qpos_lim = np.max(self.mj_model.jnt_qposadr) + self.mj_model.jnt_qposadr[-1] - self.mj_model.jnt_qposadr[-2]
        self.qvel_lim = np.max(self.mj_model.jnt_dofadr) + self.mj_model.jnt_dofadr[-1] - self.mj_model.jnt_dofadr[-2]
        
        # These are not required but are included for future reference
        geom_type_id = mujoco.mju_str2Type("geom")
        self.floor_idx = mujoco.mj_name2id(self.mj_model, geom_type_id, "floor")

        # 执行器名列表（项目已有）
        act_names = get_actuator_names(self.mj_model)

        # 外骨骼执行器索引（来自 run.yaml）
        self.exo_names = list(getattr(self.cfg.run, "exo_actuators", []))
        self.exo_idx = [act_names.index(n) for n in self.exo_names if n in act_names]

        # 肌肉索引 = 除去 exo 的全部
        self.muscle_idx = [i for i in range(self.mj_model.nu) if i not in self.exo_idx]
        self._exo_prev = np.zeros((len(self.exo_idx),), dtype=np.float32)

        # === 新增：外骨骼力矩缩放（默认 1.0，可在 run.exo_torque_scale 覆盖） ===
        self.exo_scale = float(getattr(self.cfg.run, "exo_torque_scale", 1.0))
        # === 外骨骼力矩低通与斜率限幅 ===
        # 低通：一阶 IIR，系数 a∈(0,1]；0 关闭，越大越“跟手”
        self.exo_tau_lp_alpha = float(getattr(self.cfg.run, "exo_torque_lp_alpha", 0.25))
        # 斜率限幅：单步最大改变量 = exo_rate_limit * (hi - lo)；0 关闭
        self.exo_rate_limit  = float(getattr(self.cfg.run, "exo_rate_limit", 0.10))

        # 低通与限幅的内部状态
        self._exo_tau_lp   = np.zeros((len(self.exo_idx),), dtype=np.float32)
        self._exo_tau_prev = np.zeros((len(self.exo_idx),), dtype=np.float32)

        # （可读性）给肌肉索引再留一个同义引用，后面统计能耗用
        self.muscle_actuator_ids = list(self.muscle_idx)    

        # exo ctrlrange
        self.exo_ctrlrange = self.mj_model.actuator_ctrlrange[self.exo_idx] if self.exo_idx else np.zeros((0, 2))
        
        # 供 reward 中 r_tau 归一化：每个外骨骼通道的“半量程”
        if self.exo_ctrlrange.size > 0:
            self._exo_tau_limit = (0.5 * (self.exo_ctrlrange[:, 1] - self.exo_ctrlrange[:, 0])).astype(np.float32)
        else:
            self._exo_tau_limit = None

        # 额外缓存：EXO 平滑/对齐用
        self.curr_exo_usage = []     # ∥u_norm∥²
        self.curr_exo_rate  = []     # ∥Δu_norm∥²
        self._exo_prev      = np.zeros((len(self.exo_idx),), dtype=np.float32)

        # --- EXO dof indices (aligned with self.exo_idx), generic for hip/knee/ankle ---
        self.exo_dof_idx = []
        for aid in self.exo_idx:
            jnt_id = int(self.mj_model.actuator_trnid[aid, 0])   # joint id driven by actuator
            dof_id = int(self.mj_model.jnt_dofadr[jnt_id])       # hinge joint => single dof index
            self.exo_dof_idx.append(dof_id)
        self.exo_dof_idx = np.array(self.exo_dof_idx, dtype=np.int32)
        # === PD-angle mode parameters (from paper) ===
        self.exo_mode = str(getattr(self.cfg.run, "exo_control_mode", "direct"))  # "direct" or "pd_angle"
        self.exo_kp = float(getattr(self.cfg.run, "exo_kp", 50.0))
        self.exo_kv = float(getattr(self.cfg.run, "exo_kv", 14.14))
        # amplitude of desired joint angle around neutral, in radians
        self.exo_angle_amp = float(getattr(self.cfg.run, "exo_angle_amp_rad", 0.35))
        # second-order low-pass by cascading two first-order filters
        self.exo_lp_alpha = float(getattr(self.cfg.run, "exo_lp_alpha", 0.2))

        # buffers for 2-stage low-pass of desired angles
        self._exo_ref_raw  = np.zeros((len(self.exo_idx),), dtype=np.float32)
        self._exo_ref_lp1  = np.zeros((len(self.exo_idx),), dtype=np.float32)
        self._exo_ref_lp2  = np.zeros((len(self.exo_idx),), dtype=np.float32)

        # neutral angle: use current qpos as neutral
        self._exo_neutral = np.zeros((len(self.exo_idx),), dtype=np.float32)
        if len(self.exo_dof_idx) > 0:
            self._exo_neutral = self.mj_data.qpos[self.exo_dof_idx].astype(np.float32)


        


    
    def get_obs_size(self) -> int:
        """
        Returns the size of the observations. In the environment class, this defaults to the size of the proprioceptive observations.
        """
        return self.get_self_obs_size()

    def compute_observations(self) -> np.ndarray:
        """
        Computes the observations. In the environment class, this defaults to the proprioceptive observations.
        """
        obs = self.compute_proprioception()
        return obs
    
    def compute_info(self):
        raise NotImplementedError
    
    def get_self_obs_size(self) -> int:
        """
        Returns the size of the proprioceptive observations.

        IMPORTANT: This must count EXACTLY the same keys that `compute_proprioception()`
        may append, otherwise policy input dim will mismatch runtime observations.
        """
        inputs = self.cfg.run.proprioceptive_inputs
        tally = 0

        # counts we need
        n_mus = len(self.muscle_idx)                # muscle actuators
        n_exo = len(self.exo_idx)                   # exo actuators (knee/hip, etc.)
        nb    = getattr(self, "num_bodies", 0)      # body segments for local_* terms

        # ---- root-level features ----
        if "root_height" in inputs:
            tally += 1
        if "root_tilt" in inputs:
            tally += 1  # [cos(roll), sin(roll), cos(pitch), sin(pitch)] per your code

        # ---- per-body local kinematics ----
        # NB: These must match what you actually concatenate in compute_proprioception()
        if "local_body_pos" in inputs:
            tally += 3 * nb
        if "local_body_rot" in inputs:
            tally += 6 * nb  # you use 6D rot representation for each body
        if "local_body_vel" in inputs:
            tally += 3 * nb
        if "local_body_ang_vel" in inputs:
            tally += 3 * nb

        # ---- muscle channels ----
        if "muscle_len" in inputs:
            tally += n_mus
        if "muscle_vel" in inputs:
            tally += n_mus
        if "muscle_force" in inputs:
            tally += n_mus

        # ---- contacts ----下
        if "feet_contacts" in inputs:
            tally += 4  # 2 feet × 2 sensors (heel/toe) per your get_touch()

        # ---- exo channels (if included into observation) ----
        if "exo_torque" in inputs:
            tally += n_exo
        if "exo_vel" in inputs:
            tally += n_exo

        # ---- joint channels explicitly added ----
        if "knee_pos" in inputs:
            tally += 2
        if "knee_vel" in inputs:
            tally += 2
        if "hip_pos" in inputs:
            tally += 2
        if "hip_vel" in inputs:
            tally += 2

        return tally


    def compute_proprioception(self) -> np.ndarray:
        """
        Computes proprioceptive observations for the current simulation state.

        Updates the humanoid's body and actuator states, and generates observations 
        based on the configured inputs.

        Returns:
            np.ndarray: Flattened array of proprioceptive observations.

        Notes:
            - The observations are also stored in the `self.proprioception` attribute.
        """


        mujoco.mj_kinematics(self.mj_model, self.mj_data)  # update xpos to the latest simulation values
        
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]
        
        body_vel = self.get_body_linear_vel()[None,]
        body_ang_vel = self.get_body_angular_vel()[None,]

        obs_dict =  compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
        
        root_rot = sRot.from_quat(self.mj_data.qpos[[4, 5, 6, 3]])
        root_rot_euler = root_rot.as_euler("xyz")

        myolegs_obs = OrderedDict()
        
        inputs = self.cfg.run.proprioceptive_inputs

        if "root_height" in inputs:
            myolegs_obs["root_height"] = obs_dict["root_h_obs"] # 1
        if "root_tilt" in inputs:
            myolegs_obs["root_tilt"] = np.array([np.cos(root_rot_euler[0]), np.sin(root_rot_euler[0]), np.cos(root_rot_euler[1]), np.sin(root_rot_euler[1])]) # 4
        if "local_body_pos" in inputs:
            myolegs_obs["local_body_pos"] = obs_dict["local_body_pos"][0] # 3 * num_bodies
        if "local_body_rot" in inputs:
            myolegs_obs["local_body_rot"] = obs_dict["local_body_rot_obs"][0] # 6 * num_bodies
        if "local_body_vel" in inputs:
            myolegs_obs["local_body_vel"] = obs_dict["local_body_vel"][0] # 3 * num_bodies
        if "local_body_ang_vel" in inputs:
            myolegs_obs["local_body_ang_vel"] = obs_dict["local_body_ang_vel"][0] # 3 * num_bodies
        if "muscle_len" in inputs:
            myolegs_obs["muscle_len"] = np.nan_to_num(
                self.mj_data.actuator_length[self.muscle_idx].copy()
            ).astype(self.dtype, copy=False)

        if "muscle_vel" in inputs:
            myolegs_obs["muscle_vel"] = np.nan_to_num(
                self.mj_data.actuator_velocity[self.muscle_idx].copy()
            ).astype(self.dtype, copy=False)

        if "muscle_force" in inputs:
            myolegs_obs["muscle_force"] = np.nan_to_num(
                self.mj_data.actuator_force[self.muscle_idx].copy()
            ).astype(self.dtype, copy=False)

        if "feet_contacts" in inputs:
            myolegs_obs["feet_contacts"] = self.get_touch() # 4
        # 小工具：按名字读传感器（1 维）
        def _sensor_scalar(name: str) -> float:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.mj_model.sensor_adr[sid]
            return float(self.mj_data.sensordata[adr])

        # 小工具：读关节角/角速度
        def _knee_pos_vel(jname: str):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = self.mj_model.jnt_qposadr[jid]
            dadr = self.mj_model.jnt_dofadr[jid]
            return float(self.mj_data.qpos[qadr]), float(self.mj_data.qvel[dadr])
        

        if "exo_torque" in inputs:
            # actuator_force 按 actuator 顺序给出，对齐 self.exo_idx
            if len(self.exo_idx) > 0:
                myolegs_obs["exo_torque"] = self.mj_data.actuator_force[self.exo_idx].astype(self.dtype, copy=False)
            else:
                myolegs_obs["exo_torque"] = np.zeros(0, dtype=self.dtype)

        if "exo_vel" in inputs:
            # 取外骨骼所驱动关节的角速度 qvel，索引 self.exo_dof_idx
            if len(self.exo_dof_idx) > 0:
                myolegs_obs["exo_vel"] = self.mj_data.qvel[self.exo_dof_idx].astype(self.dtype, copy=False)
            else:
                myolegs_obs["exo_vel"] = np.zeros(0, dtype=self.dtype)


        if "knee_pos" in inputs:
            pr, _ = _knee_pos_vel("knee_angle_r")
            pl, _ = _knee_pos_vel("knee_angle_l")
            myolegs_obs["knee_pos"] = np.array([pr, pl], dtype=self.dtype)

        if "knee_vel" in inputs:
            _, vr = _knee_pos_vel("knee_angle_r")
            _, vl = _knee_pos_vel("knee_angle_l")
            myolegs_obs["knee_vel"] = np.array([vr, vl], dtype=self.dtype)

        if "hip_pos" in inputs:
            hr = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
            hl = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_l")
            pr = float(self.mj_data.qpos[self.mj_model.jnt_qposadr[hr]])
            pl = float(self.mj_data.qpos[self.mj_model.jnt_qposadr[hl]])
            myolegs_obs["hip_pos"] = np.array([pr, pl], dtype=self.dtype)

        if "hip_vel" in inputs:
            hr = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
            hl = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_l")
            vr = float(self.mj_data.qvel[self.mj_model.jnt_dofadr[hr]])
            vl = float(self.mj_data.qvel[self.mj_model.jnt_dofadr[hl]])
            myolegs_obs["hip_vel"] = np.array([vr, vl], dtype=self.dtype)
    


        self.proprioception = myolegs_obs

        obs_vec = np.concatenate([v.ravel() for v in myolegs_obs.values()], axis=0).astype(self.dtype, copy=False)
        declared = self.get_self_obs_size()
        if obs_vec.shape[0] != declared:
            logger.error(f"[OBS MISMATCH] proprio dim(real)={obs_vec.shape[0]} vs declared={declared}. "
                        f"Check get_self_obs_size()/compute_proprioception() keys and run.proprioceptive_inputs.")
        return obs_vec

    
    def get_body_xpos(self):
        """
        Returns the body positions of the agent in X, Y, Z coordinates.
        """
        return self.mj_data.xpos.copy()[self.robot_idx_start : self.robot_idx_end]

    def get_body_xquat(self):
        """
        Returns the body rotations of the agent in quaternion
        """
        return self.mj_data.xquat.copy()[self.robot_idx_start : self.robot_idx_end]
    
    def get_body_linear_vel(self):
        """
        Returns the linear velocity of the agent's body parts.
        """
        return self.mj_data.sensordata[:self.num_vel_limit].reshape(self.num_bodies, 3).copy()
    
    def get_body_angular_vel(self):
        """
        Returns the angular velocity of the agent's body parts.
        """
        return self.mj_data.sensordata[self.num_vel_limit:2 * self.num_vel_limit].reshape(self.num_bodies, 3).copy()
    
    def get_touch(self):
        """
        Return ONLY foot touch sensors as a 4-D vector.
        - 兼容不同 MuJoCo 版本: 使用 mjSENS_TOUCH
        - 若没有 TOUCH 传感器: 返回 4 维零
        - 若找到但维度≠4: 按名字排序后截断/零填充到 4 维
        """
        # 取 TOUCH 类型常量（MuJoCo 正确名是 mjSENS_TOUCH）
        try:
            TOUCH = int(mujoco.mjtSensor.mjSENS_TOUCH)
        except Exception:
            TOUCH = None

        vals = []
        touch_sids = []
        if TOUCH is not None and hasattr(self.mj_model, "sensor_type"):
            for sid in range(self.mj_model.nsensor):
                if int(self.mj_model.sensor_type[sid]) == TOUCH:
                    touch_sids.append(sid)

        # 没有 TOUCH → 返回 4 维零，保证与 get_self_obs_size 的 +4 对齐
        if not touch_sids:
            return np.zeros(4, dtype=self.dtype)

        # 按传感器名字稳定排序，避免训练时顺序漂移
        names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            for sid in touch_sids
        ]
        touch_sids = [sid for _, sid in sorted(zip(names, touch_sids))]

        for sid in touch_sids:
            adr = int(self.mj_model.sensor_adr[sid])
            dim = int(self.mj_model.sensor_dim[sid])
            vals.extend(self.mj_data.sensordata[adr:adr + dim])

        arr = np.asarray(vals, dtype=self.dtype)

        # 统一成 4 维（过长截断，不足补零）
        if arr.size < 4:
            arr = np.concatenate([arr, np.zeros(4 - arr.size, dtype=self.dtype)], axis=0)
        elif arr.size > 4:
            arr = arr[:4]
        return arr


        
    def get_qpos(self):
        """
        Returns the joint positions of the agent.
        """
        return self.mj_data.qpos.copy()[: self.qpos_lim]

    def get_qvel(self):
        """
        Returns the joint velocities of the agent.
        """
        return self.mj_data.qvel.copy()[:self.qvel_lim]
    
    def get_root_pos(self):
        """
        Returns the position of the agent's root.
        """
        return self.get_body_xpos()[0].copy()

    def compute_reward(self, action):
        """
        Placeholder for reward computation. In the environment class, this defaults to 0.
        """
        reward = 0
        return reward

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the episode should reset based on termination and truncation conditions.

        In the environment class, the episode ends if the current time step exceeds the maximum episode length.
        """
        if self.cur_t > self.max_episode_length:
            return False, True
        else:
            return False, False

    def pre_physics_step(self, action):
        """
        Placeholder for pre-physics-step computations. In the environment class, this defaults to no operation
        """
        pass

    def _exo_lowpass2(self, target: np.ndarray) -> np.ndarray:
        """Second-order low-pass by cascading two first-order filters."""
        a = float(self.exo_lp_alpha)
        # stage 1
        self._exo_ref_lp1 = (1.0 - a) * self._exo_ref_lp1 + a * target
        # stage 2
        self._exo_ref_lp2 = (1.0 - a) * self._exo_ref_lp2 + a * self._exo_ref_lp1
        return self._exo_ref_lp2
    
    def _filter_and_limit_exo_tau(self, tau_raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        对外骨骼原始力矩做一阶低通 + 斜率限幅（逐 actuator）。
        - 低通： tau_lp = (1-a)*prev + a*tau_raw
        - 限幅： |tau - tau_prev| <= rate_limit * (hi - lo)
        """
        tau = np.asarray(tau_raw, dtype=np.float32)

        # 一阶低通
        a = float(self.exo_tau_lp_alpha)
        if a > 0.0:
            self._exo_tau_lp = (1.0 - a) * self._exo_tau_lp + a * tau
        else:
            self._exo_tau_lp = tau.copy()

        tau_lp = self._exo_tau_lp

        # 斜率限幅
        r = float(self.exo_rate_limit)
        if r > 0.0:
            rng = hi - lo
            # 防止 0 量程
            rng = np.where(rng > 1e-8, rng, 1.0)
            max_step = r * rng
            delta = np.clip(tau_lp - self._exo_tau_prev, -max_step, max_step)
            tau_lim = self._exo_tau_prev + delta
        else:
            tau_lim = tau_lp

        return tau_lim


    def physics_step(self, action: np.ndarray = None) -> None:
        """
        Executes a physics step in the simulation with the given action.

        Depending on the control mode, computes muscle activations and applies them 
        to the simulation. Tracks power usage during the step.

        Args:
            action (np.ndarray): The action to apply. If None, a random action is sampled.
        """
        self.curr_power_usage = []
        # 清空 EXO 缓存（本步累积）
        try:
            self.curr_exo_usage = []
            self.curr_exo_rate = []
        except:
            pass

        if action is None:
            action = self.action_space.sample()
        self.last_action = action.copy()

        nu = self.mj_model.nu
        full_ctrl = np.zeros(nu, dtype=self.dtype)

        # --- 肌肉（PD） ---
        if self.control_mode == "PD":
            action_for_pd = np.zeros(nu, dtype=self.dtype)
            if hasattr(self, "muscle_idx") and len(self.muscle_idx) > 0:
                action_for_pd[self.muscle_idx] = action[self.muscle_idx]
            target_lengths = action_to_target_length(action_for_pd, self.mj_model)

        for _ in range(self.control_freq_inv):
            if self.paused:
                continue

            if self.control_mode == "PD":
                muscle_activity_full = target_length_to_activation(target_lengths, self.mj_data, self.mj_model)
                muscle_activity_full = np.clip(np.asarray(muscle_activity_full, dtype=np.float32), 0.0, 1.0)

                if self.cfg.run.deactivate_muscles:
                    muscle_activity_full = self.deactivate_muscles(muscle_activity_full, ["tibant_l", "tibant_r"])

                if hasattr(self, "muscle_idx") and len(self.muscle_idx) > 0:
                    full_ctrl[self.muscle_idx] = muscle_activity_full[self.muscle_idx]
    
            elif self.control_mode == "direct":
                if hasattr(self, "muscle_idx") and len(self.muscle_idx) > 0:
                    full_ctrl[self.muscle_idx] = (action[self.muscle_idx] + 1.0) / 2.0
                if hasattr(self, "muscle_idx") and len(self.muscle_idx) > 0:
                    self.action_human = full_ctrl[self.muscle_idx].copy()
            else:
                raise NotImplementedError

            # --- 外骨骼（直接力矩） ---
            # --- Exoskeleton control ---
            if getattr(self.cfg.run, "use_exo_in_action", False) and len(self.exo_idx) > 0:
                lo = self.exo_ctrlrange[:, 0]
                hi = self.exo_ctrlrange[:, 1]
                a_exo = action[self.exo_idx].astype(self.dtype)

                if self._exo_zero:
                    exo_torque = np.zeros_like(a_exo, dtype=self.dtype)
                    # 清零滤波/限幅状态，避免下一帧“误差突变”
                    self._exo_tau_lp[:]   = 0.0
                    self._exo_tau_prev[:] = 0.0

                elif self.exo_mode == "pd_angle":
                    # pd_angle：策略输出映射为期望角；先对期望角做二阶低通，再做 PD
                    if len(self.exo_dof_idx) == 0:
                        exo_torque = lo + 0.5 * (a_exo + 1.0) * (hi - lo)
                    else:
                        p_neu = self._exo_neutral
                        p_des_raw = p_neu + self.exo_angle_amp * a_exo
                        p_des = self._exo_lowpass2(p_des_raw)
                        q  = self.mj_data.qpos[self.exo_dof_idx]
                        dq = self.mj_data.qvel[self.exo_dof_idx]
                        tau = self.exo_kp * (p_des - q) - self.exo_kv * dq
                        exo_torque = np.clip(tau, lo, hi).astype(self.dtype)
                        # 对 torque 再做低通+斜率限幅
                        exo_torque = self._filter_and_limit_exo_tau(exo_torque, lo, hi)

                else:
                    # direct：策略输出直接映射扭矩，再做低通+斜率限幅
                    tau_raw = lo + 0.5 * (a_exo + 1.0) * (hi - lo)
                    exo_torque = self._filter_and_limit_exo_tau(tau_raw, lo, hi)

                # 全局缩放（放最后，数值最直观）
                exo_torque = exo_torque * self.exo_scale
                full_ctrl[self.exo_idx] = exo_torque.astype(self.dtype, copy=False)
                self.tau_exo = exo_torque.astype(self.dtype, copy=False)

                # EXO energy / smoothness stats (normalized)
                if not self._exo_zero:
                    u_norm = np.where((hi - lo) > 1e-8,
                                      (exo_torque - 0.5*(hi+lo)) / (0.5*(hi - lo)),
                                      0.0)
                    self.curr_exo_usage.append(float(np.mean(u_norm * u_norm)))

                    du = exo_torque - self._exo_prev
                    du_norm = np.where((hi - lo) > 1e-8, du / (0.5*(hi - lo)), 0.0)
                    self.curr_exo_rate.append(float(np.mean(du_norm * du_norm)))

                    self._exo_prev = exo_torque.copy()
                    # 记录上一输出，供下一步斜率限幅
                    self._exo_tau_prev = exo_torque.astype(np.float32, copy=False)
                else:
                    self._exo_prev      = np.zeros_like(exo_torque, dtype=self.dtype)
                    self._exo_tau_prev  = np.zeros_like(exo_torque, dtype=np.float32)




            # 下发控制并步进
            self.mj_data.ctrl[:] = full_ctrl
            mujoco.mj_step(self.mj_model, self.mj_data)

            # 生物力学记录（保留原逻辑）
            if hasattr(self, 'recording_biomechanics') and self.recording_biomechanics:
                self._current_step_muscle_forces = self.mj_data.actuator_force.copy()
                self._current_step_muscle_controls = full_ctrl.copy()

            # === 修改：human 能耗只看“肌肉通道”，外骨骼位归零 ===
            human_ctrl = np.zeros_like(full_ctrl, dtype=self.dtype)
            if hasattr(self, "muscle_actuator_ids") and len(self.muscle_actuator_ids) > 0:
                human_ctrl[self.muscle_actuator_ids] = full_ctrl[self.muscle_actuator_ids]
            else:
                # 兜底：如果没解析到索引，就退化为原先的 full_ctrl
                human_ctrl = full_ctrl
            self.curr_power_usage.append(self.compute_energy_reward(human_ctrl))


    def set_exo_zero(self, flag: bool):
        self._exo_zero = bool(flag)

    def set_exo_scale(self, scale: float) -> None:
        """
        Set global exoskeleton torque scale in [0, 1].
        Agent should call this every epoch for curriculum.
        """
        self.exo_scale = float(np.clip(scale, 0.0, 1.0))


    def clear_energy_buffers(self):
        self.curr_power_usage = []
        self.curr_exo_usage = []
        self.curr_exo_rate = []


    
    def deactivate_muscles(self, muscle_activity: np.ndarray, targetted_muscles: List[str]) -> np.ndarray:
        """
        Deactivates specific muscles by setting their activation values to zero.

        Args:
            muscle_activity (np.ndarray): Array of muscle activation values.
            targetted_muscles (list): List of muscle names (str) to deactivate.

        Returns:
            np.ndarray: Updated muscle activation values with the targeted muscles deactivated.
        """
        muscle_names = get_actuator_names(self.mj_model)
        indexes = [muscle_names.index(muscle) for muscle in targetted_muscles]
        for idx in indexes:
            muscle_activity[idx] = 0.0
        return muscle_activity

    def post_physics_step(self, action):
        """
        Processes the environment state after the physics step.

        Increments the simulation time, computes observations, reward, and checks 
        for termination or truncation conditions. Collects and returns additional 
        information about the reward components.

        Args:
            action (np.ndarray): The action applied at the current step.

        Returns:
            Tuple:
                - obs (np.ndarray): Current observations.
                - reward (float): Reward for the current step.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information, including raw reward components.
        """
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(action)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info.update(self.reward_info)
        return obs, reward, terminated, truncated, info
    
    def init_myolegs(self):
        """
        Initializes the MyoLegs environment. In the environment class, this defaults to
        setting the agent to a default position.
        """
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        self.mj_data.qpos[2] = 0.94
        self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])   

    def reset_myolegs(self):
        self.init_myolegs()
        # === 新增：每个 episode 清空 EXO 缓存 / 能耗缓存 ===
        self._exo_prev = np.zeros((len(self.exo_idx),), dtype=np.float32)
        self.clear_energy_buffers()
        # reset exo torque filters (LPF + rate limiter)
        self._exo_tau_lp   = np.zeros((len(self.exo_idx),), dtype=np.float32)
        self._exo_tau_prev = np.zeros((len(self.exo_idx),), dtype=np.float32)

        # 不要覆盖 _exo_zero，这里只是确保是 bool
        self._exo_zero = bool(self._exo_zero)
        
        
        # reset exo PD/lpf buffers
        if len(self.exo_dof_idx) > 0:
            self._exo_neutral = self.mj_data.qpos[self.exo_dof_idx].astype(np.float32)
            self._exo_ref_raw[:] = self._exo_neutral
            self._exo_ref_lp1[:] = self._exo_neutral
            self._exo_ref_lp2[:] = self._exo_neutral
        else:
            self._exo_ref_raw[:] = 0.0
            self._exo_ref_lp1[:] = 0.0
            self._exo_ref_lp2[:] = 0.0
        
        self.n_exo = len(self.exo_idx)



    
    def forward_sim(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)


    def reset(self, seed=None, options=None):
        self.reset_myolegs()
        self.forward_sim()
        return super().reset(seed=seed, options=options)


def compute_self_observations(body_pos: np.ndarray, body_rot: np.ndarray, body_vel: np.ndarray, body_ang_vel: np.ndarray) -> OrderedDict:
    """
    Computes observations of the agent's local body state relative to its root.

    Args:
        body_pos (np.ndarray): Global positions of the bodies.
        body_rot (np.ndarray): Global rotations of the bodies in quaternion format.
        body_vel (np.ndarray): Linear velocities of the bodies.
        body_ang_vel (np.ndarray): Angular velocities of the bodies.

    Returns:
        OrderedDict: Dictionary containing:
            - `root_h_obs`: Root height observation.
            - `local_body_pos`: Local body positions excluding root.
            - `local_body_rot_obs`: Local body rotations in tangent-normalized format.
            - `local_body_vel`: Local body velocities.
            - `local_body_ang_vel`: Local body angular velocities.
    """
    obs = OrderedDict()
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    
    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    obs["root_h_obs"] = root_h
    
    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"]  = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
     ###### Angular velocity ######
    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )


    return obs

def get_actuator_names(model) -> list:
    """
    Retrieves the names of all actuators in the Mujoco model.

    Args:
        model: The Mujoco model containing actuator information.

    Returns:
        list: A list of actuator names as strings.
    """
    actuators = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in ["name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr", "name_sensoradr"]:
                v = getattr(model, el)
                if np.any(v):
                    if end_p is None:
                        end_p = v[0]
                    else:
                        end_p = min(end_p, v[0])
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i+1]
        name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
        actuators.append(name)
    return actuators

def force_to_activation(forces, model, data):
    """
    Converts actuator forces to activation levels for each actuator in the Mujoco model.

    Args:
        forces (np.ndarray): Array of forces applied to the actuators.
        model: The Mujoco model containing actuator properties.
        data: The Mujoco data structure with runtime actuator states.

    Returns:
        list: Activation levels for each actuator, clipped between 0 and 1.
    """
    activations = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator,:9]
        prmg = model.actuator_gainprm[idx_actuator,:9]
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        activations.append(np.clip((forces[idx_actuator] - bias) / gain, 0, 1))

    return activations

def target_length_to_force(lengths: np.ndarray, data, model) -> list:
    """
    Converts target muscle lengths to forces using a PD control law.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        list: Clipped forces for each actuator, constrained by peak force.
    """
    forces = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        peak_force = model.actuator_biasprm[idx_actuator, 2]
        kp = 5 * peak_force
        kd = 0.1 * kp
        force = (kp * (lengths[idx_actuator] - length) - kd * velocity)
        clipped_force = np.clip(force, -peak_force, 0)
        forces.append(clipped_force)

    return forces

def target_length_to_activation(lengths: np.ndarray, data, model) -> np.ndarray:
    """
    Converts target lengths to activation levels via force computation.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        np.ndarray: Activation levels for each actuator, clipped between 0 and 1.
    """
    forces = target_length_to_force(lengths, data, model)
    activations = force_to_activation(forces, model, data)
    return np.clip(activations, 0, 1)

def action_to_target_length(action: np.ndarray, model) -> list:
    """
    Maps actions to target lengths for actuators based on their length ranges.

    Args:
        action (np.ndarray): Action values in the range [-1, 1].
        model: Mujoco model containing actuator length range properties.

    Returns:
        list: Target lengths for each actuator.
    """
    target_lengths = []
    for idx_actuator in range(model.nu):
        # Set high to max length and low=0
        hi = model.actuator_lengthrange[idx_actuator, 1]
        lo = 0
        target_lengths.append((action[idx_actuator] + 1) / 2 * (hi - lo) + lo)
    return target_lengths