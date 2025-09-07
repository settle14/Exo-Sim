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
import joblib
import numpy as np
from collections import OrderedDict, deque
from omegaconf import DictConfig
from typing import Dict, Iterator, Optional, Tuple
import scipy
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot

from pathlib import Path
import sys
path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

from src.env.myolegs_task import MyoLegsTask
import src.utils.np_transform_utils as npt_utils
from src.utils.visual_capsule import add_visual_capsule
from easydict import EasyDict
from src.KinesisCore.kinesis_core import KinesisCore

import logging

logger = logging.getLogger(__name__)

MYOLEG_TRACKED_BODIES = [
    "root",
    "tibia_l",
    "tibia_r",
    "talus_l",
    "talus_r",
    "toes_l",
    "toes_r",
]

SMPL_TRACKED_IDS = [
    0,
    2,
    6,
    3,
    7,
    4,
    8,
]


class MyoLegsIm(MyoLegsTask):

    def __init__(self, cfg):
        self.initial_pose = None
        self.previous_pose = None
        self.ref_motion_cache = EasyDict()
        self.global_offset = np.zeros([1, 3])
        self.gender_betas = [np.zeros(17)]  # current, all body shape is mean.

        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)

        super().__init__(cfg)
        # Parent __init__ sets self.cfg; now we can safely refresh obs space
        if hasattr(self, "refresh_observation_space"):
            self.refresh_observation_space()
        # === 新增：缓存外骨骼关节的力矩上限，用于 r_tau 归一化 ===
        self._exo_tau_limit = None
        if hasattr(self, "exo_idx") and len(self.exo_idx) > 0:
            rng = self.mj_model.actuator_ctrlrange[self.exo_idx]  # shape: (n_exo, 2)
            self._exo_tau_limit = np.maximum(np.abs(rng).max(axis=1), 1.0)  # 防止除0

        self._exo_hist = deque([], maxlen=3)  # for r_as (2nd-order smoothness)
        self.setup_motionlib()
        self.load_initial_pose_data()
        self.initialize_biomechanical_recording()
        self.initialize_evaluation_metrics()

        self.motions_to_remove = []
        # === exo curriculum scale (0~1), default 1.0 ===
        # 如果父类(MyoLegsEnv)已经在setup_configs里设置了exo_scale，这里不会覆盖
        if not hasattr(self, "exo_scale"):
            self.exo_scale = 1.0


    def initialize_env_params(self, cfg: DictConfig) -> None:
        """
        Initializes environment parameters from the configuration.

        Args:
            cfg (DictConfig): Configuration object.

        Sets:
            - `num_traj_samples`: Number of trajectory samples (default: 1).
            - `reward_specs`: Reward weight specifications.
            - `termination_distance`: Distance threshold for task termination.
        """
        self.num_traj_samples = 1  # cfg.env.get("num_traj_samples", 1) # parameter for number of future time steps
        self.reward_specs = cfg.env.reward_specs
        self.termination_distance = cfg.env.termination_distance

    def initialize_run_params(self, cfg: DictConfig) -> None:
        """
        Initializes run-specific parameters from the configuration.

        Args:
            cfg (DictConfig): Configuration object.

        Sets:
            - Various motion-related parameters (e.g., `motion_start_idx`, `motion_file`).
            - Evaluation and testing flags (e.g., `im_eval`, `test`).
            - Data recording and randomization flags.
        """
        self.motion_start_idx = cfg.run.motion_id
        self.im_eval = cfg.run.im_eval
        self.test = cfg.run.test
        self.num_motion_max = cfg.run.num_motions
        self.motion_file = cfg.run.motion_file
        self.initial_pose_file = cfg.run.initial_pose_file
        # safe default to avoid struct errors if key is missing in YAML
        self.smpl_data_dir = getattr(cfg.run, "smpl_data_dir", "data/smpl")
        # Accept both names for backward-compatibility:
        # prefer 'random_sample'; fall back to legacy 'random_start'
        self.random_sample = getattr(cfg.run, "random_sample",
                                    getattr(cfg.run, "random_start", False))
        self.random_start = getattr(cfg.run, "random_start", False)

        self.recording_biomechanics = cfg.run.recording_biomechanics
        # 新增：训练期随机关掉外骨骼的概率（默认 0）
        self.exo_dropout_prob = float(getattr(cfg.run, "exo_dropout_prob", 0.0))



    def load_initial_pose_data(self) -> None:
        """
        Loads initial pose data from a specified file.

        Checks for the existence of the file defined by `self.initial_pose_file` 
        and loads it using `joblib`. If the file does not exist, initializes 
        an empty dictionary and logs a warning.

        Sets:
            - `self.initial_pos_data`: Loaded pose data or an empty dictionary if the file is not found.
        """
        if os.path.exists(self.initial_pose_file):
            self.initial_pos_data = joblib.load(
                self.initial_pose_file
            )
        else:
            logger.warning("!!! Initial pose data not found !!!")
            self.initial_pos_data = {}

    def initialize_evaluation_metrics(self) -> None:
        """
        Initializes metrics used for evaluating motion performance.

        Sets:
            - `mpjpe` (list): Mean per-joint position error metric.
            - `frame_coverage` (float): Tracks the percentage of frames covered.
        """
        self.mpjpe = []
        self.frame_coverage = 0

    def set_exo_scale(self, scale: float) -> None:
        """
        设置外骨骼扭矩的课程化缩放因子，范围 [0, 1]。
        训练早期设小一点，减少对模仿项的干扰；后期逐步增大到1。
        """
        self.exo_scale = float(np.clip(scale, 0.0, 1.0))

    def initialize_biomechanical_recording(self):
        """
        Initializes storage for biomechanical data recording.

        If `self.recording_biomechanics` is True, prepares lists to store 
        biomechanical data.

        Sets:
            - Various lists for biomechanical recording, including:
            - `self.feet`, `self.joint_pos`, `self.joint_vel`
            - `self.body_pos`, `self.body_rot`, `self.body_vel`
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`
            - `self.motion_id`, `self.muscle_forces`, `self.muscle_controls`, `self.policy_outputs`
        """
        if self.recording_biomechanics:
            self.feet = []
            self.joint_pos = []
            self.joint_vel = []
            self.body_pos = []
            self.body_rot = []
            self.body_vel = []
            self.ref_pos = []
            self.ref_rot = []
            self.ref_vel = []
            self.motion_id = []
            self.muscle_forces = []
            self.muscle_controls = []
            self.policy_outputs = []

    def create_task_visualization(self) -> None:
        """
        Creates visual representations of tracked bodies in the task.

        Adds visual capsules to the viewer and renderer scenes for each tracked body. 
        Capsules are color-coded for differentiation, with red and blue indicating different roles.
        """
        if self.viewer is not None:  # this implies that headless == False
            for _ in range(len(self.track_bodies)):
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 0, 0, 1]),
                )
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([0, 0, 1, 1]),
                )

        if self.renderer is not None:
            for _ in range(len(self.track_bodies)):
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 0, 0, 1]),
                )

    def draw_task(self) -> None:
        """
        Updates the positions of visualized tracked bodies in the scene.

        Synchronizes visual objects in the viewer and renderer with the current 
        task state, using positions from the motion library and simulation.
        """
        def draw_obj(scene):
            sim_time = (
                (self.cur_t) * self.dt
                + self._motion_start_times
                + self._motion_start_times_offset
            )
            ref_dict = self.get_state_from_motionlib_cache(
                self._sampled_motion_ids, sim_time, self.global_offset
            )
            ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]

            for i in range(len(self.track_bodies)):
                scene.geoms[2 * i].pos = ref_pos_subset[0, i]
                scene.geoms[2 * i + 1].pos = self.get_body_xpos()[
                    self.track_bodies_id[i]
                ]

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)

    def setup_myolegs_params(self) -> None:
        """
        Configures body tracking and reset properties for MyoLeg.

        Initializes lists of tracked and resettable bodies, as well as their 
        corresponding indices based on the original body names.

        Sets:
            - `self.full_track_bodies`: List of all original body names.
            - `self.track_bodies`: Names of tracked bodies specific to MyoLeg.
            - `self.reset_bodies`: Names of bodies to reset.
            - `self.track_bodies_id`: Indices of tracked bodies in `self.body_names`.
            - `self.reset_bodies_id`: Indices of reset bodies in `self.body_names`.
        """
        super().setup_myolegs_params()
        self.full_track_bodies = self.body_names
        self.track_bodies = MYOLEG_TRACKED_BODIES
        self.reset_bodies = (
            self.track_bodies
        )
        self.track_bodies_id = [
            self.body_names.index(j) for j in self.track_bodies
        ]
        self.reset_bodies_id = [
            self.body_names.index(j) for j in self.reset_bodies
        ]

    def setup_motionlib(self) -> None:
        """
        Sets up the motion library for managing SMPL motions.

        Configures the motion library with parameters such as data directories, motion files, 
        SMPL type, and randomization settings. Loads motions based on the current mode 
        (test or training), applying shape parameters and optional motion subsets.

        Sets:
            - `self.motion_lib_cfg`: Configuration dictionary for motion library setup.
            - `self.motion_lib`: Instance of `KinesisCore` initialized with the given config.
            - `self._sampled_motion_ids`: Array of sampled motion IDs (default: [0]).
            - `self._motion_start_times`: Start times for the motions.
            - `self._motion_start_times_offset`: Offset times for motion playback.
        """
        self.motion_lib_cfg = EasyDict(
            {
                "data_dir": self.smpl_data_dir,
                "motion_file": self.motion_file,
                "device": torch.device("cpu"),
                "min_length": -1,
                "max_length": -1,
                "multi_thread": True if getattr(self.cfg.run, "num_threads", 1) > 1 else False,
                "smpl_type": "smpl",
                "randomize_heading": not self.test,
            }
        )
        self.motion_lib = KinesisCore(self.motion_lib_cfg)

        # These are initial values that will be updated in reset
        self._sampled_motion_ids = np.array([0])
        self._motion_start_times = np.zeros(1)
        self._motion_start_times_offset = np.zeros(1)
        return

    def sample_motions(self) -> None:
        """
        Samples motions in the motion library based on the current configuration.

        Loads motions into the motion library `self.motion_lib` using the specified configuration, 
        with options for random sampling, custom subsets, and shape parameters.

        Notes:
            - See `KinesisCore.load_motions` for more details on the loading process.
            - The number of motions to load is determined by the length of the `shape_params` argument.
        """
        self.motion_lib.load_motions(
            self.motion_lib_cfg,
            shape_params=self.gender_betas
            * min(self.num_motion_max, self.motion_lib.num_all_motions()),
            random_sample=self.random_sample,
            start_idx=self.motion_start_idx,
        )

    def forward_motions(self) -> Iterator[int]:
        """
        Iterates through motions in the motion library.

        Determines the range of motion IDs to process and sequentially loads each motion 
        into the motion library. Yields the current motion start index during each iteration.

        Yields:
            int: The currently loaded motion index.
        """
        motion_ids = range(self.motion_lib.num_all_motions())
        for motion_start_idx in motion_ids:
            self.motion_start_idx = motion_start_idx
            self.motion_lib.load_motions(
                self.motion_lib_cfg,
                shape_params=self.gender_betas,
                random_sample=self.random_sample,
                silent=False,
                start_idx=self.motion_start_idx,
                specific_idxes=None,
            )
            yield motion_start_idx

    def init_myolegs(self) -> None:
        """
        Initializes the MyoLegs environment state.

        This function sets up the initial state of the simulation, including position, velocity, 
        and kinematics, using motion library data and cached or precomputed initial poses. 
        It also initializes evaluation metrics and biomechanical recording if enabled.
        """
        super().init_myolegs()

        # Initialize motion states and poses
        self.initialize_motion_state()

        # Initialize evaluation metrics
        self.reset_evaluation_metrics()

        # Set up biomechanical recording if enabled
        if self.recording_biomechanics:
            self.delineate_biomechanical_recording()

    def initialize_motion_state(self) -> None:
        """
        Retrieves motion data from the motion library and sets the initial pose.
        """
        motion_return = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, self._motion_start_times, self.global_offset
        )
        initial_rot = sRot.from_euler("XYZ", [-np.pi / 2, 0, -np.pi / 2])
        ref_qpos = motion_return.qpos.flatten()
        self.mj_data.qpos[:3] = ref_qpos[:3]
        rotated_quat = (sRot.from_quat(ref_qpos[[4, 5, 6, 3]]) * initial_rot).as_quat()
        self.mj_data.qpos[3:7] = np.roll(rotated_quat, 1)
        
        if self.im_eval == True:
            motion_id = self.motion_start_idx
        else:
            # All motions are cached, so we just need to start the index from self.motion_start_idx
            motion_id = self._sampled_motion_ids[0] + self.motion_start_idx

        if motion_id in self.initial_pos_data:
            # Load the initial pose from the initial_pos_data
            if self.random_start:
                self.initial_pose = self.initial_pos_data[motion_id][self._motion_start_times[0]]
            else:
                self.initial_pose = self.initial_pos_data[motion_id][0]
            if self.initial_pose is None:
                breakpoint()
            self.mj_data.qpos[7:] = self.initial_pose[7:]
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
        elif self.initial_pose is not None:
            # Constant initial pose
            self.mj_data.qpos[:] = self.initial_pose
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
        else:
            # During IK, the humanoid sometimes reaches unfeasible positions. If that happens, we flag the motion and remove it from the dataset.
            if self.mj_data.qpos[2] < 0.86 or self.mj_data.qpos[2] > 1:
                if self.motion_lib._curr_motion_ids[0] not in self.motions_to_remove:
                    self.motions_to_remove.append(self.motion_lib._curr_motion_ids[0])
                    print(f"Motion {self.motions_to_remove[-1]} removed")
            else:
                # If the motion is flagged, just skip, otherwise compute the initial pose on the fly
                if self.motion_lib._curr_motion_ids[0] not in self.motions_to_remove:
                    self.compute_initial_pose()

        # Set up velocity
        ref_qvel = motion_return.qvel.flatten()[:6]
        self.mj_data.qvel[:3] = ref_qvel[:3]
        self.mj_data.qvel[3:6] = initial_rot.inv().apply(ref_qvel[3:6])
        self.mj_data.qvel[6:] = np.zeros_like(self.mj_data.qvel[6:])

        # Run kinematics
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def reset_evaluation_metrics(self) -> None:
        """
        Resets evaluation metrics for motion imitation performance.
        """
        self.mjpe = []
        self.mjve = []

    def delineate_biomechanical_recording(self) -> None:
        """
        Adds a nan buffer to the biomechanical recording lists to indicate that a new episode has started.
        """
        self.feet.append(np.nan)
        self.joint_pos.append(np.full(self.get_qpos().shape, np.nan))
        self.joint_vel.append(np.full(self.get_qvel().shape, np.nan))
        self.body_pos.append(np.full(self.get_body_xpos()[None,].shape, np.nan))
        self.body_rot.append(np.full(self.get_body_xquat()[None,][..., self.track_bodies_id, :].shape, np.nan))
        self.body_vel.append(np.full(self.get_body_linear_vel()[None,][..., self.track_bodies_id, :].shape, np.nan))
        self.ref_pos.append(np.full(self.get_body_xpos()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.ref_rot.append(np.full(self.get_body_xquat()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.ref_vel.append(np.full(self.get_body_linear_vel()[None,][..., SMPL_TRACKED_IDS, :].shape, np.nan))
        self.motion_id.append(np.nan)
        self.muscle_forces.append(np.full(self.get_muscle_force().shape, np.nan))
        self.muscle_controls.append(np.full(self.mj_data.ctrl.shape, np.nan))
        self.policy_outputs.append(np.full(self.mj_data.ctrl.shape, np.nan))

    def get_task_obs_size(self) -> int:
        """
        Calculates the size of the task observation vector based on configured inputs.

        This function sums up the dimensions of the observation components specified 
        in `self.cfg.run.task_inputs`. Each component contributes to the observation size 
        based on the number of tracked bodies and its dimensionality (e.g., 3 for position or velocity).

        Returns:
            int: The total size of the task observation vector.
        """
        inputs = self.cfg.run.task_inputs
        obs_size = 0
        if "diff_local_body_pos" in inputs:
            obs_size += 3 * len(self.track_bodies)
        if "diff_local_vel" in inputs:
            obs_size += 3 * len(self.track_bodies)
        if "local_ref_body_pos" in inputs:
            obs_size += 3 * len(self.track_bodies)

        return obs_size

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should reset based on termination and truncation conditions.

        This function checks if the task should terminate early due to positional deviation 
        from reference motions (termination) or if the task duration has exceeded the 
        motion length (truncation). If either condition is met, evaluation metrics are computed 
        and reset.

        Returns:
            Tuple containing
            - `terminated` (bool): True if the task exceeded the termination distance.
            - `truncated` (bool): True if the task exceeded the duration of the episode.

        Updates:
            - Calls `compute_evaluation_metrics` and `reset_evaluation_metrics` if reset conditions are met.
        """
        terminated, truncated = False, False
        sim_time = (
            (self.cur_t) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, sim_time, self.global_offset
        )
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        body_pos_subset = body_pos[..., self.reset_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]
        terminated = compute_humanoid_im_reset(
            body_pos_subset,
            ref_pos_subset,
            termination_distance=self.termination_distance,
            use_mean=True if self.im_eval else False,
        )[0]
        truncated = (
            sim_time > self.motion_lib.get_motion_length(self._sampled_motion_ids)
        )[0]

        if terminated or truncated:
            self.compute_evaluation_metrics(terminated, sim_time)
            self.reset_evaluation_metrics()

        return terminated, truncated
    
    def compute_evaluation_metrics(self, terminated, sim_time) -> None:
        """
        Computes evaluation metrics for the current simulation.

        This function calculates the mean per-joint position error (MPJPE) and frame 
        coverage for the simulation. The frame coverage 
        indicates the proportion of the motion completed before termination or completion.

        Args:
            terminated (bool): Indicates whether the simulation terminated early.
            sim_time (np.ndarray): Current time index for the simulation.

        Updates:
            - `self.mpjpe_value`: Average MPJPE across all frames.
            - `self.frame_coverage`: Ratio of completed frames to total motion length.
        """
        self.mpjpe_value = np.array(self.mpjpe).mean()
        if terminated:
            self.frame_coverage = sim_time / self.motion_lib.get_motion_length(self._sampled_motion_ids)
        else:
            self.frame_coverage = 1.0

    def reset_task(self, options: Optional[dict]=None) -> None:
        """
        Resets the task to an initial state based on the current configuration.

        This function initializes motion sampling and start times, considering the mode 
        (test or training), evaluation settings, and optional start time configurations. 
        Random starting times can also be applied if enabled.

        Args:
            options (dict, optional): A dictionary containing reset options. Supports:
                - `start_time`: Specifies a custom start time for the motion.

        Updates:
            - `self._sampled_motion_ids`: IDs of the motions to use after reset.
            - `self._motion_start_times`: Start times for the motions, either specified, 
            randomized, or set to zero.

        Notes:
            - If `self.random_start` is True, the start time is randomly selected from the
            available time indices for which an initial pose is available.
        """
        if self.test:
            if self.im_eval:
                self._sampled_motion_ids[:] = 0  # options['motion_id']
                self._motion_start_times[:] = 0
                if options is not None and "start_time" in options:
                    self._motion_start_times[:] = options["start_time"]
            else:
                self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
                self._motion_start_times[:] = 0
                if options is not None and "start_time" in options:
                    self._motion_start_times[:] = options["start_time"]
                elif self.random_start:
                    motion_id = self.get_true_motion_id()
                    # sample from the keys of initial_pos_dict[motion_id]
                    start_time = np.random.choice(list(self.initial_pos_data[motion_id].keys()))
                    self._motion_start_times[:] = start_time
        else:
            self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
            self._motion_start_times[:] = 0
            if options is not None and "start_time" in options:
                self._motion_start_times[:] = options["start_time"]
            elif self.random_start:
                motion_id = self.get_true_motion_id()
                # sample from the keys of initial_pos_dict[motion_id]
                start_time = np.random.choice(list(self.initial_pos_data[motion_id].keys()))
                self._motion_start_times[:] = start_time
        # === 新增：按模式设置外骨骼是否置零 ===
        # 测试/评估：遵循命令行或 yaml 传入的 run.exo_zero
        if self.test:
            if hasattr(self, "set_exo_zero"):
                self.set_exo_zero(bool(getattr(self.cfg.run, "exo_zero", False)))
            if hasattr(self, "_exo_hist"):
                self._exo_hist.clear()  # 确保每个 episode 的平滑度度量从头算

        else:
            # 训练：按概率随机关掉外骨骼，避免过拟合外骨骼输出
            p = float(getattr(self.cfg.run, "exo_zero", 0.0))  # 若也想全程关，可直接传 1.0
            if self.exo_dropout_prob > 0.0:
                p = 1.0 - (1.0 - p) * (1.0 - self.exo_dropout_prob)  # 合并：显式 exo_zero 与 dropout
            if hasattr(self, "set_exo_zero"):
                self.set_exo_zero(np.random.rand() < p)
                # === 新增：每个 episode 重置外骨骼动作历史（用于 r_as 的二阶差分）===
            if hasattr(self, "_exo_hist"):
                self._exo_hist.clear()

    
    def get_true_motion_id(self) -> int:
        """
        Calculates the true motion ID based on the current configuration.

        Returns:
            int: The true motion ID in the full motion library
        """
        motion_id = self._sampled_motion_ids[0] + self.motion_start_idx
        return motion_id

    def get_state_from_motionlib_cache(self, motion_ids: np.ndarray, motion_times: np.ndarray, offset: Optional[np.ndarray]=None) -> dict:
        """
        Retrieves the motion state from the motion library, with caching for efficiency.

        This function checks if the requested motion state (defined by `motion_ids`, 
        `motion_times`, and `offset`) is already cached. If the cache is valid, it returns 
        the cached state. Otherwise, it updates the cache with new data from the motion library.

        Args:
            motion_ids (np.ndarray): IDs of the motions to retrieve.
            motion_times (np.ndarray): Time indices for the motions.
            offset (np.ndarray, optional): Offset to apply to the motions. Defaults to None.

        Returns:
            dict: Cached or newly retrieved motion state data, containing all the values required
            for motion imitation.

        Updates:
            - `self.ref_motion_cache`: Stores the motion IDs, times, offsets, and motion state 
            data for reuse.
        """
        if (
            offset is None
            or not "motion_ids" in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["motion_ids"]) != len(motion_ids)
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or np.abs(self.ref_motion_cache["motion_ids"] - motion_ids).sum()
            + np.abs(self.ref_motion_cache["motion_times"] - motion_times).sum()
            + np.abs(self.ref_motion_cache["offset"] - offset).sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = (
                motion_ids.copy()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = (
                motion_times.copy()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = (
                offset.copy() if not offset is None else None
            )
            motion_res = self.motion_lib.get_motion_state_intervaled(
                motion_ids.copy(), motion_times.copy(), offset=offset
            )

            self.ref_motion_cache.update(motion_res)

            return self.ref_motion_cache
        
        else:
            return self.ref_motion_cache

    def compute_task_obs(self) -> np.ndarray:
        """
        Computes task-specific observations for the current simulation step.

        This function calculates and returns the observation vector based on the 
        current and reference states of the simulated bodies. It includes positional 
        and velocity differences as well as reference positions, tailored to the 
        configured task inputs.

        Returns:
            np.ndarray: A concatenated array of task observations based on selected 
            input features. The array is flattened for compatibility with downstream models.

        Updates:
            - Calls `record_evaluation_metrics` to update position and velocity metrics.
            - Calls `record_biomechanics` to store biomechanical data if enabled.

        Observation Features (if configured):
            - `diff_local_body_pos`: Differences in local body positions relative to references.
            - `diff_local_vel`: Differences in local body velocities relative to references.
            - `local_ref_body_pos`: Local reference body positions.
        """
        motion_times = (
            (self.cur_t + 1) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, motion_times, self.global_offset
        )

        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]

        body_pos_subset = body_pos[..., self.track_bodies_id, :]
        body_rot_subset = body_rot[..., self.track_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]
        ref_rot_subset = ref_dict.xquat[..., SMPL_TRACKED_IDS, :]

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.track_bodies_id, :]
        ref_body_vel_subset = ref_dict.body_vel[..., SMPL_TRACKED_IDS, :]

        if self.recording_biomechanics:
            self.record_biomechanics(body_pos_subset, body_rot_subset, body_vel_subset, ref_pos_subset, ref_rot_subset, ref_body_vel_subset)

        full_task_obs = compute_imitation_observations(
            root_pos,
            root_rot,
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            self.num_traj_samples,
        )

        task_obs = {}
        if "diff_local_body_pos" in self.cfg.run.task_inputs:
            task_obs["diff_local_body_pos"] = full_task_obs["diff_local_body_pos"]
        if "diff_local_vel" in self.cfg.run.task_inputs:
            task_obs["diff_local_vel"] = full_task_obs["diff_local_vel"]
        if "local_ref_body_pos" in self.cfg.run.task_inputs:
            task_obs["local_ref_body_pos"] = full_task_obs["local_ref_body_pos"]

        return np.concatenate([v.ravel() for v in task_obs.values()], axis=0).astype(self.dtype, copy=False)


    def record_biomechanics(self, 
                            body_pos: np.ndarray, 
                            body_rot: np.ndarray,
                            body_vel: np.ndarray, 
                            ref_pos: np.ndarray, 
                            ref_rot: np.ndarray, 
                            ref_vel: np.ndarray
                            ) -> None:
        """
        Records biomechanical data for the current simulation step.

        Captures and stores data related to body states, reference states, joint 
        positions and velocities, and muscle forces/controls for biomechanical analysis.

        Args:
            body_pos (np.ndarray): Current body positions.
            body_rot (np.ndarray): Current body rotations.
            body_vel (np.ndarray): Current body velocities.
            ref_pos (np.ndarray): Reference body positions.
            ref_rot (np.ndarray): Reference body rotations.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.feet`: Tracks foot contact states (e.g., left, right, or both planted).
            - `self.joint_pos`, `self.joint_vel`: Joint positions and velocities.
            - `self.body_pos`, `self.body_rot`, `self.body_vel`: Current body states.
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`: Reference body states.
            - `self.motion_id`: Current motion ID.
            - `self.muscle_forces`, `self.muscle_controls`: Muscle forces and control inputs.
        """
        feet_contacts = self.proprioception["feet_contacts"]
        planted_feet = -1
        if feet_contacts[0] > 0 or feet_contacts[1] > 0:
            planted_feet = 1
        if feet_contacts[2] > 0 or feet_contacts[3] > 0:
            planted_feet = 0
        if (feet_contacts[0] > 0 or feet_contacts[1] > 0) and (feet_contacts[2] > 0 or feet_contacts[3] > 0):
            planted_feet = 2
        self.feet.append(planted_feet)
        self.joint_pos.append(self.get_qpos().copy())
        self.joint_vel.append(self.get_qvel().copy())
        self.body_pos.append(body_pos.copy())
        self.body_rot.append(body_rot.copy())
        self.body_vel.append(body_vel.copy())
        self.ref_pos.append(ref_pos.copy())
        self.ref_rot.append(ref_rot.copy())
        self.ref_vel.append(ref_vel.copy())
        self.motion_id.append(self.motion_start_idx)
        self.muscle_forces.append(self.get_muscle_force().copy())
        self.muscle_controls.append(self.mj_data.ctrl.copy())

    def record_evaluation_metrics(self, 
                                  body_pos: np.ndarray, 
                                  ref_pos: np.ndarray, 
                                  body_vel: np.ndarray, 
                                  ref_vel: np.ndarray
                                  ) -> None:
        """
        Records evaluation metrics (MPJPE) for the current simulation step.

        Args:
            body_pos (np.ndarray): Current body positions.
            ref_pos (np.ndarray): Reference body positions.
            body_vel (np.ndarray): Current body velocities.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.mpjpe`: Appends the mean position error for the current step.
        """
        self.mpjpe.append(np.linalg.norm(body_pos - ref_pos, axis=-1).mean())

    def compute_muscle_work_proxy(self) -> float:
        """
        Prefer instantaneous positive muscle power sum (F * -v, positive part) IF muscle fiber
        velocity is available from the model; otherwise fallback to average squared activation.
        Note: MuJoCo default mjData has no 'actuator_velocity'; this path typically falls back.
        """

        # --- Option A: true instantaneous power from MuJoCo ---
        try:
            # use env indices
            mus_ids = getattr(self, "muscle_idx", None)
            if mus_ids is not None and len(mus_ids) > 0:
                F = self.mj_data.actuator_force[mus_ids]
                v = self.mj_data.actuator_velocity[mus_ids]  # fiber velocity proxy
                P = F * (-v)                  # positive when muscle does work on skeleton
                P_pos = float(np.maximum(P, 0.0).sum())
                return P_pos / 1000.0         # normalize to O(1)
        except Exception:
            pass

        # --- Option B: activation proxy (human only) ---
        act = getattr(self, "action_human", None)
        if act is None:
            # if last_action stores [human, exo] concatenated, strip exo tail
            la = getattr(self, "last_action", None)
            if la is not None and hasattr(self, "exo_idx") and len(self.exo_idx) > 0:
                act = la[:-len(self.exo_idx)]
            else:
                act = la
        if act is None: 
            return 0.0
        return float((act**2).mean())



    def compute_exo_smoothness(self, a_exo_curr: np.ndarray, sigma_as: float = 1.0) -> float:
        """
        r_as：动作平滑度（论文二阶差分）：exp( - sigma_as * ||a_t - 2a_{t-1} + a_{t-2}||^2 )
        缓冲不够三帧时返回 1.0（不处罚）。
        """
        self._exo_hist.append(a_exo_curr.copy())
        if len(self._exo_hist) < 3:
            return 1.0
        a_t  = self._exo_hist[-1]
        a_t1 = self._exo_hist[-2]
        a_t2 = self._exo_hist[-3]
        jerk = a_t - 2.0 * a_t1 + a_t2
        return float(np.exp(-sigma_as * float(np.dot(jerk, jerk))))


    
    def compute_reward(self, action: Optional[np.ndarray] = None) -> float:
        """
        Computes the reward for the current simulation step.

        The reward is a combination of imitation reward, upright posture reward, 
        and energy efficiency. It is calculated by comparing the current body state 
        to the reference motion and includes weighted contributions based on the 
        reward specifications.
        """
        # ===== 取参考状态 =====
        motion_times = (
            (self.cur_t) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        ref_dict = self.get_state_from_motionlib_cache(
            self._sampled_motion_ids, motion_times, self.global_offset
        )

        # ===== 取当前身体状态（已按 track_bodies 对齐）=====
        body_pos = self.get_body_xpos()[None,]
        body_pos_subset    = body_pos[..., self.track_bodies_id, :]
        ref_pos_subset     = ref_dict.xpos[..., SMPL_TRACKED_IDS, :]

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset    = body_vel[..., self.track_bodies_id, :]
        ref_body_vel_subset= ref_dict.body_vel[..., SMPL_TRACKED_IDS, :]

        # ===== 模仿奖励（root 对齐 + 去掉 root 本体；r_body_pos / r_vel 已是 float 标量）=====
        reward_im, reward_raw = compute_imitation_reward(
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            self.reward_specs,
        )

        # 组合出 imitation 的加权总和，便于日志观察（不改变 reward_im 本身）
        w_pos = float(self.reward_specs.get("w_pos", 1.0))
        w_vel = float(self.reward_specs.get("w_vel", 0.5))
        r_im  = float(w_pos * reward_raw.get("r_body_pos", 0.0) +
                    w_vel * reward_raw.get("r_vel", 0.0))
        reward = float(reward_im)
        reward_raw["r_im"] = r_im  # 仅记录

        # ===== Upright =====
        w_upright = float(self.reward_specs.get("w_upright", 0.0))
        upright_reward = float(self.compute_upright_reward())
        reward += w_upright * upright_reward

        # ===== paper-inspired exo terms: r_m, r_as, r_tau =====
        # 1) r_m : 肌肉功/努力代理（指数整形）
        sigma_m  = float(self.reward_specs.get("sigma_m", 1.0))
        work_val = float(self.compute_muscle_work_proxy())
        r_m = float(np.exp(-sigma_m * work_val))

        # 2) r_as : 外骨骼动作二阶平滑
        n_exo = len(getattr(self, "exo_idx", []))
        if hasattr(self, "last_action") and (self.last_action is not None) and n_exo > 0:
            a_exo = np.asarray(self.last_action[-n_exo:], dtype=np.float32)
        else:
            a_exo = np.zeros(n_exo, dtype=np.float32)

        sigma_as = float(self.reward_specs.get("sigma_as", 1.0))
        exo_on = (not bool(getattr(self, "_exo_zero", False))) and (n_exo > 0)
        if not exo_on:
            r_as = 1.0
            if hasattr(self, "_exo_hist"):
                self._exo_hist.clear()   # 关时清历史
        else:
            r_as = float(self.compute_exo_smoothness(a_exo, sigma_as=sigma_as))

        # 3) r_tau : 外骨骼能耗惩罚 -> 奖励（用 torque^2）
        if not exo_on:
            tau = np.zeros(n_exo, dtype=np.float32)
        else:
            # 优先使用“滤波 + 限幅后”的力矩；没有时退回原始 tau_exo
            tau = np.asarray(
                getattr(self, "tau_exo_filt",
                        getattr(self, "tau_exo", np.zeros(n_exo, dtype=np.float32))),
                dtype=np.float32
            )

        if getattr(self, "_exo_tau_limit", None) is not None:
            tau_norm = tau / self._exo_tau_limit
            tau_cost = float(np.dot(tau_norm, tau_norm))   # 归一化平方和
        else:
            tau_cost = float(np.dot(tau, tau)) * 1e-4      # 原始扭矩的缩放平方和

        sigma_tau = float(self.reward_specs.get("sigma_tau", 1.0))
        r_tau = float(np.exp(-sigma_tau * tau_cost))
        reward_raw["tau_sq"] = float(tau_cost)

        # ===== 与 imitation / upright 融合 =====
        w_m   = float(self.reward_specs.get("w_m",   0.5))
        w_as  = float(self.reward_specs.get("w_as",  0.2))
        w_tau = float(self.reward_specs.get("w_tau", 0.1))

        # 外骨骼相关三项随 exo_scale 渐进；人本体 imitation / upright 不乘 s
        s = float(getattr(self, "exo_scale", 1.0))  # 0~1，由外部课程化控制
        reward += s * (w_m * r_m + w_as * r_as + w_tau * r_tau)

        # ===== 能耗观测的记录（不参与 reward；避免 IDE 警告灰色）=====
        energy_human = float(np.mean(self.curr_power_usage)) if hasattr(self, "curr_power_usage") and len(self.curr_power_usage) > 0 else 0.0
        exo_energy   = float(np.mean(self.curr_exo_usage))  if hasattr(self, "curr_exo_usage")  and len(self.curr_exo_usage)  > 0 else 0.0
        exo_smooth   = float(np.mean(self.curr_exo_rate))   if hasattr(self, "curr_exo_rate")   and len(self.curr_exo_rate)   > 0 else 0.0

        if hasattr(self, "clear_energy_buffers"):
            self.clear_energy_buffers()
        else:
            self.curr_power_usage = []

        # 可选：仅人能耗作为附加项（权重通常很小，默认 0）
        reward += energy_human * float(self.reward_specs.get("w_energy_human", 0.0))

        # ===== 组装 reward_info（供 wandb 打印）=====
        reward_raw["r_m"]   = float(r_m)
        reward_raw["r_as"]  = float(r_as)
        reward_raw["r_tau"] = float(r_tau)
        reward_raw["work_val_proxy"] = float(work_val)
        reward_raw["exo_scale"] = s

        # 透传 MSE（来自 compute_imitation_reward 内部）
        self.reward_info = reward_raw
        self.reward_info["upright_reward"] = upright_reward
        self.reward_info["energy_human"]   = energy_human
        self.reward_info["exo_energy"]     = exo_energy     # ← 记录，不参与 reward
        self.reward_info["exo_smooth"]     = exo_smooth     # ← 记录，不参与 reward
        self.reward_info["exo_on"]         = float(exo_on)

        # ===== 数值健壮性保护 =====
        if not np.isfinite(reward):
            # 避免 NaN/Inf 传播；做个标注，便于排查
            self.reward_info["nan_guard"] = 1.0
            reward = 0.0
        else:
            self.reward_info["nan_guard"] = 0.0

        # 评估用的额外记录（你已有的函数）
        self.record_evaluation_metrics(body_pos_subset, ref_pos_subset, body_vel_subset, ref_body_vel_subset)

        return float(reward)

    
    def compute_upright_reward(self) -> float:
        """
        Computes the reward for maintaining an upright posture.

        The reward is based on the angles of tilt in the forward and sideways directions, 
        calculated using trigonometric components of the root tilt.

        Returns:
            float: The upright reward, where a value close to 1 indicates a nearly upright posture.
        """
        upright_trigs = self.proprioception['root_tilt']
        fall_forward = np.angle(upright_trigs[0] + 1j * upright_trigs[1])
        fall_sideways = np.angle(upright_trigs[2] + 1j * upright_trigs[3])
        upright_reward = np.exp(-3 * (fall_forward ** 2 + fall_sideways ** 2))
        return upright_reward

    def compute_energy_reward(self, action: np.ndarray) -> float:
        """
        Computes the energy efficiency reward based on the L1 and L2 norms of the action.

        The reward penalizes high energy usage, with an exponential scaling defined 
        by a configurable parameter.

        Args:
            action (np.ndarray): The action vector applied at the current step.

        Returns:
            float: The energy reward, where higher values indicate more efficient energy usage.
        """
        l1_energy = float(np.abs(action).sum())
        l2_energy = float(np.linalg.norm(action))
        cost = l1_energy + l2_energy
        k_energy = float(self.reward_specs.get("k_energy", 1e-3))  # 正数
        energy_reward = float(np.exp(-k_energy * cost))
        return energy_reward


    def start_eval(self, im_eval=True):
        """
        Prepares the environment for evaluation.

        Args:
            im_eval (bool): Whether to enable imitation evaluation mode. Defaults to True.
        """
        self.motion_lib_cfg.randomize_heading = False
        self.im_eval = im_eval
        self.test = True

        self._temp_termination_distance = self.termination_distance

    def end_eval(self):
        """
        Concludes the evaluation process and restores training settings.
        """
        self.motion_lib_cfg.randomize_heading = True
        self.im_eval = False
        self.test = False
        self.termination_distance = self._temp_termination_distance
        self.sample_motions()

    def get_muscle_force(self) -> np.ndarray:
        """
        Get current muscle forces from MuJoCo simulation.
        
        Returns:
            np.ndarray: Array of muscle forces with shape (n_muscles,)
        """
        # Use step-captured data if available (most accurate)
        if hasattr(self, '_current_step_muscle_forces') and self._current_step_muscle_forces is not None:
            return self._current_step_muscle_forces.copy()
        
        # Fallback to direct access
        if hasattr(self, 'mj_data') and hasattr(self.mj_data, 'actuator_force'):
            forces = self.mj_data.actuator_force.copy()
            forces = np.where(np.isfinite(forces), forces, 0.0)
            return forces
        else:
            # Final fallback: return zeros
            return np.zeros(self.mj_model.nu, dtype=np.float32)

    
    def get_muscle_controls(self) -> np.ndarray:
        """
        Get current muscle control signals from MuJoCo simulation.
        
        Returns:
            np.ndarray: Array of muscle control values with shape (n_muscles,)
        """
        # Use step-captured data if available (most accurate)
        if hasattr(self, '_current_step_muscle_controls') and self._current_step_muscle_controls is not None:
            return self._current_step_muscle_controls.copy()
        
        # Fallback to direct access
        if hasattr(self, 'mj_data') and hasattr(self.mj_data, 'ctrl'):
            controls = self.mj_data.ctrl.copy()
            controls = np.where(np.isfinite(controls), controls, 0.0)
            return controls
        else:
            # Final fallback: return zeros
            return np.zeros(self.mj_model.nu, dtype=np.float32)


    def compute_initial_pose(self, ref_dict: Optional[dict] = None) -> None:
        """
        Computes the initial pose by optimizing joint positions to minimize deviations from defaults 
        while aligning the body positions with reference positions.

        Uses a constrained optimization method to determine the optimal joint configuration.

        Args:
            ref_dict (Optional[dict]): Reference motion data containing target positions. 
                If None, reference data is retrieved from the motion library cache.

        Notes:
            - Joint bounds are derived from the Mujoco model's joint range.
        """
        initial_qpos = self.mj_data.qpos.copy()
        if ref_dict is None:
            ref_dict = self.get_state_from_motionlib_cache(
                self._sampled_motion_ids, self._motion_start_times, self.global_offset
            )
        ref_pos_subset = ref_dict.xpos[..., SMPL_TRACKED_IDS[1:], :]  # remove root

        joint_range = self.mj_model.jnt_range.copy()
        bounds = joint_range[1:, :]
        # make each row a 2-tuple
        bounds = [tuple(b) for b in bounds]


        def distance_to_default(qpos):
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            return np.linalg.norm(qpos - initial_qpos[7:]) * 5

        def distance_to_ref(qpos):
            self.mj_data.qpos[7:] = qpos
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            body_pos = self.get_body_xpos()[None,]
            body_pos_subset = body_pos[..., self.track_bodies_id[1:], :]  # remove root
            return np.linalg.norm(body_pos_subset - ref_pos_subset, axis=-1).sum()

        out = scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            # x0=self.mj_data.qpos[7:],
            x0=self.previous_pose[7:] if self.previous_pose is not None else initial_qpos[7:],
            eqcons=[distance_to_ref],
            bounds=bounds,
            iprint=1,
            iter=200,
            acc=0.02,
        )

        self.initial_pose = np.concatenate([initial_qpos[:7], out])

    def post_physics_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Processes the environment state after each physics step.

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

    def step(self, action):
        """
        Executes a single step in the environment with the given action.

        Args:
            action: The action to apply at the current step.

        Returns:
            Tuple:
                - observation (np.ndarray): Current observations after the step.
                - reward (float): Reward for the applied action.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information about the step, including reward details.
        """
        if self.recording_biomechanics:
            self.policy_outputs.append(action)

        self.last_action = np.asarray(action).copy()

        self.physics_step(action)
        observation, reward, terminated, truncated, info = self.post_physics_step(action)
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


def compute_imitation_observations(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    time_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Computes imitation observations based on differences between current and reference states.

    Observations include local differences in body positions and velocities, 
    as well as local reference positions relative to the root.

    Args:
        root_pos (np.ndarray): Root position of the current state.
        root_rot (np.ndarray): Root rotation of the current state.
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        time_steps (int): Number of time steps for observation history.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - `diff_local_body_pos`: Differences in local body positions.
            - `diff_local_vel`: Differences in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.
    """
    obs = OrderedDict()
    B, J, _ = body_pos.shape

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)

    heading_inv_rot_expand = np.tile(
        heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis=1),
        (time_steps, 1, 1),
    )

    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(
        B, 1, J, 3
    )

    diff_local_body_pos_flat = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3)
    )

    obs["diff_local_body_pos"] = diff_local_body_pos_flat  # 1 * J * 3

    ##### Velocities
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(
        B, 1, J, 3
    )
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3)
    )

    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(
        B, 1, 1, 3
    )  # preserves the body position
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)
    )

    return obs


def compute_imitation_reward(
    body_pos, body_vel, ref_body_pos, ref_body_vel, rwd_specs,
    pos_joint_w=None, vel_joint_w=None,
):
    """
    Root-aligned imitation reward:
    - 对齐各自 root 的平移（去绝对位移）
    - 去掉 root 本体，只比较四肢/非 root 段
    - 可选逐段加权（pos_joint_w/vel_joint_w），长度需与 track_bodies 对齐
    """
    # 温和默认 + 轻度限幅（避免指数饱和）
    k_pos = float(np.clip(float(rwd_specs.get("k_pos", 30.0)), 1e-6, 100.0))
    k_vel = float(np.clip(float(rwd_specs.get("k_vel", 0.50)), 1e-6, 100.0))
    w_pos = float(rwd_specs.get("w_pos", 1.0))
    w_vel = float(rwd_specs.get("w_vel", 0.5))

    # --- 关键：对齐根平移，并去掉 root 本体（index 0） ---
    body_pos_rel = body_pos - body_pos[:, :1, :]
    ref_pos_rel  = ref_body_pos - ref_body_pos[:, :1, :]
    body_vel_rel = body_vel - body_vel[:, :1, :]
    ref_vel_rel  = ref_body_vel - ref_body_vel[:, :1, :]

    body_pos_rel = body_pos_rel[:, 1:, :]
    ref_pos_rel  = ref_pos_rel[:, 1:, :]
    body_vel_rel = body_vel_rel[:, 1:, :]
    ref_vel_rel  = ref_vel_rel[:, 1:, :]

    # 位置项
    dpos = ref_pos_rel - body_pos_rel
    if pos_joint_w is not None:
        w = np.asarray(pos_joint_w[1:], dtype=np.float32).reshape(1, -1, 1)
        pos_mse = ((dpos**2) * w).sum(axis=(1,2)) / (w.sum() + 1e-6)
    else:
        pos_mse = (dpos**2).mean(axis=(1,2))
    r_body_pos = np.exp(-k_pos * pos_mse)

    # 速度项
    dvel = ref_vel_rel - body_vel_rel
    if vel_joint_w is not None:
        wv = np.asarray(vel_joint_w[1:], dtype=np.float32).reshape(1, -1, 1)
        vel_mse = ((dvel**2) * wv).sum(axis=(1,2)) / (wv.sum() + 1e-6)
    else:
        vel_mse = (dvel**2).mean(axis=(1,2))
    r_vel = np.exp(-k_vel * vel_mse)

    # 组合 + 把可诊断信息写成 float
    reward = w_pos * r_body_pos + w_vel * r_vel
    r_body_pos_scalar = float(np.asarray(r_body_pos).mean())
    r_vel_scalar      = float(np.asarray(r_vel).mean())
    reward_raw = {
        "r_body_pos": r_body_pos_scalar,
        "r_vel":      r_vel_scalar,
        "pos_mse":    float(np.asarray(pos_mse)[0]),
        "vel_mse":    float(np.asarray(vel_mse)[0]),
    }
    return float(np.asarray(reward)[0]), reward_raw



def compute_humanoid_im_reset(
    rigid_body_pos, ref_body_pos, termination_distance, use_mean
) -> np.ndarray:
    """
    Determines whether the humanoid should reset based on deviations from reference positions.

    Args:
        rigid_body_pos (np.ndarray): Current positions of the humanoid's rigid bodies.
        ref_body_pos (np.ndarray): Reference positions of the humanoid's rigid bodies.
        termination_distance (float): Threshold distance for termination.
        use_mean (bool): Whether to use the mean or maximum deviation for the reset condition.

    Returns:
        bool: Indicates whether the humanoid has exceeded the termination distance.
    """
    if use_mean:
        has_fallen = np.any(
            np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1).mean(
                axis=-1, keepdims=True
            )
            > termination_distance,
            axis=-1,
        )
    else:
        has_fallen = np.any(
            np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1)
            > termination_distance,
            axis=-1,
        )

    return has_fallen