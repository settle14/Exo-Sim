# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

import math
from collections import defaultdict
import numpy as np

class LoggerRL:

    def __init__(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.avg_episode_len = 0
        self.total_reward = 0
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.min_reward = math.inf
        self.max_reward = -math.inf
        self.episode_reward = 0
        self.avg_episode_reward = 0
        self.sample_time = 0
        self.info_dict = defaultdict(list)
        

    def start_episode(self, env):
        self.episode_reward = 0
    
    def start_sampling(self):
        """
        Reset all counters/stats before a new sampling phase.
        """
        self.num_steps = 0
        self.num_episodes = 0
        self.avg_episode_len = 0.0
        self.total_reward = 0.0
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.min_reward = math.inf
        self.max_reward = -math.inf
        self.episode_reward = 0.0
        self.avg_episode_reward = 0.0
        self.sample_time = 0.0
        # 清空每轮的 info 聚合（配合 assist/* 新指标）
        self.info_dict = defaultdict(list)


    def step(self, env, reward, info):
        self.episode_reward += reward
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.num_steps += 1
        for k, v in info.items():
            self.info_dict[k].append(v)


    def end_episode(self, env):
        self.num_episodes += 1
        self.total_reward += self.episode_reward
        self.min_episode_reward = min(self.min_episode_reward, self.episode_reward)
        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)

    def end_sampling(self):
        self.avg_episode_len = (self.num_steps / self.num_episodes) if self.num_episodes > 0 else 0.0
        self.avg_episode_reward = (self.total_reward / self.num_episodes) if self.num_episodes > 0 else 0.0

    def add_info_dict(self, d: dict):
        """
        Add a whole dict of scalars to info_dict.
        Non-scalar values are ignored.
        """
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            try:
                # 只记录能安全转成 float 的值
                self.info_dict[k].append(float(v))
            except Exception:
                # 跳过 list/ndarray/非数值类型
                continue



    @classmethod
    def merge(cls, logger_list):
        # 先过滤 None，避免聚合时报错
        valid = [lg for lg in logger_list if isinstance(lg, cls)]
        logger = cls()
        if len(valid) == 0:
            return logger  # 空壳返回，避免下游崩

        logger.total_reward  = sum(x.total_reward  for x in valid)
        logger.num_episodes  = sum(x.num_episodes  for x in valid)
        logger.num_steps     = sum(x.num_steps     for x in valid)

        logger.avg_episode_len    = (logger.num_steps / logger.num_episodes) if logger.num_episodes > 0 else 0.0
        logger.avg_episode_reward = (logger.total_reward / logger.num_episodes) if logger.num_episodes > 0 else 0.0
        logger.max_episode_reward = max(x.max_episode_reward for x in valid)
        logger.min_episode_reward = min(x.min_episode_reward for x in valid)
        logger.avg_reward         = (logger.total_reward / logger.num_steps) if logger.num_steps > 0 else 0.0
        logger.max_reward         = max(x.max_reward for x in valid)
        logger.min_reward         = min(x.min_reward for x in valid)

        # —— 关键：union 聚合 info，避免某个 worker 少汇报时整项被“交集”抹掉 ——
        all_keys = set().union(*(lg.info_dict.keys() for lg in valid))
        info_out = {}
        for k in all_keys:
            arrs = []
            for lg in valid:
                vals = lg.info_dict.get(k, [])
                if len(vals) > 0:
                    arrs.append(np.asarray(vals))
            info_out[k] = float(np.mean(np.concatenate(arrs))) if len(arrs) > 0 else 0.0
        logger.info_dict = info_out

        return logger
