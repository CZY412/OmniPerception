'''
Author: Raymon Yip
Description: ElSpider with RayCast backend + LidarSensor (with noise)
'''

import torch
from isaacgym.torch_utils import quat_apply

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_depthcam import LeggedRobotDepth
from legged_gym.utils import (
    GaitScheduler, GaitSchedulerCfg,
    AsyncGaitScheduler, AsyncGaitSchedulerCfg
)
from legged_gym.utils.helpers import class_to_dict

# ✅ 教程里的 LidarSensor
from sensors.lidar_sensor import LidarSensor


class ElSpiderRayCast(LeggedRobotDepth):
    """
    ElSpider with mesh-based world + LidarSensor (noise handled inside sensor)
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # ================= Gait Scheduler =================
        cfg_gait = GaitSchedulerCfg()
        cfg_gait.dt = self.dt
        cfg_gait.period = 1.4
        cfg_gait.swing_height = 0.07

        self.gait_scheduler = GaitScheduler(
            self.height_samples,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.foot_positions,
            self.foot_velocities,
            self.num_envs,
            self.device,
            cfg_gait
        )

        cfg_async = AsyncGaitSchedulerCfg()
        self.async_gait_scheduler = AsyncGaitScheduler(
            self.height_samples,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.foot_positions,
            self.foot_velocities,
            self.num_envs,
            self.device,
            cfg_async
        )

        # ================= ✅ LIDAR SENSOR =================
        # 关键点：
        # 1. 不在这里加噪声
        # 2. 不自己算 ray
        # 3. 噪声 / dropout / 扫描模型全部在 lidar_sensor.py
        self.lidar = LidarSensor(
            meshes=self.ray_caster.meshes,   # RayCast / Warp mesh
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.lidar                # 教程里的 lidar cfg（含噪声）
        )

    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if hasattr(self, 'ray_caster'):
            self.ray_caster.reset(env_ids)

        # ✅ 重要：reset lidar 内部状态（噪声随机性 / dropout）
        self.lidar.reset(env_ids)

    # ------------------------------------------------------------------

    def post_physics_step(self):
        super().post_physics_step()

        self.gait_scheduler.step(
            self.foot_positions,
            self.foot_velocities,
            self.commands
        )

        # ✅ 每一步更新雷达
        # LidarSensor 内部：
        # - 生成扫描方向
        # - ray-mesh intersection
        # - 加噪声 / dropout
        self.lidar.update(
            base_pos=self.root_states[:, 0:3],
            base_quat=self.base_quat
        )

    # ------------------------------------------------------------------

    def _get_observations(self):
        obs = super()._get_observations()

        # ✅ 直接用雷达输出（已经含噪声）
        lidar_obs = self.lidar.get_observation()

        return torch.cat([obs, lidar_obs], dim=-1)

    # ------------------------------------------------------------------

    def _reward_gait_scheduler(self):
        return self.gait_scheduler.reward_foot_z_track()

    def _reward_async_gait_scheduler(self):
        scales = class_to_dict(self.cfg.rewards.async_gait_scheduler)

        def w(k):
            return scales[k]

        return (
            self.async_gait_scheduler.reward_dof_align() * w('dof_align')
            + self.async_gait_scheduler.reward_dof_nominal_pos() * w('dof_nominal_pos')
            + self.async_gait_scheduler.reward_foot_z_align() * w('reward_foot_z_align')
        )

    # ------------------------------------------------------------------

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= (self.projected_gravity[:, 2] > 0)
