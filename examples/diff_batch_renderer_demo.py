#!/usr/bin/env python3
"""
DiffBatchRenderer 端到端示例
演示可微分渲染在强化学习中的应用

这个示例展示了一个简单的视觉导向的机器人控制任务：
- 机器人通过摄像头观察环境
- 神经网络策略基于RGB图像输出动作
- 使用可微分渲染训练策略网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
import ezsim
import numpy as np
import time
from typing import Dict, Tuple


class VisionBasedPolicy(nn.Module):
    """基于视觉的策略网络"""
    
    def __init__(self, image_size: Tuple[int, int] = (128, 128), action_dim: int = 6):
        super().__init__()
        self.image_size = image_size
        self.action_dim = action_dim
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )
        
    def forward(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            rgb_obs: RGB观察 [batch_size, cameras, height, width, 3]
            
        Returns:
            actions: 动作向量 [batch_size, action_dim]
        """
        # 处理批量多相机输入
        batch_size = rgb_obs.shape[0]
        
        # 如果有多个相机，选择第一个
        if rgb_obs.dim() == 5:  # [batch, cameras, h, w, c]
            rgb_obs = rgb_obs[:, 0]  # [batch, h, w, c]
        
        # 转换为 [batch, channels, height, width]
        if rgb_obs.shape[-1] == 3:  # channels last
            rgb_obs = rgb_obs.permute(0, 3, 1, 2)
        
        # CNN特征提取
        features = self.cnn(rgb_obs)
        features = features.reshape(batch_size, -1)  # 使用reshape代替view
        
        # 输出动作
        actions = self.fc(features)
        return actions


class DifferentiableRobotTrainer:
    """可微分机器人训练器"""
    
    def __init__(self, scene_builder_func, policy_network: nn.Module):
        self.scene_builder_func = scene_builder_func
        self.policy = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)
        
        # 训练统计
        self.training_history = {
            'losses': [],
            'rewards': [],
            'episode_lengths': []
        }
        
    def setup_environment(self, n_envs: int = 4):
        """设置环境"""
        print(f"设置 {n_envs} 个并行环境...")
        
        # 初始化EzSim
        ezsim.init(backend=ezsim.cuda, seed=42)
        
        # 创建场景with DiffBatchRenderer
        self.scene = ezsim.Scene(
            renderer=ezsim.options.renderers.DiffBatchRenderer(
                enable_gradient=True,
                channels_last=True,
                pin_memory=False
            ),
            show_viewer=False,
        )
        
        # 使用builder函数构建场景
        self.scene_builder_func(self.scene)
        
        # 构建场景
        self.scene.build(n_envs=n_envs)
        self.n_envs = n_envs
        
        print(f"✓ 环境设置完成，渲染器类型: {type(self.scene.visualizer.renderer)}")
        
    def get_observations(self) -> Dict[str, torch.Tensor]:
        """获取当前观察"""
        # 执行仿真步骤
        self.scene.step()
        
        # 渲染观察
        render_results = self.scene.visualizer.renderer.render_differentiable(
            rgb=True, 
            depth=False, 
            normal=False, 
            segmentation=False,
            return_dict=True,
            gradient_enabled=True
        )
        
        return render_results
    
    def compute_reward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """计算奖励（示例实现）"""
        rgb = observations['rgb']
        
        # 简单的奖励函数：鼓励图像亮度适中
        target_brightness = 0.4
        current_brightness = rgb.mean(dim=[1, 2, 3, 4])  # [batch_size]
        brightness_reward = -torch.abs(current_brightness - target_brightness)
        
        # 动作平滑奖励：惩罚大的动作
        action_penalty = -0.1 * torch.norm(actions, dim=1)
        
        total_reward = brightness_reward + action_penalty
        return total_reward
    
    def apply_actions(self, actions: torch.Tensor):
        """应用动作到环境"""
        # 这里应该根据具体机器人实现动作应用
        # 示例：假设前3维是位置，后3维是旋转
        actions_np = actions.detach().cpu().numpy()
        
        # 应用到场景中的机器人（这里需要根据具体实现调整）
        for env_idx in range(self.n_envs):
            if hasattr(self.scene, 'robots') and len(self.scene.robots) > env_idx:
                robot = self.scene.robots[env_idx]
                # 示例动作应用
                pos_delta = actions_np[env_idx, :3] * 0.01  # 缩放位置变化
                rot_delta = actions_np[env_idx, 3:] * 0.1   # 缩放旋转变化
                
                # 应用动作（需要根据具体机器人API调整）
                current_pos = robot.get_pos() if hasattr(robot, 'get_pos') else np.zeros(3)
                new_pos = current_pos + pos_delta
                robot.set_pos(new_pos) if hasattr(robot, 'set_pos') else None
    
    def train_episode(self, max_steps: int = 100) -> Dict[str, float]:
        """训练一个episode"""
        episode_losses = []
        episode_rewards = []
        
        for step in range(max_steps):
            # 获取观察
            observations = self.get_observations()
            
            # 策略网络推理
            rgb_obs = observations['rgb']
            actions = self.policy(rgb_obs)
            
            # 计算奖励
            rewards = self.compute_reward(observations, actions)
            
            # 计算损失（最大化奖励）
            loss = -rewards.mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 应用动作
            self.apply_actions(actions)
            
            # 记录统计
            episode_losses.append(loss.item())
            episode_rewards.append(rewards.mean().item())
            
            if step % 20 == 0:
                print(f"  Step {step:3d}: Loss={loss.item():.4f}, Reward={rewards.mean().item():.4f}")
        
        return {
            'mean_loss': np.mean(episode_losses),
            'mean_reward': np.mean(episode_rewards),
            'episode_length': max_steps
        }
    
    def train(self, n_episodes: int = 10):
        """完整训练流程"""
        print(f"开始训练 {n_episodes} 个episodes...")
        
        for episode in range(n_episodes):
            print(f"\n=== Episode {episode + 1}/{n_episodes} ===")
            start_time = time.time()
            
            # 训练一个episode
            episode_stats = self.train_episode()
            
            # 记录统计
            self.training_history['losses'].append(episode_stats['mean_loss'])
            self.training_history['rewards'].append(episode_stats['mean_reward'])
            self.training_history['episode_lengths'].append(episode_stats['episode_length'])
            
            episode_time = time.time() - start_time
            print(f"Episode完成: Loss={episode_stats['mean_loss']:.4f}, "
                  f"Reward={episode_stats['mean_reward']:.4f}, Time={episode_time:.2f}s")
        
        self.print_training_summary()
    
    def print_training_summary(self):
        """打印训练总结"""
        print(f"\n{'='*60}")
        print("训练总结")
        print(f"{'='*60}")
        
        losses = self.training_history['losses']
        rewards = self.training_history['rewards']
        
        print(f"总Episodes: {len(losses)}")
        print(f"平均Loss: {np.mean(losses):.4f} (±{np.std(losses):.4f})")
        print(f"平均Reward: {np.mean(rewards):.4f} (±{np.std(rewards):.4f})")
        print(f"最终Loss: {losses[-1]:.4f}")
        print(f"最终Reward: {rewards[-1]:.4f}")
        
        if len(losses) >= 2:
            loss_improvement = losses[0] - losses[-1]
            reward_improvement = rewards[-1] - rewards[0]
            print(f"Loss改善: {loss_improvement:+.4f}")
            print(f"Reward改善: {reward_improvement:+.4f}")


def build_simple_scene(scene):
    """构建简单的机器人场景"""
    # 添加地面
    scene.add_entity(ezsim.morphs.Plane())
    
    # 添加一些物体作为视觉目标
    scene.add_entity(
        morph=ezsim.morphs.Sphere(radius=0.3, pos=(1, 0, 0.5)),
        surface=ezsim.surfaces.Default(color=(1.0, 0.2, 0.2))
    )
    scene.add_entity(
        morph=ezsim.morphs.Box(size=(0.2, 0.2, 0.4), pos=(-1, 0.5, 0.2)),
        surface=ezsim.surfaces.Default(color=(0.2, 1.0, 0.2))
    )
    scene.add_entity(
        morph=ezsim.morphs.Cylinder(radius=0.15, height=0.3, pos=(0, -1, 0.15)),
        surface=ezsim.surfaces.Default(color=(0.2, 0.2, 1.0))
    )
    
    # 添加相机
    scene.add_camera(
        res=(128, 128),
        pos=(2, 1, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=False,
    )
    
    # 添加光源
    scene.add_light(pos=[3, 2, 3], dir=[-1, -0.5, -1], directional=1, 
                   castshadow=1, cutoff=45.0, intensity=1.5)
    scene.add_light(pos=[-2, -2, 2], dir=[1, 1, -1], directional=1, 
                   castshadow=0, cutoff=30.0, intensity=0.8)


def main():
    """主函数"""
    print("DiffBatchRenderer 端到端示例")
    print("="*60)
    
    try:
        # 创建策略网络
        policy = VisionBasedPolicy(image_size=(128, 128), action_dim=6)
        policy = policy.cuda()  # 移动到GPU
        
        print(f"策略网络参数数量: {sum(p.numel() for p in policy.parameters()):,}")
        
        # 创建训练器
        trainer = DifferentiableRobotTrainer(build_simple_scene, policy)
        
        # 设置环境
        trainer.setup_environment(n_envs=2)  # 使用2个并行环境
        
        # 开始训练
        trainer.train(n_episodes=5)
        
        print(f"\n✓ 训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
