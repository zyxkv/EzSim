#!/usr/bin/env python3
"""
单臂机器人可微分批量渲染示例
演示DiffBatchRenderer在强化学习训练中的应用
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path

import ezsim
from ezsim.utils.geom import trans_to_T
from ezsim.utils.image_exporter import FrameImageExporter


class VisionBasedPolicy(nn.Module):
    """视觉驱动的策略网络"""
    def __init__(self, action_dim=7, image_size=128):
        super().__init__()
        
        # 卷积特征提取器
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )
        
    def forward(self, images):
        """
        前向传播
        Args:
            images: (batch_size, height, width, 3) 或 (batch_size, 3, height, width)
        Returns:
            actions: (batch_size, action_dim)
        """
        # 确保输入格式为 (batch, channels, height, width)
        if images.dim() == 4 and images.size(-1) == 3:
            images = images.permute(0, 3, 1, 2)
        
        # 归一化到 [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images / 255.0
            
        features = self.conv_layers(images)
        features = features.flatten(1)
        actions = self.fc_layers(features)
        
        return actions


def setup_scene(n_envs=4, use_diff_renderer=True, image_size=128):
    """设置仿真场景"""
    
    # 选择渲染器
    if use_diff_renderer:
        renderer = ezsim.options.renderers.DiffBatchRenderer(
            use_rasterizer=True,
            enable_gradient=True,
            channels_last=True,
        )
    else:
        renderer = ezsim.options.renderers.BatchRenderer(
            use_rasterizer=True,
        )
    
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.02,
            substeps=4,
        ),
        renderer=renderer,
    )

    # 添加实体
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
        surface=ezsim.surfaces.Default(color=(0.8, 0.8, 0.8))
    )
    
    # 机器人
    franka = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    
    # 目标物体 (不同颜色)
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]
    targets = []
    for i in range(n_envs):
        target = scene.add_entity(
            morph=ezsim.morphs.Sphere(
                radius=0.05,
                pos=(0.5 + np.random.uniform(-0.1, 0.1), 
                     0.0 + np.random.uniform(-0.1, 0.1), 
                     0.5 + np.random.uniform(-0.1, 0.1))
            ),
            surface=ezsim.surfaces.Default(color=colors[i % len(colors)]),
            material=ezsim.materials.Rigid(density=100.0)
        )
        targets.append(target)

    # 相机配置
    cam = scene.add_camera(
        res=(image_size, image_size),
        pos=(1.2, -0.8, 1.2),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=False,
    )

    # 光源
    scene.add_light(
        pos=[1.5, -1.5, 2.0],
        dir=[-1.0, 1.0, -1.0],
        directional=True,
        castshadow=True,
        cutoff=60.0,
        intensity=0.8,
    )
    scene.add_light(
        pos=[-1.5, 1.5, 2.0],
        dir=[1.0, -1.0, -1.0],
        directional=True,
        castshadow=False,
        cutoff=60.0,
        intensity=0.4,
    )

    scene.build(n_envs=n_envs)
    
    return scene, franka, targets, cam


def demonstrate_training_loop(scene, policy, optimizer, n_steps=50):
    """演示训练循环"""
    print(f"\n{'='*60}")
    print("演示可微分训练循环")
    print(f"{'='*60}")
    
    criterion = nn.MSELoss()
    franka = None
    for entity in scene.entities:
        if hasattr(entity, 'links') and len(entity.links) > 7:  # 找到机器人
            franka = entity
            break
    
    if franka is None:
        print("未找到机器人实体")
        return
    
    performance_metrics = {
        'render_time': [],
        'forward_time': [],
        'backward_time': [],
        'total_time': []
    }
    
    print("开始训练循环...")
    
    for step in range(n_steps):
        step_start = time.time()
        
        # 1. 物理仿真步进
        scene.step()
        
        # 2. 可微分渲染
        render_start = time.time()
        
        if hasattr(scene.visualizer.renderer, 'render_for_training'):
            # 使用DiffBatchRenderer的优化接口
            render_results = scene.visualizer.renderer.render_for_training(
                rgb=True, 
                depth=False,
                target_size=(128, 128),  # 缩放到网络输入尺寸
                gradient_enabled=True
            )
            rgb_batch = render_results['rgb']
        else:
            # 使用标准接口
            rgb_batch, _, _, _ = scene.render_all_cameras(rgb=True, depth=False)
            rgb_batch = torch.stack([r for r in rgb_batch if r is not None])
            if rgb_batch.dtype == torch.uint8:
                rgb_batch = rgb_batch.float() / 255.0
        
        render_time = time.time() - render_start
        
        # 3. 策略网络前向传播
        forward_start = time.time()
        optimizer.zero_grad()
        
        actions = policy(rgb_batch)  # 形状: (n_envs, action_dim)
        forward_time = time.time() - forward_start
        
        # 4. 损失计算 (模拟任务 - 让机器人朝目标移动)
        # 简单的随机目标动作，实际应用中应该是基于任务的奖励
        target_actions = 0.1 * torch.randn_like(actions)  # 小幅度动作
        loss = criterion(actions, target_actions)
        
        # 5. 反向传播
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        
        # 6. 应用动作到机器人
        with torch.no_grad():
            # 将动作应用到机器人关节
            current_qpos = franka.get_qpos()
            action_scaled = actions * 0.1  # 缩放动作幅度
            new_qpos = current_qpos + action_scaled
            franka.set_qpos(new_qpos)
        
        total_time = time.time() - step_start
        
        # 记录性能
        performance_metrics['render_time'].append(render_time)
        performance_metrics['forward_time'].append(forward_time)
        performance_metrics['backward_time'].append(backward_time)
        performance_metrics['total_time'].append(total_time)
        
        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, "
                  f"Render={render_time*1000:.1f}ms, "
                  f"Forward={forward_time*1000:.1f}ms, "
                  f"Backward={backward_time*1000:.1f}ms, "
                  f"Total={total_time*1000:.1f}ms")
    
    # 性能报告
    print(f"\n性能统计 (平均值, {n_steps} 步):")
    print(f"  渲染时间: {np.mean(performance_metrics['render_time'])*1000:.2f}ms")
    print(f"  前向传播: {np.mean(performance_metrics['forward_time'])*1000:.2f}ms")
    print(f"  反向传播: {np.mean(performance_metrics['backward_time'])*1000:.2f}ms")
    print(f"  总时间: {np.mean(performance_metrics['total_time'])*1000:.2f}ms")
    print(f"  等效FPS: {1.0/np.mean(performance_metrics['total_time']):.1f}")


def compare_renderers(n_envs=4, n_steps=20):
    """比较不同渲染器的性能"""
    print(f"\n{'='*60}")
    print("渲染器性能对比")
    print(f"{'='*60}")
    
    results = {}
    
    for renderer_name, use_diff in [("BatchRenderer", False), ("DiffBatchRenderer", True)]:
        print(f"\n测试 {renderer_name}...")
        
        # 重新初始化以避免冲突
        scene, franka, targets, cam = setup_scene(n_envs=n_envs, use_diff_renderer=use_diff, image_size=128)
        
        # 预热
        for _ in range(5):
            scene.step()
            if use_diff and hasattr(scene.visualizer.renderer, 'render_for_training'):
                _ = scene.visualizer.renderer.render_for_training(rgb=True)
            else:
                _ = scene.render_all_cameras(rgb=True)
        
        torch.cuda.synchronize()
        
        # 计时测试
        start_time = time.time()
        memory_start = torch.cuda.memory_allocated()
        
        for _ in range(n_steps):
            scene.step()
            if use_diff and hasattr(scene.visualizer.renderer, 'render_for_training'):
                render_results = scene.visualizer.renderer.render_for_training(rgb=True, gradient_enabled=True)
                rgb_batch = render_results['rgb']
                # 模拟梯度计算
                if rgb_batch.requires_grad:
                    loss = rgb_batch.mean()
                    loss.backward()
            else:
                rgb_batch, _, _, _ = scene.render_all_cameras(rgb=True)
        
        torch.cuda.synchronize()
        
        end_time = time.time()
        memory_end = torch.cuda.memory_allocated()
        
        total_time = end_time - start_time
        memory_used = (memory_end - memory_start) / 1024 / 1024
        
        results[renderer_name] = {
            'total_time': total_time,
            'avg_step_time': total_time / n_steps,
            'fps': n_steps / total_time,
            'memory_mb': memory_used
        }
        
        print(f"  总时间: {total_time:.3f}s")
        print(f"  平均步时: {total_time/n_steps*1000:.2f}ms")
        print(f"  FPS: {n_steps/total_time:.1f}")
        print(f"  内存使用: {memory_used:.1f}MB")
        
        # 清理
        if hasattr(scene.visualizer.renderer, 'destroy'):
            scene.visualizer.renderer.destroy()
    
    # 性能对比
    if len(results) == 2:
        diff_result = results["DiffBatchRenderer"]
        batch_result = results["BatchRenderer"]
        
        speedup = batch_result['avg_step_time'] / diff_result['avg_step_time']
        memory_diff = diff_result['memory_mb'] - batch_result['memory_mb']
        
        print(f"\n性能对比:")
        print(f"  速度提升: {speedup:.2f}x")
        print(f"  内存差异: {memory_diff:+.1f}MB")


def export_sample_renders(scene, output_dir="img_output/diff_batch_test", n_frames=5):
    """导出样本渲染结果"""
    print(f"\n导出样本渲染到 {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exporter = FrameImageExporter(output_dir)
    
    for i in range(n_frames):
        scene.step()
        
        if hasattr(scene.visualizer.renderer, 'render_for_training'):
            # 使用DiffBatchRenderer
            results = scene.visualizer.renderer.render_for_training(
                rgb=True, depth=True, normal=True, segmentation=True,
                gradient_enabled=False  # 导出时不需要梯度
            )
            
            # 转换格式以兼容现有导出器
            rgb_list = [results['rgb'][j] for j in range(results['rgb'].shape[0])]
            depth_list = [results['depth'][j] for j in range(results['depth'].shape[0])] if results['depth'] is not None else None
            normal_list = [results['normal'][j] for j in range(results['normal'].shape[0])] if results['normal'] is not None else None
            seg_list = [results['segmentation'][j] for j in range(results['segmentation'].shape[0])] if results['segmentation'] is not None else None
            
            exporter.export_frame_all_cameras(i, rgb=tuple(rgb_list), depth=tuple(depth_list) if depth_list else None,
                                             normal=tuple(normal_list) if normal_list else None, 
                                             segmentation=tuple(seg_list) if seg_list else None)
        else:
            # 使用标准BatchRenderer
            rgba, depth, normal, seg = scene.render_all_cameras(rgb=True, depth=True, normal=True, segmentation=True)
            exporter.export_frame_all_cameras(i, rgb=rgba, depth=depth, normal=normal, segmentation=seg)
    
    print(f"✓ 已导出 {n_frames} 帧到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiffBatchRenderer可微分渲染演示")
    parser.add_argument("-e", "--n_envs", type=int, default=4, help="环境数量")
    parser.add_argument("-s", "--n_steps", type=int, default=50, help="训练步数")
    parser.add_argument("-c", "--compare", action="store_true", help="比较渲染器性能")
    parser.add_argument("-t", "--training", action="store_true", help="演示训练循环")
    parser.add_argument("-o", "--output_dir", type=str, default="img_output/diff_batch_demo", help="输出目录")
    parser.add_argument("--image_size", type=int, default=128, help="渲染图像尺寸")
    args = parser.parse_args()

    print("DiffBatchRenderer 可微分渲染演示")
    print("="*60)

    # 初始化EzSim
    ezsim.init(backend=ezsim.cuda, seed=42, precision="32")

    if args.compare:
        compare_renderers(n_envs=args.n_envs, n_steps=20)

    if args.training:
        # 设置场景
        scene, franka, targets, cam = setup_scene(
            n_envs=args.n_envs, 
            use_diff_renderer=True, 
            image_size=args.image_size
        )
        
        # 创建策略网络和优化器
        policy = VisionBasedPolicy(action_dim=7, image_size=args.image_size).cuda()
        optimizer = optim.Adam(policy.parameters(), lr=0.001)
        
        # 运行训练演示
        demonstrate_training_loop(scene, policy, optimizer, n_steps=args.n_steps)
        
        # 导出样本渲染
        export_sample_renders(scene, args.output_dir, n_frames=5)

    if not args.compare and not args.training:
        # 默认演示
        print("运行默认演示 (使用 --training 或 --compare 查看更多功能)")
        
        scene, franka, targets, cam = setup_scene(
            n_envs=args.n_envs, 
            use_diff_renderer=True, 
            image_size=args.image_size
        )
        
        # 简单渲染测试
        print("执行基本渲染测试...")
        for i in range(10):
            scene.step()
            results = scene.visualizer.renderer.render_for_training(rgb=True, depth=True)
            if i == 0:
                print(f"RGB形状: {results['rgb'].shape}")
                print(f"Depth形状: {results['depth'].shape if results['depth'] is not None else 'None'}")
        
        export_sample_renders(scene, args.output_dir, n_frames=3)
        print("✓ 基本演示完成")

    print(f"\n{'='*60}")
    print("演示完成!")


if __name__ == "__main__":
    main()
