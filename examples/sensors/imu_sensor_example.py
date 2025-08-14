#!/usr/bin/env python3
"""
IMU Sensor Usage Example

This script demonstrates how to use the IMU sensor in various scenarios
and compares the results with ground truth simulation data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import ezsim
from ezsim.sensors.deprecated_imu import IMUSensor, IMUNoiseConfig
from ezsim import Scene


def example_static_imu():
    """Example: IMU readings from a static object"""
    print("=" * 60)
    print("Example 1: Static IMU Test")
    print("=" * 60)
    
    # Create scene with a static box
    scene = Scene()
    entity = scene.add_entity(
        morph=ezsim.morphs.Box(
            pos=[0, 0, 1], 
            size=[0.2, 0.2, 0.2],
            fixed=True  # Static object
        )
    )
    scene.build()
    
    # Create IMU without noise
    imu_clean = IMUSensor(entity, noise_config=IMUNoiseConfig(enable_noise=False))
    
    # Create IMU with noise
    imu_noisy = IMUSensor(entity, noise_config=IMUNoiseConfig(
        enable_noise=True,
        accel_noise_density=0.01,
        gyro_noise_density=0.001,
        seed=42
    ))
    
    # Collect data
    clean_data = []
    noisy_data = []
    
    for i in range(100):
        scene.step()
        clean_data.append(imu_clean.read())
        noisy_data.append(imu_noisy.read())
    
    # Print results
    print(f"Clean IMU - Mean Lin Acc: {torch.stack([d.lin_acc_body for d in clean_data]).mean(dim=0)}")
    print(f"Clean IMU - Std Lin Acc:  {torch.stack([d.lin_acc_body for d in clean_data]).std(dim=0)}")
    print(f"Noisy IMU - Mean Lin Acc: {torch.stack([d.lin_acc_body for d in noisy_data]).mean(dim=0)}")
    print(f"Noisy IMU - Std Lin Acc:  {torch.stack([d.lin_acc_body for d in noisy_data]).std(dim=0)}")
    
    print("\nExpected: Clean IMU should have near-zero acceleration (gravity compensated)")
    print("Expected: Noisy IMU should have non-zero standard deviation due to noise\n")


def example_falling_object():
    """Example: IMU readings from a falling object"""
    print("=" * 60)
    print("Example 2: Falling Object IMU Test")
    print("=" * 60)
    
    # Create scene with a falling box
    scene = Scene()
    entity = scene.add_entity(
        morph=ezsim.morphs.Box(
            pos=[0, 0, 5], 
            size=[0.2, 0.2, 0.2],
            fixed=False  # Free falling object
        )
    )
    scene.build()
    
    # Create IMU without noise
    imu = IMUSensor(entity, noise_config=IMUNoiseConfig(enable_noise=False))
    
    # Collect data during fall
    time_data = []
    position_data = []
    velocity_data = []
    imu_data = []
    
    for i in range(200):
        scene.step()
        
        # Get ground truth
        pos = entity.get_links_pos()
        vel = entity.get_links_vel()
        
        # Get IMU data
        imu_reading = imu.read()
        
        time_data.append(i * scene._substep_dt)
        position_data.append(pos.clone())
        velocity_data.append(vel.clone())
        imu_data.append(imu_reading)
    
    # Convert to numpy for plotting
    time_np = np.array(time_data)
    positions = torch.stack(position_data).squeeze().numpy()
    velocities = torch.stack(velocity_data).squeeze().numpy()
    accelerations = torch.stack([d.lin_acc_body for d in imu_data]).squeeze().numpy()
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Position
    axes[0].plot(time_np, positions[:, 2], 'b-', linewidth=2)
    axes[0].set_title('Falling Object - Z Position')
    axes[0].set_ylabel('Position (m)')
    axes[0].grid(True)
    
    # Velocity
    axes[1].plot(time_np, velocities[:, 2], 'g-', linewidth=2)
    axes[1].set_title('Falling Object - Z Velocity')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].grid(True)
    
    # IMU Acceleration
    axes[2].plot(time_np, accelerations[:, 2], 'r-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_title('IMU Linear Acceleration Z (Body Frame)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('imu_falling_object.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mean IMU acceleration during free fall: {np.mean(accelerations, axis=0)}")
    print("Expected: Should be close to zero (gravity compensated)")
    print("Plot saved as 'imu_falling_object.png'\n")


def example_rotating_object():
    """Example: IMU readings from a rotating object"""
    print("=" * 60)
    print("Example 3: Rotating Object IMU Test")
    print("=" * 60)
    
    # This example would require the ability to apply torques
    # For now, we'll demonstrate the coordinate transformation
    
    scene = Scene()
    entity = scene.add_entity(
        morph=ezsim.morphs.Box(
            pos=[0, 0, 1], 
            size=[0.2, 0.2, 0.2],
            fixed=False
        )
    )
    scene.build()
    
    imu = IMUSensor(entity, noise_config=IMUNoiseConfig(enable_noise=False))
    
    # Apply initial angular velocity (if method exists)
    # entity.set_links_ang_vel(torch.tensor([[0, 0, 1.0]]))  # 1 rad/s around Z
    
    print("This example requires angular velocity setting capability")
    print("Current implementation shows coordinate frame transformation")
    
    # Collect a few data points
    for i in range(10):
        scene.step()
        imu_reading = imu.read()
        print(f"Step {i}: Angular velocity = {imu_reading.ang_vel_body}")


def example_noise_analysis():
    """Example: Analyze IMU noise characteristics"""
    print("=" * 60)
    print("Example 4: IMU Noise Analysis")
    print("=" * 60)
    
    scene = Scene()
    entity = scene.add_entity(
        morph=ezsim.morphs.Box(
            pos=[0, 0, 1], 
            size=[0.2, 0.2, 0.2],
            fixed=True
        )
    )
    scene.build()
    
    # Create IMU with different noise levels
    noise_configs = {
        'low_noise': IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.001,
            gyro_noise_density=0.0001,
            seed=42
        ),
        'medium_noise': IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.01,
            gyro_noise_density=0.001,
            seed=42
        ),
        'high_noise': IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.1,
            gyro_noise_density=0.01,
            seed=42
        )
    }
    
    imus = {name: IMUSensor(entity, noise_config=config) 
            for name, config in noise_configs.items()}
    
    # Collect noise data
    data_points = 1000
    noise_data = {name: {'accel': [], 'gyro': []} for name in imus.keys()}
    
    for i in range(data_points):
        scene.step()
        for name, imu in imus.items():
            reading = imu.read()
            noise_data[name]['accel'].append(reading.lin_acc_body.clone())
            noise_data[name]['gyro'].append(reading.ang_vel_body.clone())
    
    # Analyze noise statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (name, data) in enumerate(noise_data.items()):
        accel_data = torch.stack(data['accel']).numpy()
        gyro_data = torch.stack(data['gyro']).numpy()
        
        # Accelerometer noise
        axes[0, i].hist(accel_data[:, 0], bins=50, alpha=0.7)
        axes[0, i].set_title(f'Accelerometer Noise - {name}')
        axes[0, i].set_xlabel('Acceleration (m/s²)')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True)
        
        # Gyroscope noise
        axes[1, i].hist(gyro_data[:, 0], bins=50, alpha=0.7)
        axes[1, i].set_title(f'Gyroscope Noise - {name}')
        axes[1, i].set_xlabel('Angular Velocity (rad/s)')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True)
        
        # Print statistics
        print(f"{name}:")
        print(f"  Accel std: {np.std(accel_data, axis=0)}")
        print(f"  Gyro std:  {np.std(gyro_data, axis=0)}")
    
    plt.tight_layout()
    plt.savefig('imu_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'imu_noise_analysis.png'\n")


def example_multi_environment():
    """Example: IMU in multi-environment setup"""
    print("=" * 60)
    print("Example 5: Multi-Environment IMU Test")
    print("=" * 60)
    
    # This example would demonstrate using IMU across multiple environments
    # if the simulation supports it
    
    scene = Scene()
    entity = scene.add_entity(
        morph=ezsim.morphs.Box(
            pos=[0, 0, 1], 
            size=[0.2, 0.2, 0.2],
            fixed=False
        )
    )
    scene.build()
    
    imu = IMUSensor(entity, noise_config=IMUNoiseConfig(enable_noise=False))
    
    # Test environment indexing
    for i in range(5):
        scene.step()
        
        # Read all environments
        all_data = imu.read()
        print(f"Step {i}: IMU data shape = {all_data.lin_acc_body.shape}")
        
        # Read specific environments (if applicable)
        # subset_data = imu.read(envs_idx=[0])
        # print(f"  Subset data shape = {subset_data.lin_acc_body.shape}")


def main():
    """Run all IMU examples"""
    print("IMU Sensor Examples and Validation")
    print("==================================\n")
    
    # Run examples
    try:
        example_static_imu()
        example_falling_object()
        example_rotating_object()
        example_noise_analysis()
        example_multi_environment()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
