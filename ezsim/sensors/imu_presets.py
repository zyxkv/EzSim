#!/usr/bin/env python3
"""
IMU Configuration Presets

This module provides pre-configured IMU settings that match real-world sensors.
These configurations can be used directly or as starting points for custom configurations.
"""

from ezsim.sensors.deprecated_imu import IMUNoiseConfig


class IMUPresets:
    """Predefined IMU configurations based on real sensors"""
    
    @staticmethod
    def consumer_grade() -> IMUNoiseConfig:
        """
        Consumer-grade IMU (like those in smartphones)
        
        Typical specifications:
        - Lower precision, higher noise
        - Suitable for general orientation tracking
        """
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.1,      # m/s²/√Hz
            accel_bias_stability=0.01,    # m/s²
            accel_random_walk=0.001,      # m/s²/√s
            gyro_noise_density=0.01,      # rad/s/√Hz
            gyro_bias_stability=0.001,    # rad/s
            gyro_random_walk=0.0001,      # rad/s/√s
            seed=42
        )
    
    @staticmethod
    def tactical_grade() -> IMUNoiseConfig:
        """
        Tactical-grade IMU (automotive, UAV applications)
        
        Typical specifications:
        - Medium precision and cost
        - Good for navigation and control systems
        """
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.01,     # m/s²/√Hz
            accel_bias_stability=0.001,   # m/s²
            accel_random_walk=0.0001,     # m/s²/√s
            gyro_noise_density=0.001,     # rad/s/√Hz
            gyro_bias_stability=0.0001,   # rad/s
            gyro_random_walk=0.00001,     # rad/s/√s
            seed=42
        )
    
    @staticmethod
    def navigation_grade() -> IMUNoiseConfig:
        """
        Navigation-grade IMU (aerospace, precision applications)
        
        Typical specifications:
        - High precision, low noise
        - Suitable for inertial navigation systems
        """
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.001,    # m/s²/√Hz
            accel_bias_stability=0.0001,  # m/s²
            accel_random_walk=0.00001,    # m/s²/√s
            gyro_noise_density=0.0001,    # rad/s/√Hz
            gyro_bias_stability=0.00001,  # rad/s
            gyro_random_walk=0.000001,    # rad/s/√s
            seed=42
        )
    
    @staticmethod
    def perfect() -> IMUNoiseConfig:
        """
        Perfect IMU (no noise)
        
        Useful for:
        - Algorithm development
        - Ground truth generation
        - Debugging
        """
        return IMUNoiseConfig(
            enable_noise=False,
            seed=42
        )
    
    @staticmethod
    def custom(accel_noise=0.01, gyro_noise=0.001, enable_noise=True) -> IMUNoiseConfig:
        """
        Custom IMU configuration
        
        Parameters
        ----------
        accel_noise : float
            Accelerometer noise density (m/s²/√Hz)
        gyro_noise : float
            Gyroscope noise density (rad/s/√Hz)
        enable_noise : bool
            Whether to enable noise simulation
        """
        return IMUNoiseConfig(
            enable_noise=enable_noise,
            accel_noise_density=accel_noise,
            accel_bias_stability=accel_noise * 0.1,
            accel_random_walk=accel_noise * 0.01,
            gyro_noise_density=gyro_noise,
            gyro_bias_stability=gyro_noise * 0.1,
            gyro_random_walk=gyro_noise * 0.01,
            seed=42
        )


# Real sensor specifications (for reference)
REAL_SENSOR_SPECS = {
    "ADXL345": {  # Consumer accelerometer
        "type": "accelerometer",
        "noise_density": 0.15,  # mg/√Hz → ~0.0015 m/s²/√Hz
        "bias_stability": 1.0,  # mg → ~0.01 m/s²
        "description": "Popular consumer-grade 3-axis accelerometer"
    },
    
    "MPU6050": {  # Consumer IMU
        "type": "imu",
        "accel_noise": 0.004,   # g/√Hz → ~0.04 m/s²/√Hz
        "gyro_noise": 0.003,    # dps/√Hz → ~5.2e-5 rad/s/√Hz
        "description": "Common consumer IMU in smartphones and drones"
    },
    
    "BMI088": {  # Automotive grade
        "type": "imu", 
        "accel_noise": 0.15,    # mg/√Hz → ~0.0015 m/s²/√Hz
        "gyro_noise": 0.014,    # dps/√Hz → ~2.4e-4 rad/s/√Hz
        "description": "Automotive-grade IMU for stability control"
    },
    
    "STIM300": {  # Tactical grade
        "type": "imu",
        "accel_noise": 0.05,    # mg/√Hz → ~0.0005 m/s²/√Hz
        "gyro_noise": 0.15,     # deg/hr/√Hz → ~7.3e-7 rad/s/√Hz  
        "description": "High-performance tactical grade IMU"
    },
    
    "Honeywell_HG4930": {  # Navigation grade
        "type": "imu",
        "accel_bias_stability": 0.25,  # mg → ~0.0025 m/s²
        "gyro_bias_stability": 1.0,    # deg/hr → ~4.8e-6 rad/s
        "description": "Military/aerospace navigation grade IMU"
    }
}


def get_sensor_config(sensor_name: str) -> IMUNoiseConfig:
    """
    Get IMU configuration based on real sensor specifications
    
    Parameters
    ----------
    sensor_name : str
        Name of the sensor from REAL_SENSOR_SPECS
        
    Returns
    -------
    IMUNoiseConfig
        Configuration matching the specified sensor
    """
    if sensor_name not in REAL_SENSOR_SPECS:
        available = list(REAL_SENSOR_SPECS.keys())
        raise ValueError(f"Unknown sensor '{sensor_name}'. Available: {available}")
    
    spec = REAL_SENSOR_SPECS[sensor_name]
    
    # Convert specifications to our noise model
    if sensor_name == "ADXL345":
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.0015,
            accel_bias_stability=0.01,
            accel_random_walk=0.0001,
            gyro_noise_density=0.001,  # Default since only accelerometer
            gyro_bias_stability=0.0001,
            gyro_random_walk=0.00001,
            seed=42
        )
    
    elif sensor_name == "MPU6050":
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.04,
            accel_bias_stability=0.004,
            accel_random_walk=0.0004,
            gyro_noise_density=5.2e-5,
            gyro_bias_stability=5.2e-6,
            gyro_random_walk=5.2e-7,
            seed=42
        )
    
    elif sensor_name == "BMI088":
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.0015,
            accel_bias_stability=0.00015,
            accel_random_walk=0.000015,
            gyro_noise_density=2.4e-4,
            gyro_bias_stability=2.4e-5,
            gyro_random_walk=2.4e-6,
            seed=42
        )
    
    elif sensor_name == "STIM300":
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.0005,
            accel_bias_stability=0.00005,
            accel_random_walk=0.000005,
            gyro_noise_density=7.3e-7,
            gyro_bias_stability=7.3e-8,
            gyro_random_walk=7.3e-9,
            seed=42
        )
    
    elif sensor_name == "Honeywell_HG4930":
        return IMUNoiseConfig(
            enable_noise=True,
            accel_noise_density=0.0001,
            accel_bias_stability=0.0025,
            accel_random_walk=0.00001,
            gyro_noise_density=1e-7,
            gyro_bias_stability=4.8e-6,
            gyro_random_walk=1e-8,
            seed=42
        )
    
    else:
        # Default to tactical grade
        return IMUPresets.tactical_grade()


# Example usage
if __name__ == "__main__":
    print("IMU Configuration Examples")
    print("=" * 50)
    
    configs = {
        "Perfect": IMUPresets.perfect(),
        "Consumer": IMUPresets.consumer_grade(),
        "Tactical": IMUPresets.tactical_grade(),
        "Navigation": IMUPresets.navigation_grade(),
        "MPU6050": get_sensor_config("MPU6050"),
        "STIM300": get_sensor_config("STIM300")
    }
    
    for name, config in configs.items():
        print(f"\n{name} Grade IMU:")
        print(f"  Noise enabled: {config.enable_noise}")
        if config.enable_noise:
            print(f"  Accel noise density: {config.accel_noise_density:.6f} m/s²/√Hz")
            print(f"  Gyro noise density:  {config.gyro_noise_density:.6f} rad/s/√Hz")
            print(f"  Accel bias stability: {config.accel_bias_stability:.6f} m/s²")
            print(f"  Gyro bias stability:  {config.gyro_bias_stability:.6f} rad/s")
    
    print(f"\nAvailable real sensor configs: {list(REAL_SENSOR_SPECS.keys())}")
