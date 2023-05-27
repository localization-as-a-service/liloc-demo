@echo off

start "IMU" cmd /c "conda activate lidar & python lidar_imu_capture.py --mode imu"
start "Depth" cmd /c "conda activate lidar & python lidar_imu_capture.py --mode cam"