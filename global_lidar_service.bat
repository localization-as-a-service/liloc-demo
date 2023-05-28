@echo off

@REM start "GPC Service" cmd /c "conda activate lidar & python global_lidar_server.py"
start "Secondary Serive" cmd /c "conda activate lidar & python secondary_device.py"
start "Localization Serive" cmd /c "conda activate lidar & python localization_server.py"