# LUMIR: Robot hỗ trợ người khiếm thị

## Robot (Raspberry PI)
### Terminal 1
```
source install/setup.bash
ros2 run robot_odom_cmd_vel bridge_odom
```

### Terminal 2
```
source install/setup.bash
ros2 launch urg_lidar urg_lidar.launch.py 
```

## Laptop

### Terminal 1: Khởi chạy description
```
source install/setup.bash
ros2 launch robot_description display.launch.py
```

### Terminal 2: khởi chạy joystick
```
source install/setup.bash
ros2 launch robot_joy joystick.launch.py 
```

### Terminal 3: Khởi chạy cartographer (Mapping)
```
source install/setup.bash
ros2 launch robot_mapping cartographer.launch.py
```
Lưu map:
```
ros2 run nav2_map_server map_saver_cli -f my_map
```

### Terminal 4: khởi chạy navigation (khi đã có map)
```
source install/setup.bash
ros2 launch robot_navigation navigation.launch.py 
```

### Terminal 5: Khởi chạy supervisor 
```
source install/setup.bash
ros2 launch robot_supervisor supervisor.launch.py
```