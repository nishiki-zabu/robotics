Deep Q Learning
=========================

Deep Q Learning

入力：深度画像（640×480）  
出力・報酬：6（前進・100/弱右折・50/強右折・50/弱左折・50/強左折・50/後退・0)  
ペナルティ：0.0008*math.exp((1-depth)(1-depth))　全ピクセルの合計値  
*depth:depth_image[j,i]の深度値

to run Deep Q Learning
- `roslaunch cube_gazebo kit_office.launch`
- `roslaunch cube_control kit_control_gazebo.launch`
- `python train_random_walk.py`

Subscribe: '/camera/depth/image_raw'  
Publish: '/base/diff_drive_controller/cmd_vel_raw'

QR Tracking System

to run QR Tracking System
- `roslaunch cube_gazebo cube_office.launch`
- `roslaunch cube_control base_control.launch` 
- `python qr_tracking.py`

Subscribe: '/camera/rgb/image_raw' '/camera/depth/image_raw'  
Publish: '/base/diff_drive_controller/cmd_vel'
