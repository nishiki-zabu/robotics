#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('cv_bridge')
import rospy
import sys
import termios
import tty
import cv2
import numpy as np
import threading
import time
import random
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import math
import zbar
import PIL.Image
import os

cmd_vel = '/base/diff_drive_controller/cmd_vel_raw'
Q_val = '/Q_val'

QSIZE = 1
exitFlag = 0
THRESHOLD = 2.2
THRESHOLD2 = 1.4

WINDOW_W = 48
WINDOW_H = 48

WINDOW_CENTER_W = 215
WINDOW_CENTER_H = 215

class RandomWalk:

    def __init__(self):
        print"init_RandomWalk"
        rospy.init_node('vel_publisher')

        self.cmd_pub = rospy.Publisher(cmd_vel, Twist, queue_size=QSIZE)
        self.Q_val = Q_val

        # ROS init
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.callback)

        # １０ loop/sec
        self.r = rospy.Rate(10)

        # Q parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 480
        self.screen_n_cols = 480
        self.enable_actions = (0, 1, 2, 3, 4, 5)
        self.reward = 0.0

    def Q_save(self,data):
        try:
            self.Q_val = data
        except CvBridgeError as e:
            print(e)

    def callback(self,data):
        try:
            """
            # IPL_DEPTH_16U CV_16UC1 チャンネルのバイト数2 チャンネル数（一要素）1 一要素のバイト数2 符号無 整数
            self._cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            """
            self._cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
        except CvBridgeError as e:
            print(e)

    def update(self, action):

        """
        action:
            0: go straight
            1: turn right
            2: turn left
            3: turn right little
            4: turn left little
        """

        self.reward = 0.0

        if action == self.enable_actions[1]:
            # turn left
            self.turnAroundLeft(cmd_vel)
            self.reward += 50
        elif action == self.enable_actions[2]:
            # turn left little
            self.turnAroundLeftLittle(cmd_vel)
            self.reward += 50
        elif action == self.enable_actions[3]:
            # turn right
            self.turnAroundRight(cmd_vel)
            self.reward += 50
        elif action == self.enable_actions[4]:
            # turn right little
            self.turnAroundRightLittle(cmd_vel)
            self.reward += 50 
        elif action == self.enable_actions[5]:
            # go back
            self.goBack(cmd_vel)
            self.reward = 0
        else:
            # go straight
            self.goStraight(cmd_vel)
            self.reward += 100

        # update state
        self.get_state()

        # collision depth
        for j in range(0,479):
            for i in range(0,639):
                depth = self.get_distance(j,i)
                if depth < 1:
                    self.reward -= 0.0008*math.exp((1-depth)*(1-depth))

    def get_state(self):
        # reset state
        self.state = np.zeros((self.screen_n_cols, self.screen_n_rows))
        self.cv_image = cv2.resize(self._cv_image, (480,480))
        # get state
        for j in range(0,479):
            for i in range(0,479):
                # 距離が遠い場合
                if self.cv_image[j][i]==0:
                    self.state[j, i] = 5
                # 距離が近い場合
                elif self.cv_image[j][i] is None:
                    self.state[j, i] = 0
                else:
                    self.state[j, i] = self.cv_image[j][i]

    def get_distance(self, j, i):
        # update distance
        # 距離が遠い場合
        if self._cv_image[j][i]==0:
            average_return = 5
        # 距離が近い場合
        elif self._cv_image[j][i] is None:
            average_return =0.4
        else:
            average_return = self._cv_image[j][i]
        return average_return

    def observe(self):
        self.get_state()
        return self.state, self.reward

    def execute_action(self, action):
        self.update(action)

    def realKey2():
        global exitFlag
        ch = '9' 
        while ch != 'q':
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == 'q':
                    exitFlag = -1
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, old)

    def turnAroundLeft(self, vel):
        vel = Twist()
        for i in range(14):
            vel.angular.z = 3.14/2
            self.cmd_pub.publish(vel)
            self.r.sleep()     
            continue
        print "try to turn left with the dgree of ", vel.angular.z*180/3.14
    def turnAroundLeftLittle(self, vel):
        vel = Twist()
        for i in range(6):
            vel.angular.z = 3.14/2
            self.cmd_pub.publish(vel)
            self.r.sleep()     
            continue
        print "try to turn left little with the dgree of ", vel.angular.z*180/3.14
    def turnAroundRight(self, vel):
        vel = Twist()
        for i in range(14):
            vel.angular.z = -3.14/2
            self.cmd_pub.publish(vel)
            self.r.sleep()
            continue
        print "try to turn right with the degree of ", vel.angular.z*180/3.14
    def turnAroundRightLittle(self, vel):
        vel = Twist()
        for i in range(6):
            vel.angular.z = -3.14/2
            self.cmd_pub.publish(vel)
            self.r.sleep()
            continue
        print "try to turn right little with the degree of ", vel.angular.z*180/3.14
    def goStraight(self, vel):
        vel = Twist()
        for i in range(10):
            vel.linear.x = 0.3
            self.cmd_pub.publish(vel)
            self.r.sleep()
            continue
        print "try to go straight with speed of ", vel.linear.x
    def goBack(self, vel):
        vel = Twist()
        for i in range(10):
            vel.linear.x = -0.3
            self.cmd_pub.publish(vel)
            self.r.sleep()
            continue
        print "try to go back with speed of ", vel.linear.x