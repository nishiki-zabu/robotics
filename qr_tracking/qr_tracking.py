#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('cv_bridge')
import rospy
import os
import sys
import termios
import tty
import cv2
import numpy as np
import time
import random
import math
import colorsys
import traceback
import PIL.Image
import threading
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
import tf
import argparse

cmd_vel = '/base/diff_drive_controller/cmd_vel'
# cmd_vel = '/input_key/cmd_vel'
QSIZE = 1
exitFlag = 0
IMGWIDTH = 640
IMGHEIGHT = 480
DEPTHWIDTH = 640
DEPTHHEIGHT = 480
ANGLEWIDTH = 60
ANGLEHEIGHT = 30

rospy.init_node('vel_publisher')
pub = rospy.Publisher(cmd_vel,Twist, queue_size=QSIZE)

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1
def is_left_side(p, p0, p1):
    v0 = p1 - p0
    v1 = p - p0
    return  np.cross(v0, v1) < 0

class QRCode():
    def __init__(self, points, image):
        self.points = points
        self.image = image

class QRCodeFinder():
    def __init__(self):
        self.threshold = 64
        self.min_width = 640
        self.min_height = 480
        self.output_width= 128
        self.output_height = 128

        self._img_work = None
        self._img_dbg = None
        self._img_out = None

        self._contours = None
        self._hierarchies = None
        self._candidates = None
        self._patterns = None
        self._pattern_center = None
        self._outer_points = None
        self._center_point = None
        self._qr_codes = None
        self._drawer = QRCodeDebugDrawer()
        self._patterns_FLAG = 0

        self.bridge = CvBridge()
        self.camera = rospy.Subscriber('/camera/rgb/image_raw',Image, self.find)

        pass

    def find(self, img):
        self._prepare_img(img)
        self._find_contours()
        self._find_pattern_candidates()
        self._find_patterns()
        self._patterns_FLAG = 0
        """
        # QR debug
        check_QR()
        """
        if len(self._patterns) >= 3:
            self._sort_patterns()
            self._get_outer_points()
            self._patterns_FLAG = 1
            return self._build_result()

    def _prepare_img(self, image):
        img = self.bridge.imgmsg_to_cv2(image,"rgb8")
        self._check_QR_img = img

        w, h = img.shape[:2]
        r = max(self.min_width/w, self.min_height/h)

        if r > 1.0:
            itp = cv2.INTER_LANCZOS4
            img = cv2.resize(img, (int(r*h), int(r*w)), interpolation=itp)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_bin = cv2.threshold(img_gray, self.threshold, 255, 0)
        kernel = np.ones((2, 2), np.uint8)
        img_ibin = 255-img_bin
        img_ibin2 = cv2.morphologyEx(img_ibin, cv2.MORPH_OPEN, kernel)

        # img_ibin/img_ibin2 白黒反転画像
        self._img_work = img_ibin2
        # img_bin 閾値処理後画像
        self._img_out = img_bin
        self._img_bin = img_bin
        self._img_dbg = img.copy()

    def _find_contours(self):
        temp = self._img_work.copy()
        contours, hierarchies = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 頂点数を減らす
        new_contours = []
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            new_contours.append(approx)

        self._contours = new_contours
        self._hierarchies = hierarchies

        # cv2.polylines(self._img_dbg, self._contours, True, [0, 0, 255], 1)
        self._drawer.draw_has_child_or_parent(self._img_dbg, self._contours, self._hierarchies)

    def _find_root_node(self, hierarchy, leaf_index, target_depth):
        h = hierarchy[leaf_index]
        root_index = -1
        depth = 1
        # h[3]: parent
        while h[3] != -1 and depth < target_depth:
            root_index = h[3]
            depth += 1
            h = hierarchy[h[3]]
        if h[3] == -1 and depth == target_depth:
            return root_index
        return -1
        
    def _find_pattern_candidates(self):
        contours = self._contours
        hierarchies = self._hierarchies
        candidates = []
        for hierarchy in hierarchies:
            for i, h in enumerate(hierarchy):
                next_cnt, prev_cnt, first_child, parent = h
                if not (first_child == -1 and parent != -1):
                    continue
                root_index = self._find_root_node(hierarchy, i, 3)
                if root_index != -1 and not (root_index in candidates):
                    candidates.append(root_index)
        self._candidates = candidates
        self._drawer.draw_candidates(self._img_dbg, self._contours, self._candidates)

    def _is_valid_pattern(self, pat):
        if len(pat) != 7:
            return False
        a = pat[0]
        if not (pat[2] == pat[3] == pat[4] == pat[6] == a):
            return False
        if not (pat[1] != a and pat[5] != a):
            return False
        return True

    def _contour_to_box(self, cnt):
        box = np.array(map(lambda x: x[0], cnt))
        return box

    def _find_pattern(self, img, cnt):
        box = self._contour_to_box(cnt)
        if len(box) != 4:
            return None
        num = 7
        denom = float(num)
        pts_dbg = []
        # check p0 to p2 and p1 to p3
        for j in range(2):
            x0, y0 = box[j]
            x1, y1 = box[j+2]
            dx, dy = (x1 - x0) / denom, (y1 - y0) / denom
            x0 += dx * 0.5
            y0 += dy * 0.5
            pat = []
            for i in range(num):
                x, y = int(x0 + i * dx), int(y0 + i * dy)
                pat.append(img[y, x])
                pts_dbg.append((x, y))
            if not self._is_valid_pattern(pat):
                box = None
                break
        # debug draw
        for p in pts_dbg:
            cv2.circle(self._img_dbg, p, 1, [255, 255, 0], 2)
        return box

    def _find_patterns(self):
        contours = self._contours
        patterns = []
        for c in self._candidates:
            cnt = contours[c]
            box = self._find_pattern(self._img_work, cnt)
            if box is None:
                #print 'contour[%d] is not pattern' % c
                continue
            patterns.append((c, box))
        # TODO: 候補が4個以上の場合の対応
        if len(patterns) > 3:
            patterns = patterns[:3]
        self._patterns = patterns

    # patterns = [(index0, box0), (index1, box1), (index2, box2)]
    def _find_top_left_pattern(self, patterns):
        max_index = -1
        max_length = 0

        for i, pat in enumerate(patterns):
            _, box0 = patterns[i]
            _, box1 = patterns[(i + 1) % 3]
            c0 = np.mean(box0, axis=0)
            c1 = np.mean(box1, axis=0)
            l = sum((c1 - c0) ** 2)
            if max_index == -1 or l > max_length:
                max_index = (i + 2) % 3
                max_length = l
        return max_index

    def _sort_patterns(self):
        patterns = self._patterns
        idx_tl = self._find_top_left_pattern(patterns)
        if idx_tl == -1:
            raise Exception('_sort_patterns [ERROR] cannot find top-left pattern')

        # calculate center point
        idx_tr = (idx_tl + 1) % 3
        idx_bl = (idx_tl + 2) % 3
        c0 = np.mean(patterns[idx_tr][1], axis=0)
        c1 = np.mean(patterns[idx_bl][1], axis=0)
        center = np.mean([c0, c1], axis=0)

        # sort patterns to (top-left, top-right, bottom-left)
        p = np.mean(patterns[idx_tl][1], axis=0)
        if not is_left_side(c0, p, center):
            idx_tr, idx_bl = idx_bl, idx_tr

        indices = [idx_tl, idx_tr, idx_bl]
        self._patterns = [patterns[i] for i in indices]
        self._pattern_center = center

        # draw top-left pattern with blue color
        cv2.polylines(self._img_dbg, [self._patterns[0][1]], True, [255, 0, 0], 2)
        # draw center with green color
        temp = np.int32(self._pattern_center)
        cv2.circle(self._img_dbg, (temp[0], temp[1]), 4, [0, 255, 0], 2)
        
    def _get_outer_point(self, points, center):
        max_index = -1
        max_length = 0
        for i, p in enumerate(points):
            l = sum((center - p) ** 2)
            if max_index == -1 or l > max_length:
                max_index = i
                max_length = l
        return max_index

    def _get_outer_points(self):

        patterns = []
        patterns = self._patterns
        # center:QR二次元中心座標  
        center = self._pattern_center
        # _outer_points(0:左上　1:右上　2:　左下　3:右下)/_center_point(中央)
        self._qr_center = center
        outer_points = []

        # top-left
        points = patterns[0][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])
        self._qr_topleft = points[idx]
        
        # top-right
        points = patterns[1][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])
        self._qr_topright = points[idx]

        idx_next = (idx + 1) % 4
        if not is_left_side(points[idx_next], points[idx], points[(idx + 2) % 4]):
            idx_next = (idx + 3) % 4
        p_tr = points[idx_next]

        # bottom-left
        points = patterns[2][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])
        self._qr_bottomleft = points[idx]

        idx_next = (idx + 1) % 4
        if is_left_side(points[idx_next], points[idx], points[(idx + 2) % 4]):
            idx_next = (idx + 3) % 4
        p_bl = points[idx_next]

        # calculate bottom-right point
        try:
            p_br = seg_intersect(outer_points[1], p_tr, outer_points[2], p_bl)
            p_br = np.int32(p_br)
            outer_points.append(p_br)
            self._qr_bottomright = p_br
        except:
            pass

        self._outer_points = outer_points

        # draw outer points
        for p in outer_points:
            cv2.circle(self._img_dbg, (p[0], p[1]), 4, [0, 255, 0], 2)
        cv2.circle(self._img_dbg, (p_tr[0], p_tr[1]), 4, [0, 255, 255], 2)
        cv2.circle(self._img_dbg, (p_bl[0], p_bl[1]), 4, [0, 255, 255], 2)

    # _crop_qr_coder:並進回転補正関数
    def _crop_qr_code(self, img_src, pts_src, width, height):
        if (pts_src is None):
            print "not detect"
        
        if pts_src is None:
            num =  1
        else:
            num = min(len(pts_src), 4)

        pts_src = np.array(pts_src[:num], dtype=np.float32)
        pts_dst = [(0, 0), (width, 0), (0, height)]
        if num == 3:
            pts_dst = np.array(pts_dst, dtype=np.float32).reshape(3, 2)
            M = cv2.getAffineTransform(pts_src, pts_dst)
            # アフィン変換:cv2.warpAffine(画像,回転・移動行列(2*3),画像サイズ,flags)
            img_dst = cv2.warpAffine(img_src, M, (height, width))
            return img_dst
        if num == 4:
            pts_dst.append((width, height))
            pts_dst = np.array(pts_dst, dtype=np.float32).reshape(4, 2)
            # 透視変換行列の生成:cv2.getPerspectiveTransform(変換前座標,変換後座標)
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            # 透視変換:cv2.warpPerspective(画像,回転・並進行列（3*3）,画像サイズ)
            img_dst = cv2.warpPerspective(img_src, M, (height, width))
            return img_dst
        return None

    """
    # _crop_qr_coder:並進回転補正関数抜き
    def _crop_qr_code(self, img_src, pts_src, width, height):
        if (pts_src is None):
            print "pts_src is not detected"
        if pts_src is None:
            num =  1
        else:
            num = min(len(pts_src), 4)
        pts_src = np.array(pts_src[:num], dtype=np.float32)
        return img_src
    """

    def _build_result(self):
        qrs = []

        w, h = self.output_width, self.output_height
        # self._img_out:閾値処理済画像/self._outer_points:QR二次元座標
        img = self._crop_qr_code(self._img_out, self._outer_points, w, h)

        if not (img is None):
            # TODO: 最小サイズの考慮
            qr = QRCode(self._outer_points, img)
            qrs.append(qr)
        
        return qrs

    def show_bin(self, title='bin', wait=True):
        cv2.imshow(title, self._img_bin)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAll

    def show_debug(self, title='img', wait=True):
        cv2.imshow(title, self._img_dbg)

class QRCodeDebugDrawer():
    def __init__(self):
        pass

    def draw_polyline(self, img, cnt, isClosed=True, color=[0, 0, 0], thickness=1):
        length = len(cnt)
        if length < 2:
            return
        p0 = tuple(cnt[0][0][:2])
        first = p0
        for i in range(1, length):
            p1 = tuple(cnt[i][0][:2])
            cv2.line(img, p0, p1, color, thickness)
            p0 = p1
        if isClosed:
            cv2.line(img, p0, first, color, thickness)

    # draw only has child or parent contours
    def draw_has_child_or_parent(self, img, contours, hierarchies):
        for hierarchy in hierarchies:
            for i, h in enumerate(hierarchy):
                next_cnt, prev_cnt, first_child, parent = h
                if first_child == -1 and parent == -1:
                    continue
                cnt = contours[i]
                # cv2.polylines(img, cnt, True, [0, 0, 255], 1)
                self.draw_polyline(img, cnt, True, [0, 0, 255], 1)

    def draw_candidates(self, img, contours, candidates):
        for c in candidates:
            cnt = contours[c]
            self.draw_polyline(img, cnt, True, [0, 255, 255], 1)

class depth_converter:
    def __init__(self):
        self.bridge = CvBridge()
        global Image
        self._cv_image = []
        self.image_sub = rospy.Subscriber('/camera/depth/image_raw',Image, self.callback)
    def callback(self, data):
        try:
            self._cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding="passthrough") 
        except Exception as e:
            print 'dc type:'+str(type(e))
            print 'dc args:'+str(e.args)
            print 'dc message:' + e.message
            print 'dc e itself' + str(e)

def realKey():
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

class get_pose:
    def __init__(self):
        self._pose_euler = Twist()
        self._pose_quaternion= Quaternion()
        self.pose_sub = rospy.Subscriber("/base/diff_drive_controller/odom",Odometry,self.pose_callback)
    def quaternion_to_euler(self,quaternion):
        euler = tf.transformations.euler_from_quaternion((quaternion.x,quaternion.y,quaternion.z,quaternion.w))
        return Vector3(x=euler[0],y=euler[1],z=euler[2])
    def pose_callback(self,odom_data):
        self.current_time = odom_data.header.stamp
        self._pose_quaternion = odom_data.pose.pose
        self._pose_euler.linear = self._pose_quaternion.position
        self._pose_euler.angular = self.quaternion_to_euler(self._pose_quaternion.orientation)

def check_QR():

    finder = QRCodeFinder()
    finder.threshold = 127
    finder.output_width = 640
    finder.output_height = 480
    qrs = []

    while True:
        try:
            qrs = finder.find(finder._check_rgb)
        except:
            traceback.print_exc()
        print '%d QR code(s) found' % len(qrs)
        for qr in qrs:
        # dump region of QR code
            print ', '.join(['(%d, %d)' % (p[0], p[1]) for p in qr.points])
        # finder.show_bin(wait=False)
        finder.show_debug()
        for i, qr in enumerate(qrs):
            cv2.imshow('qr_%d' % i, qr.image)
        k = cv2.waitKey(1)
        if k == 113:
            break

    cap.release()
    cv2.destroyAllWindows()
    pass

def QR_tracking():
    global exitFlag
    global run

    qr_image = QRCodeFinder()
    depth_image = depth_converter()
    pose = get_pose()

    image_width_center = (640.0/2)
    image_height_center = (480.0/2)
    depth_wide_center = (640.0/2)
    depth_height_center = (480.0/2)
    angle_width = (120.0/2)
    angle_height = (60.0/2)
    terminal_horizontal = 0
    vel = Twist()

    while not rospy.is_shutdown() :

        print "start rospy"

        while True :
            if qr_image._patterns_FLAG == 1:
                break

        # 光軸とQRの角度偏差
        qr_center_x = qr_image._qr_center[0]
        qr_center_y = qr_image._qr_center[1]
        theta_center = angle_width*(qr_center_x-image_width_center)/image_width_center
        phi_center = angle_height*(qr_center_y-image_height_center)/image_height_center

        while True :
            if len(depth_image._cv_image) != 0 :
                break 

        qr_center_x_int = int(qr_center_x)
        qr_center_y_int = int(qr_center_y)
        depth_center = depth_image._cv_image[qr_center_y_int][qr_center_x_int]

        # QRを光軸に捉える
        while abs(theta_center) > 5 : 
            if abs(theta_center) < 5:
                vel.angular.z = 0
                pub.publish(vel)
                break
            qr_center_x = qr_image._qr_center[0]
            theta_center = angle_width*(qr_center_x-image_width_center)/image_width_center
            rad_theta_center = math.radians(theta_center)
            vel.angular.z = -rad_theta_center
            pub.publish(vel)

        print "finish rotation1"

        # QRの正面に移動する（回転+横進+回転）
        while True :

            # QRの二次元座標+角度偏差+depthの取得
            qr_topleft_x = qr_image._qr_topleft[0]
            qr_topleft_y = qr_image._qr_topleft[1]
            qr_topright_x = qr_image._qr_topright[0]
            qr_topright_y = qr_image._qr_topright[1]
            qr_bottomleft_x = qr_image._qr_bottomleft[0]
            qr_bottomleft_y = qr_image._qr_bottomleft[0]

            theta_topleft = angle_width*(qr_topleft_x-image_width_center)/image_width_center
            phi_topleft = angle_height*(qr_topleft_y-image_height_center)/image_height_center
            theta_topright = angle_width*(qr_topright_x-image_width_center)/image_width_center
            phi_topright = angle_height*(qr_topright_y-image_height_center)/image_height_center
            theta_bottomleft = angle_width*(qr_bottomleft_x-image_width_center)/image_width_center
            phi_bottomleft = angle_height*(qr_bottomleft_y-image_height_center)/image_height_center

            rad_theta_topleft = math.radians(theta_topleft)
            rad_phi_topleft = math.radians(phi_topleft)
            rad_theta_topright = math.radians(theta_topright)
            rad_phi_topright = math.radians(phi_topright)
            rad_theta_bottomleft = math.radians(theta_bottomleft)
            rad_phi_bottomleft = math.radians(phi_bottomleft)

            depth_topleft = depth_image._cv_image[qr_topleft_y][qr_topleft_x]
            depth_topright = depth_image._cv_image[qr_topright_y][qr_topright_x]
            depth_bottomleft = depth_image._cv_image[qr_bottomleft_y][qr_bottomleft_x]

            # 三次元相対座標の取得/右手系(原点:自機　x軸：右方向　y軸:上方向　z軸:手前方向)
            topleft_coordinate_x = depth_topleft*math.cos(rad_phi_topleft)*math.sin(rad_theta_topleft)
            topleft_coordinate_y = depth_topleft*math.sin(rad_phi_topleft)
            topleft_coordinate_z = depth_topleft*math.cos(rad_phi_topleft)*math.cos(rad_theta_topleft)

            topright_coordinate_x = depth_topright*math.cos(rad_phi_topright)*math.sin(rad_theta_topright)
            topright_coordinate_y = depth_topright*math.sin(rad_phi_topright)
            topright_coordinate_z = depth_topright*math.cos(rad_phi_topright)*math.cos(rad_theta_topright)

            bottomleft_coordinate_x = depth_bottomleft*math.cos(rad_phi_bottomleft)*math.sin(rad_theta_bottomleft)
            bottomleft_coordinate_y = depth_bottomleft*math.sin(rad_phi_bottomleft)
            bottomleft_coordinate_z = depth_bottomleft*math.cos(rad_phi_bottomleft)*math.cos(rad_theta_bottomleft)

            # 外積で利用するQRの水平ベクトル/垂直ベクトル
            qr_horizontal_vector = ((topleft_coordinate_x-topright_coordinate_x),(topleft_coordinate_y-topright_coordinate_y),(topleft_coordinate_z-topright_coordinate_z))
            qr_vertical_vector = ((topleft_coordinate_x-bottomleft_coordinate_x),(topleft_coordinate_y-bottomleft_coordinate_y),(topleft_coordinate_z-bottomleft_coordinate_z))

            # 外積ベクトル
            cross_product_x = 0
            cross_product_y = 0
            cross_product_z = 0

            cross_product_x = qr_horizontal_vector[1]*qr_vertical_vector[2]-qr_horizontal_vector[2]*qr_vertical_vector[1]
            cross_product_y = -(qr_horizontal_vector[0]*qr_vertical_vector[2]-qr_horizontal_vector[2]*qr_vertical_vector[0])
            cross_product_z = qr_horizontal_vector[0]*qr_vertical_vector[1]-qr_horizontal_vector[1]*qr_vertical_vector[0]

            cross_product = (cross_product_x,cross_product_y,cross_product_z)

            # 光軸のベクトル
            image_width_center_int = int(image_width_center)
            image_height_center_int = int(image_height_center)

            pre_depth_optical_axis_x = qr_image._qr_center[0]
            pre_depth_optical_axis_y = qr_image._qr_center[1]
            pre_depth_optical_axis_x_int = int(pre_depth_optical_axis_x)
            pre_depth_optical_axis_y_int = int(pre_depth_optical_axis_y)
            phi_depth_optical = angle_height*(pre_depth_optical_axis_y-image_height_center)/image_height_center
            pre_depth_optical_axis = depth_image._cv_image[pre_depth_optical_axis_y_int][pre_depth_optical_axis_x_int]
            depth_optical_axis = pre_depth_optical_axis*math.cos(phi_depth_optical)
            center_vector = (0,0,depth_optical_axis)

            cross_product_dot = (cross_product[0],0,cross_product[2])

            if (cross_product[0] >= 0):
                theta_horizontal_FLAG = 1
            else:
                theta_horizontal_FLAG = 0

            dot_product = 0
            abs_vector_squar = 0

            # 光軸と外積ベクトルとの内積/角度
            for i in range(3):
                dot_product += center_vector[i]*cross_product_dot[i]
                abs_vector_squar += (center_vector[i]*center_vector[i]+cross_product_dot[i]*cross_product_dot[i])
            abs_vector = math.sqrt(abs_vector_squar)
            cos_theta_horizontal = dot_product/abs_vector
            pre_rad_theta_horizontal = math.acos(cos_theta_horizontal)
            rad_theta_horizontal = math.pi-pre_rad_theta_horizontal
            theta_horizontal = math.degrees(rad_theta_horizontal)

            if terminal_horizontal == 0:
                terminal_horizontal = theta_horizontal
            else:
                print abs(theta_horizontal-90)/abs(terminal_horizontal-90)

            if abs(terminal_horizontal-90)*0.95>abs(theta_horizontal-90) :
                break
            # depth_optical_axis:qr_horizontal_vectorとの平面距離/lateral_distance:横移動距離
            lateral_distance = depth_optical_axis*math.cos(rad_theta_horizontal)

            # 回転運動(rad_theta_horizontal)
            pre_pose_angular_z = pose._pose_euler.angular.z
            pose_angular_z = pose._pose_euler.angular.z

            while True :
                pose_angular_z = pose._pose_euler.angular.z
                if abs(pre_pose_angular_z-pose_angular_z) >= abs(rad_theta_horizontal):
                    vel.angular.z = 0
                    pub.publish(vel)
                    print "finish rotation2"
                    break
                if theta_horizontal_FLAG == 0:
                    vel.angular.z = rad_theta_horizontal
                elif theta_horizontal_FLAG == 1:
                    vel.angular.z = -rad_theta_horizontal
                pub.publish(vel)
            
            pre_pose_linear_x = pose._pose_euler.linear.x
            pre_pose_linear_y = pose._pose_euler.linear.y
            pre_pose_linear_z = pose._pose_euler.linear.z

            # 直進運動(lateral_distance)
            while True :
                pose_linear_x = pose._pose_euler.linear.x
                pose_linear_y = pose._pose_euler.linear.y
                pose_linear_z = pose._pose_euler.linear.z

                pose_deviation_x = pose_linear_x-pre_pose_linear_x
                pose_deviation_y = pose_linear_y-pre_pose_linear_y
                pose_deviation_z = pose_linear_z-pre_pose_linear_z

                pose_linear_distance_square = pose_deviation_x*pose_deviation_x+pose_deviation_y*pose_deviation_y+pose_deviation_z*pose_deviation_z
                pose_linear_distance = math.sqrt(pose_linear_distance_square)

                if abs(pose_linear_distance) > abs(lateral_distance):
                    vel.linear.x = 0
                    pub.publish(vel)
                    print "finish laternal"
                    break

                vel.linear.x = 0.1
                pub.publish(vel)

            # 回転運動(90度)
            pre_pose_angular_z = pose._pose_euler.angular.z
            pose_angular_z = pose._pose_euler.angular.z

            while True :
                pose_angular_z = pose._pose_euler.angular.z
                if abs(pre_pose_angular_z-pose_angular_z) >= math.pi/2:
                    vel.angular.z = 0
                    pub.publish(vel)
                    print "finish rotation3"
                    break

                if theta_horizontal_FLAG == 1:
                    vel.angular.z = rad_theta_horizontal
                elif theta_horizontal_FLAG == 0:
                    vel.angular.z = -rad_theta_horizontal

                pub.publish(vel)

            break

        if abs(terminal_horizontal-90)*0.95>abs(theta_horizontal-90):
            print "finish rospy"
            break

        print "finish all process"
        # 繰り返す場合コメントアウト
        break

if __name__=='__main__':
    thread = threading.Thread(target = realKey)
    thread.start()
    print "start QR_tracking()"
    QR_tracking()
