#!/usr/bin/env python3
import math
import time
import serial
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

K_PULSE2V = 0.00013
K_PULSE2W = 0.0035

def quaternion_from_yaw(yaw: float) -> Quaternion:
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

class BridgeOdom(Node):
    def __init__(self):
        super().__init__('bridge_odom')

        # ===== Parameters =====
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("wheel_radius", 0.0425)
        self.declare_parameter("wheel_separation", 0.345)
        self.declare_parameter("ticks_per_rev", 300.0)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("broadcast_tf", True)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.R = float(self.get_parameter("wheel_radius").value)
        self.L = float(self.get_parameter("wheel_separation").value)
        self.ticks_per_rev = float(self.get_parameter("ticks_per_rev").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.broadcast_tf = bool(self.get_parameter("broadcast_tf").value)

        hz = float(self.get_parameter("publish_hz").value)
        self.dt = 1.0 / max(1.0, hz)

        # ===== Serial (OPEN ONCE) =====
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)  # đợi Mega reset
        self.get_logger().info(f"Serial opened: {self.port} @ {self.baud}")

        # ===== Odom state =====
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_l = None
        self.prev_r = None
        self.prev_time = self.get_clock().now()

        # ===== ROS I/O =====
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ✅ ADD: subscriber cmd_vel (từ ArduinoBridge)
        self.sub_cmd = self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)

        # timer
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("Bridge+Odom ready (1 serial port).")

    # ===== parse encoder line (3 formats) =====
    def parse_encoder_line(self, line: str):
        line = line.strip()

        # 1) E,<L>,<R>
        if line.startswith("E,"):
            try:
                _, l, r = line.split(",")
                return int(l), int(r)
            except Exception:
                return None

        # 2) L,R
        if "," in line and "ticksL=" not in line:
            try:
                l, r = line.split(",")
                return int(l), int(r)
            except Exception:
                return None

        # 3) ticksL=... ticksR=... | ...
        if "ticksL=" in line and "ticksR=" in line:
            try:
                tmp = line.split("|")[0].strip()
                tmp = tmp.replace("ticksL=", "").replace("ticksR=", "")
                parts = tmp.split()
                return int(parts[0]), int(parts[1])
            except Exception:
                return None

        return None

    # ✅ ADD: cmd_vel -> w/a/s/d (lấy từ ArduinoBridge nhưng dùng chung self.ser)
    def cmd_callback(self, msg: Twist):
        V = float(msg.linear.x)      # m/s
        W = float(msg.angular.z)     # rad/s

        # differential drive: tốc độ bánh trái/phải (m/s)
        v_left  = V - (W * self.L * 0.5)
        v_right = V + (W * self.L * 0.5)

        # đổi m/s -> pulse (theo hằng số bạn đang dùng)
        # pulse có dấu để thể hiện chiều (+ tiến, - lùi)
        pL = int(v_left / K_PULSE2V)
        pR = int(v_right / K_PULSE2V)

        # giới hạn tránh quá lớn
        pL = max(min(pL, 9999), -9999)
        pR = max(min(pR, 9999), -9999)

        # protocol mới: "m,<pL>,<pR>"
        cmd = f"m,{pL},{pR}\n"

        try:
            self.ser.write(cmd.encode("ascii", errors="ignore"))
        except Exception as e:
            self.get_logger().warn(f"Serial write error: {e}")

    # ===== read ticks -> publish odom =====
    def loop(self):
        latest = None
        try:
            while True:
                raw = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not raw:
                    break
                parsed = self.parse_encoder_line(raw)
                if parsed:
                    latest = parsed
        except Exception as e:
            self.get_logger().warn(f"Serial read error: {e}")
            return

        if latest is None:
            return

        l_now, r_now = latest


        if self.prev_l is None:
            self.prev_l, self.prev_r = l_now, r_now
            self.prev_time = self.get_clock().now()
            return

        now_time = self.get_clock().now()
        dt = (now_time - self.prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = self.dt
        self.prev_time = now_time

        dl_ticks = l_now - self.prev_l
        dr_ticks = r_now - self.prev_r
        self.prev_l, self.prev_r = l_now, r_now

        meters_per_tick = (2.0 * math.pi * self.R) / self.ticks_per_rev
        d_left = dl_ticks * meters_per_tick
        d_right = dr_ticks * meters_per_tick

        d_s = 0.5 * (d_right + d_left)
        d_theta = (d_right - d_left) / self.L

        self.x += d_s * math.cos(self.theta + 0.5 * d_theta)
        self.y += d_s * math.sin(self.theta + 0.5 * d_theta)
        self.theta = (self.theta + d_theta + math.pi) % (2.0 * math.pi) - math.pi

        vx = d_s / dt
        wz = d_theta / dt

        odom = Odometry()
        odom.header.stamp = now_time.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = quaternion_from_yaw(self.theta)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        if self.broadcast_tf:
            t = TransformStamped()
            t.header.stamp = now_time.to_msg()
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation = quaternion_from_yaw(self.theta)
            self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        try:
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = BridgeOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
