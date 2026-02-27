import os
import json
import yaml
import math
from typing import Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from ament_index_python.packages import get_package_share_directory


CMD_TOPIC = "/robot_cmd"
STATUS_TOPIC = "/robot_status"
FRAME_MAP = "map"
NAV2_ACTION_NAME = "navigate_to_pose"  # đa số Nav2 dùng tên này


class CommanderNode(Node):
    """
    Laptop node:
    - Subscribe /robot_cmd (JSON): {"type":"go","goal":"kitchen"} hoặc {"type":"cancel"}
    - Load waypoints.yaml từ share/robot_commander/config/waypoints.yaml
    - Gửi goal tới Nav2 action navigate_to_pose
    - Publish /robot_status (JSON) cho Raspberry
    """

    def __init__(self):
        super().__init__("commander_node")

        self.cmd_sub = self.create_subscription(String, CMD_TOPIC, self.on_cmd, 10)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        self._ac = ActionClient(self, NavigateToPose, NAV2_ACTION_NAME)

        self.busy: bool = False
        self.current_goal_name: Optional[str] = None
        self.current_goal_handle = None

        self.waypoints: Dict[str, Any] = self.load_waypoints()

        self.publish_status({
            "event": "ready",
            "waypoints": list(self.waypoints.keys()),
            "cmd_topic": CMD_TOPIC,
            "status_topic": STATUS_TOPIC,
            "action": NAV2_ACTION_NAME
        })
        self.get_logger().info(f"READY. Loaded waypoints: {list(self.waypoints.keys())}")

    # -------------------------
    # Utils
    # -------------------------
    def publish_status(self, payload: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def load_waypoints(self) -> Dict[str, Any]:
        """
        Tìm config/waypoints.yaml trong share directory của package.
        YAML format gợi ý:

        waypoints:
          kitchen:
            position: {x: 1.0, y: 2.0}
            yaw: 1.57
          A:
            position: {x: 1.0, y: 2.0}
            orientation: {z: 0.0, w: 1.0}
        """
        pkg_share = get_package_share_directory("robot_supervisor")
        yaml_path = os.path.join(pkg_share, "config", "waypoints.yaml")

        if not os.path.exists(yaml_path):
            self.get_logger().error(f"Missing waypoints.yaml at: {yaml_path}")
            return {}

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        wps = data.get("waypoints", {})
        if not isinstance(wps, dict):
            self.get_logger().error("waypoints.yaml format error: 'waypoints' must be a dict")
            return {}

        return wps

    def waypoint_to_pose(self, wp: Dict[str, Any]) -> PoseStamped:
        """
        Convert 1 waypoint dict -> PoseStamped
        Hỗ trợ:
        - position: {x,y} + yaw
        - position: {x,y} + orientation: {z,w}
        """
        pose = PoseStamped()
        pose.header.frame_id = FRAME_MAP
        pose.header.stamp = self.get_clock().now().to_msg()

        pos = wp.get("position", {})
        pose.pose.position.x = float(pos.get("x", 0.0))
        pose.pose.position.y = float(pos.get("y", 0.0))
        pose.pose.position.z = 0.0

        if "orientation" in wp:
            ori = wp["orientation"]
            # chỉ dùng z,w cho 2D
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = float(ori.get("z", 0.0))
            pose.pose.orientation.w = float(ori.get("w", 1.0))
        else:
            yaw = float(wp.get("yaw", 0.0))
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

        return pose

    # -------------------------
    # Command handling
    # -------------------------
    def on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except Exception:
            self.publish_status({"event": "bad_json", "raw": msg.data})
            return

        ctype = cmd.get("type")

        if ctype == "go":
            goal_name = cmd.get("goal")
            if not goal_name:
                self.publish_status({"event": "missing_goal"})
                return

            if self.busy:
                self.publish_status({"event": "busy", "current_goal": self.current_goal_name})
                return

            if goal_name not in self.waypoints:
                self.publish_status({"event": "unknown_goal", "goal": goal_name, "known": list(self.waypoints.keys())})
                return

            self.start_navigation(goal_name)

        elif ctype == "cancel":
            self.cancel_navigation()

        elif ctype == "reload_waypoints":
            self.waypoints = self.load_waypoints()
            self.publish_status({"event": "waypoints_reloaded", "waypoints": list(self.waypoints.keys())})

        else:
            self.publish_status({"event": "unknown_cmd_type", "type": ctype})

    def start_navigation(self, goal_name: str):
        if not self._ac.wait_for_server(timeout_sec=2.0):
            self.publish_status({"event": "nav2_not_ready"})
            return

        wp = self.waypoints[goal_name]
        pose = self.waypoint_to_pose(wp)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.busy = True
        self.current_goal_name = goal_name
        self.publish_status({"event": "nav_start", "goal": goal_name})

        send_future = self._ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.publish_status({"event": "goal_rejected", "goal": self.current_goal_name})
            self.busy = False
            self.current_goal_name = None
            self.current_goal_handle = None
            return

        self.current_goal_handle = goal_handle
        self.publish_status({"event": "goal_accepted", "goal": self.current_goal_name})

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result()
        status_code = int(result.status)

        # status_code thường: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (tuỳ action impl)
        event = "nav_done"
        if status_code == 4:
            event = "nav_succeeded"
        elif status_code == 5:
            event = "nav_canceled"
        elif status_code == 6:
            event = "nav_aborted"

        self.publish_status({
            "event": event,
            "goal": self.current_goal_name,
            "status": status_code
        })

        self.busy = False
        self.current_goal_name = None
        self.current_goal_handle = None

    def cancel_navigation(self):
        if not self.current_goal_handle:
            self.publish_status({"event": "no_active_goal"})
            return

        cancel_future = self.current_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(lambda f: self.publish_status({"event": "cancel_requested"}))


def main():
    rclpy.init()
    node = CommanderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()