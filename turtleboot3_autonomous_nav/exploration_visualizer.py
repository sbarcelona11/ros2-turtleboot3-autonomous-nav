"""Read-only RViz markers for exploration coverage and motion."""

import math

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


class ExplorationVisualizer(Node):
    def __init__(self):
        super().__init__('exploration_visualizer')
        self._velocity = Twist()
        self._coverage = 0.0
        self._monitoring_map: OccupancyGrid | None = None
        self._trajectory: list[Point] = []
        self._trajectory_frame = 'odom'
        self.create_subscription(Twist, '/cmd_vel', self._on_velocity, 10)
        self.create_subscription(Float32, '/coverage_metrics', self._on_coverage, 10)
        self.create_subscription(
            OccupancyGrid, '/monitoring_map', self._on_monitoring_map, 10
        )
        self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
        self._publisher = self.create_publisher(MarkerArray, '/exploration_status', 10)
        self.create_timer(0.2, self._publish)

    def _on_velocity(self, message):
        self._velocity = message

    def _on_coverage(self, message):
        self._coverage = message.data

    def _on_monitoring_map(self, message: OccupancyGrid):
        self._monitoring_map = message

    def _on_odometry(self, message: Odometry):
        position = message.pose.pose.position
        point = Point(x=position.x, y=position.y, z=0.06)
        if (
            self._trajectory
            and math.hypot(
                point.x - self._trajectory[-1].x,
                point.y - self._trajectory[-1].y,
            ) < 0.02
        ):
            return
        self._trajectory.append(point)
        self._trajectory = self._trajectory[-5000:]
        self._trajectory_frame = message.header.frame_id or 'odom'

    def _publish(self):
        arrow = Marker()
        arrow.header.frame_id = 'base_footprint'
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'cmd_vel'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.orientation.w = 1.0
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.03, 0.07, 0.10
        arrow.color.g, arrow.color.a = 1.0, 1.0
        # Arrow shows the command's one-second direction and scaled translation.
        angle = self._velocity.angular.z
        distance = abs(self._velocity.linear.x) * 3.0
        arrow.points = [Point(z=0.15), Point(
            x=distance * math.cos(angle), y=distance * math.sin(angle), z=0.15)]
        label = Marker()
        label.header = arrow.header
        label.ns = 'coverage_metrics'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.orientation.w = 1.0
        label.pose.position.z = 0.75
        label.scale.z = 0.16
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = (f'Cobertura: {self._coverage:.1%}\n'
                      f'cmd_vel: {self._velocity.linear.x:.2f} m/s, '
                      f'{self._velocity.angular.z:.2f} rad/s')

        monitored = Marker()
        monitored.header.stamp = arrow.header.stamp
        monitored.header.frame_id = 'odom'
        monitored.ns = 'monitored_area'
        monitored.id = 2
        monitored.type = Marker.CUBE_LIST
        monitored.action = Marker.ADD
        monitored.pose.orientation.w = 1.0
        monitored.scale.z = 0.02
        monitored.color.g, monitored.color.b, monitored.color.a = 0.8, 0.7, 0.45
        grid = self._monitoring_map
        if grid is not None:
            monitored.header.frame_id = grid.header.frame_id or 'odom'
            resolution = float(grid.info.resolution)
            width, height = int(grid.info.width), int(grid.info.height)
            if resolution > 0.0 and width > 0 and height > 0 and len(grid.data) == width * height:
                monitored.scale.x = monitored.scale.y = resolution
                origin = grid.info.origin.position
                monitored.points = [
                    Point(
                        x=origin.x + (index % width + 0.5) * resolution,
                        y=origin.y + (index // width + 0.5) * resolution,
                        z=0.01,
                    )
                    for index, value in enumerate(grid.data)
                    if value == 0
                ]

        trail = Marker()
        trail.header.stamp = arrow.header.stamp
        trail.header.frame_id = self._trajectory_frame
        trail.ns = 'robot_trajectory'
        trail.id = 3
        trail.type = Marker.LINE_STRIP
        trail.action = Marker.ADD
        trail.pose.orientation.w = 1.0
        trail.scale.x = 0.035
        trail.color.r, trail.color.g, trail.color.a = 1.0, 0.85, 1.0
        trail.points = self._trajectory

        self._publisher.publish(MarkerArray(markers=[arrow, label, monitored, trail]))


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
