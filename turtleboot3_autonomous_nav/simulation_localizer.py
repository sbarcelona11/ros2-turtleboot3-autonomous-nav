"""Publish the transform tree from the pose the map is actually built with.

The map now comes from the simulator's own pose and matches the arena to 8 cm,
but Gazebo still publishes `odom -> base_footprint` from the wheels, which
drifted 1.3 m and 163.8 degrees in a mission.  RViz places the robot and its
laser through that transform, so a correct map appeared with the scan drawn a
metre away from it - it looked like a broken system.

Republishing the transform from the same odometry the map uses makes the
picture consistent.  The rest of the chain, `base_footprint -> base_link ->
base_scan`, still comes from the robot state publisher.
"""

from __future__ import annotations


def transform_from_odometry(odometry):
    """Return the transform an odometry message describes.

    The frames are taken from the message rather than configured here: they
    have to be the ones the map is drawn in, and renaming either would detach
    the robot from it.
    """
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.stamp = odometry.header.stamp
    transform.header.frame_id = odometry.header.frame_id
    transform.child_frame_id = odometry.child_frame_id
    transform.transform.translation.x = odometry.pose.pose.position.x
    transform.transform.translation.y = odometry.pose.pose.position.y
    transform.transform.translation.z = odometry.pose.pose.position.z
    transform.transform.rotation = odometry.pose.pose.orientation
    return transform


def main(args: list[str] | None = None) -> None:
    """Broadcast the robot transform from the simulator-provided odometry."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster

    class SimulationLocalizer(Node):
        """Turn the true odometry into the transform the displays rely on."""

        def __init__(self) -> None:
            super().__init__('simulation_localizer')
            self.declare_parameter('odometry_topic', '/odom_truth')
            self._broadcaster = TransformBroadcaster(self)
            self.create_subscription(
                Odometry,
                str(self.get_parameter('odometry_topic').value),
                self._on_odometry,
                10,
            )

        def _on_odometry(self, message: Odometry) -> None:
            self._broadcaster.sendTransform(transform_from_odometry(message))

    rclpy.init(args=args)
    node = SimulationLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
