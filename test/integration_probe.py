"""Read-only, bounded mission smoke probe; run after launching a mission."""

import argparse
import json
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30.0)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = rclpy.create_node('mission_integration_probe')
    cells = []
    actions = []
    velocities = []
    node.create_subscription(OccupancyGrid, '/coverage_map',
        lambda message: cells.append(sum(value >= 0 for value in message.data)), 10)
    node.create_subscription(Int32, '/exploration_action',
        lambda message: actions.append(message.data), 10)
    node.create_subscription(Twist, '/cmd_vel',
        lambda message: velocities.append((message.linear.x, message.angular.z)), 10)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        publishers = node.get_publishers_info_by_topic('/cmd_vel')
        names = sorted(node.get_node_names())
        report = {
            'map_messages': len(cells),
            'known_cells_first': cells[0] if cells else 0,
            'known_cells_last': cells[-1] if cells else 0,
            'action_messages': len(actions),
            'velocity_messages': len(velocities),
            'nonzero_velocity_messages': sum(abs(v) + abs(w) > 0 for v, w in velocities),
            'cmd_vel_publishers': [info.node_name for info in publishers],
            'nodes': names,
        }
        print(json.dumps(report, indent=2))
        assert len(cells) >= 2 and cells[-1] > cells[0], 'Coverage did not grow'
        assert actions and report['nonzero_velocity_messages'], 'No autonomous actions/motion'
        assert report['cmd_vel_publishers'] == ['safe_motion_controller'], report
        assert not any(token in name.lower() for name in names
                       for token in ('nav2', 'slam', 'amcl', 'map_server', 'teleop')), names
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
