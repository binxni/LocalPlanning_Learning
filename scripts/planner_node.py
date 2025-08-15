import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import torch

from preprocess import preprocess
from postprocess import postprocess


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.declare_parameters('', [
            ('model_path', 'models/mobilenet_dummy.pt'),
            ('horizon', 20),
            ('speed', 1.0),
            ('steering_angle', 0.0),
        ])
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.horizon = self.get_parameter('horizon').value
        self.speed = self.get_parameter('speed').value
        self.steering = self.get_parameter('steering_angle').value
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_pub = self.create_publisher(Path, '/local_path', 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        self.latest_odom = None

    def odom_callback(self, msg):
        self.latest_odom = msg

    def scan_callback(self, msg):
        x = preprocess(msg)
        with torch.no_grad():
            y = self.model(x)
        path, cmd = postprocess(y, self.horizon, self.speed, self.steering)
        path.header.stamp = msg.header.stamp
        path.header.frame_id = 'map'
        self.path_pub.publish(path)
        cmd.header.stamp = msg.header.stamp
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
