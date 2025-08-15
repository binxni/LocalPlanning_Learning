from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped


def postprocess(output, horizon, speed, steering):
    path = Path()
    cmd = AckermannDriveStamped()
    cmd.drive.speed = float(speed)
    cmd.drive.steering_angle = float(steering)
    for i in range(min(horizon, output.shape[0])):
        pose = PoseStamped()
        pose.pose.position.x = float(output[i, 0])
        pose.pose.position.y = float(output[i, 1])
        path.poses.append(pose)
    return path, cmd
