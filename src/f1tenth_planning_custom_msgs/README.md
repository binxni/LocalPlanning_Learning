# F1TENTH Planning Custom Messages

Custom ROS2 message definitions for F1TENTH autonomous racing system. Provides specialized data structures for path planning and control.

## Message Types

### PathPoint.msg
```
# Single waypoint with velocity information
geometry_msgs/Point position     # x, y, z coordinates
float64 velocity                 # target speed at this point
float64 curvature               # path curvature (optional)
```

### PathPointArray.msg
```
# Array of path points for trajectory representation
std_msgs/Header header          # timestamp and frame info
PathPoint[] points              # sequence of waypoints
```

### PathWithVelocity.msg
```
# Complete path with velocity profile for racing
std_msgs/Header header          # timestamp and frame info  
PathPoint[] path_points         # waypoints with target speeds
float64 total_length           # total path length
int32 closest_point_index      # index of closest point to robot
```

### WallCollision.msg
```
# Wall collision detection information
std_msgs/Header header          # timestamp and frame info
bool collision_detected         # true if collision imminent
geometry_msgs/Point collision_point  # predicted collision location
float64 time_to_collision      # estimated time until impact
float64 collision_distance     # distance to collision point
```

## Usage in System

### Lattice Planner
**Publishes:**
- `PathWithVelocity` on `/planned_path` - Generated trajectories with speed profiles

### Path Follower  
**Subscribes:**
- `PathWithVelocity` from `/planned_path` - Executes planned trajectories

### Safety Monitor
**Publishes:**
- `WallCollision` on `/wall_collision` - Collision warnings

## Building

```bash
# Build messages (from workspace root)
colcon build --packages-select f1tenth_planning_custom_msgs
source install/setup.bash
```

**Note**: This package must be built before other packages that depend on these messages.

## Integration

These messages enable high-level coordination between planning and control components:

```
Planner → PathWithVelocity → Path Follower → AckermannDrive → Hardware
    ↓
Safety Monitor → WallCollision → Emergency Brake
```

## Message Design

**PathWithVelocity** is the core message type that:
- Carries complete trajectory information
- Includes velocity profiles for each waypoint  
- Provides spatial and temporal path data
- Enables smooth path following and speed control

**WallCollision** provides safety-critical information for:
- Real-time collision avoidance
- Emergency braking decisions
- Path replanning triggers
- Safety monitor alerts

Built for high-frequency, low-latency communication in F1TENTH racing applications.