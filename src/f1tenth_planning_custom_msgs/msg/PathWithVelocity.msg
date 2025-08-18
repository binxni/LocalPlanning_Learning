# Path with velocity information for each point
std_msgs/Header header

# Array of path points with velocity
PathPoint[] points

# Optional: Maximum velocity for this path segment
float64 max_velocity

# Optional: Path ID for tracking
uint32 path_id