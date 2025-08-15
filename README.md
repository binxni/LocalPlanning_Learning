# Local Planner Template

Minimal ROS 2 Humble local planner using a TorchScript MobileNet model.

## Dummy model

Generate a placeholder model producing zeros with:

```bash
python - <<'PY'
import torch
class Dummy(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(20,4)
torch.jit.trace(Dummy(), torch.zeros(1)).save('models/mobilenet_dummy.pt')
PY
```

## Build

```bash
colcon build --packages-select local_planner
source install/setup.bash
```

## Run

```bash
ros2 launch local_planner planner_launch.py
```
