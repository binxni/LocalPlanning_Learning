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

## Train

Prepare a dataset saved as an `.npz` file containing arrays `laser`,
`global_wp` and `local_wp`. Each sample should stack one 1081-element
laser scan with a corresponding global waypoint track to form a
two-channel input. `local_wp` should contain the target 20 waypoints
with `(x, y, yaw, v)` for each step.

Train the MobileNetV2-based planner and export a TorchScript model with:

```bash
python scripts/train_mobilenet.py path/to/data.npz --epochs 5 --out models/mobilenet_trained.pt
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
