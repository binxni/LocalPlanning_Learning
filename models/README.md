# Model Architecture

This package provides a 1D adaptation of the MobileNetV2 architecture for waypoint prediction. The model processes a two-channel sequence consisting of a laser scan and a global plan and outputs 16 future waypoints with position, heading and velocity for each step.

## Backbone

The `MobileNetV2Backbone1D` module mirrors the original MobileNetV2 design using 1D convolutions:

- Initial `ConvBNAct1D` layer to reduce resolution.
- Series of inverted residual blocks with expansion ratios and strides `[1,16,1,1]`, `[6,24,2,2]`, `[6,32,3,2]`, `[6,64,4,2]`, `[6,96,3,1]`, `[6,160,3,2]`, `[6,320,1,1]`.
- Final `ConvBNAct1D` layer followed by global average pooling.

## Planner Head

`PlannerMobileNet1D` adds three independent fully connected heads for predicting
`x/y` offsets, `yaw` angles and `velocity` values. Each head uses a hidden layer with ReLU activation and produces tensors that are reshaped and concatenated to yield the final `(16, 4)` waypoint array.

## Usage

Instantiate the planner and pass a tensor shaped `(batch, 2, length)`:

```python
import torch
from models import PlannerMobileNet1D

model = PlannerMobileNet1D()
input_tensor = torch.randn(1, 2, 1081)  # example input
output = model(input_tensor)
print(output.shape)  # (1, 16, 4)
```

The architecture is lightweight and suitable for exporting to TorchScript for deployment within the local planner.

