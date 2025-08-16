import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2


class WaypointDataset(Dataset):
    """Dataset for local planner training.

    Expects an ``.npz`` file with arrays ``laser``, ``global_wp`` and
    ``local_wp``. ``laser`` and ``global_wp`` should have shape ``(N, 1081)``
    and will be stacked to form the 2-channel input. ``local_wp`` should have
    shape ``(N, 20, 4)`` representing ``x, y, yaw`` and ``v`` for each step in
    the horizon.
    """

    def __init__(self, path: str) -> None:
        data = np.load(path)
        laser = data["laser"].astype(np.float32)
        global_wp = data["global_wp"].astype(np.float32)
        inputs = np.stack((laser, global_wp), axis=1)
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(data["local_wp"].astype(np.float32))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class PlannerMobileNet(nn.Module):
    """MobileNetV2 backbone adjusted for 1D planner inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = mobilenet_v2(weights=None)
        # Replace first convolution: 2 input channels, 1x1 kernel, stride 1
        self.backbone.features[0][0] = nn.Conv2d(
            2, 32, kernel_size=1, stride=1, padding=0, bias=False
        )
        # Replace classifier to output 80 values (20 waypoints * 4 features)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2, 1081)
        x = x.unsqueeze(-1)  # -> (B, 2, 1081, 1)
        x = self.backbone(x)
        return x.view(-1, 20, 4)


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        print(f"epoch {epoch+1}: loss={running/len(loader.dataset):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 local planner")
    parser.add_argument("data", help="Path to training data .npz file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/mobilenet_trained.pt",
                        help="Output TorchScript model path")
    args = parser.parse_args()

    dataset = WaypointDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = PlannerMobileNet()
    train(model, loader, epochs=args.epochs, lr=args.lr)
    example = torch.zeros(1, 2, 1081)
    traced = torch.jit.trace(model.cpu(), example)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    traced.save(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
