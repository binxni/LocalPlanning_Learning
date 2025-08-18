import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    """Dataset of random images and labels for demonstration purposes."""

    def __init__(self, length: int = 1000) -> None:
        self.length = length

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, idx: int):  # type: ignore[override]
        image = torch.randn(1, 28, 28)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


def load_data(batch_size: int = 64) -> DataLoader:
    dataset = RandomDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleNet(nn.Module):
    """Simple fully connected network for classification."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_data()
    model = SimpleNet().to(device)
    train(model, train_loader, device)
    torch.save(model.state_dict(), "random_model.pth")


if __name__ == "__main__":
    main()
