"""1D adaptation of MobileNetV2 for waypoint prediction."""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import nn


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure that all layers have a channel number that is divisible by the divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class ConvBNAct1D(nn.Sequential):
    """Convolution + BatchNorm + Activation block for 1D inputs."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual1D(nn.Module):
    """Inverted residual block for 1D inputs."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1D(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend(
            [
                ConvBNAct1D(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone1D(nn.Module):
    """MobileNetV2 feature extractor using 1D convolutions."""

    def __init__(
        self,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        block = InvertedResidual1D
        input_channel = 32
        last_channel = 1280
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features: List[nn.Module] = [
            ConvBNAct1D(2, input_channel, stride=2, norm_layer=norm_layer)
        ]
        # t, c, n, s
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride_value = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride_value,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel

        features.append(
            ConvBNAct1D(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        )
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x


class PlannerMobileNet1D(nn.Module):
    """Full 1D MobileNetV2 model for predicting 20 waypoints."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2Backbone1D()
        self.classifier = nn.Linear(self.backbone.last_channel, 80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.view(-1, 20, 4)

