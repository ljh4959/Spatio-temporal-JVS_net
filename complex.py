import torch
import torch.nn as nn
from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange
from typing import Tuple

class Conv2plus1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], padding: Tuple[int, int, int], stride: Tuple[int, int, int] = (1, 1, 1), **kwargs
    ) -> None:
        super().__init__()
        mid_channels = max(in_channels, out_channels)
        
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        temporal_stride = (stride[0], 1, 1)
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_padding = (0, padding[1], padding[2])
        spatial_stride = (1, stride[1], stride[2])
        mid_channels = max(in_channels, out_channels)

        self.conv = nn.Sequential(
            ComplexConv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
                stride=temporal_stride,
                **kwargs,
            ),
            nn.ReLU(inplace=True),
            ComplexConv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                stride=spatial_stride,
                **kwargs,
            ),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")

        image = self.conv(image)

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")

        return image
