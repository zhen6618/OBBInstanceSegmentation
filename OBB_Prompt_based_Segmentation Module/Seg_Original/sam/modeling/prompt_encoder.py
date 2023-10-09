# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d
import math


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 3  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        for point_embedding in point_embeddings:
            point_embedding.weight.data.fill_(0)

        self.point_embeddings = nn.ModuleList(point_embeddings)
        # self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # indicating no-mask flag

        'Gaussian standard deviation'


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[4].weight  # foreground
        point_embedding[labels == 1] += self.point_embeddings[5].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""

        'x1, y1(0-1), x2, y2(0-1), angle(0-180)'
        coords = boxes[..., :4].reshape(-1, 2, 2) + 0.5  # shift to center of pixel
        corner_embedding = self.pe_layer.forward_with_coords(coords)  # (1, 2, 256)
        corner_embedding[:, 0, :] += self.point_embeddings[0].weight  # representing top-left corner
        corner_embedding[:, 1, :] += self.point_embeddings[1].weight  # representing bottom-right corner

        angles = boxes[..., 4].reshape(-1, 1)  # (n, 1)
        angles = angles / 180 *math.pi  # [0, 180) to [0, pi)
        angle_embedding = torch.concat([torch.sin(angles), torch.cos(angles)], dim=1)  # (n, 2) (-1, 1)

        angle_embedding = self.pe_layer.forward_with_angles(angle_embedding)  # (n, 256)
        angle_embedding += self.point_embeddings[2].weight.reshape(-1)
        angle_embedding = angle_embedding.unsqueeze(1)  # (1, 1, 256)

        output = torch.cat([corner_embedding, angle_embedding], dim=1)  # (1, 3, 256)

        return output

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())  # (1, 0, 256s)
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)  # boxes: Tensor(1, 5) to (1, 3, 256)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)  # (1, 3, 256)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        # print('scale: ', scale)  # default； None
        # if scale is None or scale <= 0.0:
        #     scale = 1.0

        torch.random.manual_seed(0)
        self.register_buffer(
            "positional_coords_encoding_gaussian_matrix",
            torch.randn((2, num_pos_feats)),  # torch.randn
        )

        torch.random.manual_seed(0)
        self.register_buffer(
            "positional_angles_encoding_gaussian_matrix",
            torch.randn((2, num_pos_feats)),
        )

        'standard deviation'
        self.sd_interval = 100
        self.coords_gaussian_sd = nn.Embedding(1, 1)
        self.coords_gaussian_sd.weight.data.fill_(4)  # learning: 4 to 0

        self.angles_gaussian_sd = nn.Embedding(1, 1)
        self.angles_gaussian_sd.weight.data.fill_(4)  # learning: 4 to 2.5


    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0, 1)."""

        'standard deviation'
        coords_gaussian_scale = self.coords_gaussian_sd.weight.view(-1)
        coords_gaussian_scale = (coords_gaussian_scale - 4) * self.sd_interval + 4
        coords_gaussian_scale = 2 ** coords_gaussian_scale   # 2**0 ~ 2**8

        coords = coords @ self.positional_coords_encoding_gaussian_matrix  * coords_gaussian_scale  # (1, 2, 2) @ (2, 128) = (1, 2, 128)
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_coords_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5  # grid center
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # (64, 64, 256)
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor
    ) -> torch.Tensor:

        coords = coords_input.clone()
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def forward_with_angles(
        self, angles_input: torch.Tensor,
    ) -> torch.Tensor:
        """Positionally encode angles that are normalized to [-1, 1]."""

        'standard deviation'
        angles_gaussian_scale = self.angles_gaussian_sd.weight.view(-1)
        angles_gaussian_scale = (angles_gaussian_scale - 4) * self.sd_interval + 4
        angles_gaussian_scale = 2 ** angles_gaussian_scale   # 2**0 ~ 2**8

        angles = angles_input.clone()
        angles = angles.to(torch.float)

        angles = angles @ self.positional_angles_encoding_gaussian_matrix * angles_gaussian_scale  # (1, 128)
        coords = 2 * np.pi * angles

        output = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # (1, 256)

        return output
