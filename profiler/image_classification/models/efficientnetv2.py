"""
EfficientNet V1 and V2 implementation reffered to
https://github.com/google/automl/blob/master/efficientnetv2
"""

"""
EfficientNet V1 and V2 model.
[1] Mingxing Tan, Quoc V. Le
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
    ICML'19, https://arxiv.org/abs/1905.11946
[2] Mingxing Tan, Quoc V. Le
    EfficientNetV2: Smaller Models and Faster Training.
    https://arxiv.org/abs/2104.00298
"""
import copy
from ctypes import Union
import functools
import math
from typing import Any, Callable, Optional, List, Sequence, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

import torchvision
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation


__all__ = [
    "EfficientNet",
    "efficientnetv2_s",
    "efficientnetv2_m",
    "efficientnetv2_l",
    "efficientnetv2_xl",
]


class ModelConfig:
    # Stores infomation of global model parameters
    def __init__(
        self,
        model_name: str,
        feature_size: int = 1280,
        stochastic_depth_prob: float = 0.2,
        conv_dropout: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.feature_size = feature_size
        self.stochastic_depth_prob = stochastic_depth_prob
        self.conv_dropout = conv_dropout
        self.dropout_rate = dropout_rate

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}"
            f", feature_size={self.feature_size}"
            f", stochastic_depth_prob={self.stochastic_depth_prob}"
            f", conv_dropout={self.conv_dropout}"
            f", dropout_rate={self.dropout_rate}"
            f")"
        )
        return s


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper and block type (0: MBConv, 1: FusedMBConv)
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        se_ratio: float,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.se_ratio = se_ratio
        self.block_type = 1 if se_ratio == 0 else 0

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f", se_ratio={self.se_ratio}"
            f", block_type={self.block_type}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return torchvision.models._utils._make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))



class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        dropout_rate: Optional[float],
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.use_dropout = dropout_rate and cnf.expand_ratio > 1
        self.dropout_rate = dropout_rate

        self._build(cnf, activation_layer, norm_layer, se_layer)

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels


    def _build(self,
        cnf: MBConvConfig,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module],
    ) -> None:
        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        if self.use_dropout:
            layers.append(nn.Dropout(self.dropout_rate))

        if cnf.se_ratio is not None and 0 < cnf.se_ratio <= 1:
            num_reduced_channels = max(
                1, int(cnf.input_channels * cnf.se_ratio))
            layers.append(
                se_layer(
                    expanded_channels,
                    num_reduced_channels,
                    activation=activation_layer,
                )
            )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = result + input
        return result


class FusedMBConv(MBConv):
    def _build(self,
        cnf: MBConvConfig,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module],
    ) -> None:
        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        if self.use_dropout:
            layers.append(nn.Dropout(self.dropout_rate))

        if cnf.se_ratio is not None and 0 < cnf.se_ratio <= 1:
            num_reduced_channels = max(
                1, int(cnf.input_channels * cnf.se_ratio))
            layers.append(
                se_layer(
                    expanded_channels,
                    num_reduced_channels,
                    activation=activation_layer,
                )
            )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1 if cnf.expand_ratio != 1 else cnf.kernel,
                stride=1 if cnf.expand_ratio != 1 else cnf.stride,
                norm_layer=norm_layer,
                activation_layer=None if cnf.expand_ratio != 1 else activation_layer,
            )
        )

        self.block = nn.Sequential(*layers)



class EfficientNet(nn.Module):
    """A class implements torch.nn.Module.

        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self,
                 model_cnf: ModelConfig,
                 inverted_residual_setting: List[MBConvConfig],
                 num_classes: int = 1000,
                 image_channels: int = 3,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwars: Any,
    ) -> None:
        """EfficientNet main class

        Args:
            model_cnf (ModelConfig): Global model parameters
            inverted_residual_setting (List[MBConvConfig]): Network structure
            num_classes (int): Number of classes
            image_channels (int): Number of image channels (default: 3)
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        layers: List[nn.Module] = []

        if activation_layer is None:
            activation_layer = functools.partial(nn.SiLU, inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # building stem layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                image_channels,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []

            block = {0: MBConv, 1: FusedMBConv}[cnf.block_type]

            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = model_cnf.stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(
                    block(
                        block_cnf, sd_prob, model_cnf.conv_dropout, activation_layer, norm_layer
                    )
                )
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building head layer
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = model_cnf.feature_size
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(model_cnf.dropout_rate, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / np.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """Implementation of forward().

        Args:
            inputs: input tensors.

        Returns:
            output tensors.
        """
        outputs = self.features(inputs)

        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs) # torch.flatten(outputs, 1)

        outputs = self.classifier(outputs)

        return outputs


def _get_block_configs(cnf: Union[str, MBConvConfig], width_mult: float, depth_mult: float) -> List[MBConvConfig]:
    if isinstance(cnf, MBConvConfig):
        return cnf
    bneck_conf = functools.partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    if cnf == 's':
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 24, 24, 2, 0),
            bneck_conf(4, 3, 2, 24, 48, 4, 0),
            bneck_conf(4, 3, 2, 48, 64, 4, 0),
            bneck_conf(4, 3, 2, 64, 128, 6, 0.25),
            bneck_conf(6, 3, 1, 128, 160, 9, 0.25),
            bneck_conf(6, 3, 2, 160, 256, 15, 0.25),
        ]
    elif cnf == 'm':
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 24, 24, 3, 0),
            bneck_conf(4, 3, 2, 24, 48, 5, 0),
            bneck_conf(4, 3, 2, 48, 80, 5, 0),
            bneck_conf(4, 3, 2, 80, 160, 7, 0.25),
            bneck_conf(6, 3, 1, 160, 176, 14, 0.25),
            bneck_conf(6, 3, 2, 176, 304, 18, 0.25),
            bneck_conf(6, 3, 1, 304, 512, 5, 0.25),
        ]
    elif cnf == 'l':
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 32, 4, 0),
            bneck_conf(4, 3, 2, 32, 64, 7, 0),
            bneck_conf(4, 3, 2, 64, 96, 7, 0),
            bneck_conf(4, 3, 2, 96, 192, 10, 0.25),
            bneck_conf(6, 3, 1, 192, 224, 19, 0.25),
            bneck_conf(6, 3, 2, 224, 384, 25, 0.25),
            bneck_conf(6, 3, 1, 384, 640, 7, 0.25),
        ]
    elif cnf == 'xl':
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 32, 4, 0),
            bneck_conf(4, 3, 2, 32, 64, 8, 0),
            bneck_conf(4, 3, 2, 64, 96, 8, 0),
            bneck_conf(4, 3, 2, 96, 192, 16, 0.25),
            bneck_conf(6, 3, 1, 192, 256, 24, 0.25),
            bneck_conf(6, 3, 2, 256, 512, 32, 0.25),
            bneck_conf(6, 3, 1, 512, 640, 8, 0.25),
        ]
    elif cnf == 'base':
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1, 0),
            bneck_conf(4, 3, 2, 16, 32, 2, 0),
            bneck_conf(4, 3, 2, 32, 48, 2, 0),
            bneck_conf(4, 3, 2, 48, 96, 3, 0.25),
            bneck_conf(6, 3, 1, 96, 112, 5, 0.25),
            bneck_conf(6, 3, 2, 112, 192, 8, 0.25),
        ]
    else:
        raise ValueError(f"Not found block configs type '{cnf}'")
    return inverted_residual_setting


def _efficientnetv2(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    cnf_kind: Union[str, MBConvConfig] = 'base',
    model_cnf: Optional[ModelConfig] = None,
    **kwargs: Any,
) -> EfficientNet:
    inverted_residual_setting = _get_block_configs(cnf_kind, width_mult, depth_mult)
    if model_cnf is None:
        model_cnf = ModelConfig(arch, dropout_rate=dropout)
    model = EfficientNet(model_cnf, inverted_residual_setting, **kwargs)
    return model


def efficientnetv2_s(**kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNetV2-S architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet (No suppot yet)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnetv2("efficientnetv2_s", 1.0, 1.0, 0.2, 's', **kwargs)


def efficientnetv2_m(**kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNetV2-M architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet (No suppot yet)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnetv2("efficientnetv2_m", 1.0, 1.0, 0.3, 'm', **kwargs)


def efficientnetv2_l(**kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNetV2-L architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet (No suppot yet)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnetv2("efficientnetv2_l", 1.0, 1.0, 0.4, 'l', **kwargs)


def efficientnetv2_xl(**kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNetV2-XL architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet (No suppot yet)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnetv2("efficientnetv2_xl", 1.0, 1.0, 0.4, 'xl', **kwargs)


