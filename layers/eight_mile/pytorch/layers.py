import copy
import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import torch.autograd
import glob
from eight_mile.utils import listify, Offsets, is_sequence
from eight_mile.utils import transition_mask as transition_mask_np

MASK_FALSE = False
logger = logging.getLogger("mead.layers")


def sequence_mask(lengths: torch.Tensor, max_len: int = -1) -> torch.Tensor:
    """Generate a sequence mask of shape `BxT` based on the given lengths

    :param lengths: A `B` tensor containing the lengths of each example
    :param max_len: The maximum width (length) allowed in this mask (default to None)
    :return: A mask
    """
    lens = lengths.cpu()
    if max_len < 0:
        max_len_v = torch.max(lens)
    else:
        max_len_v = max_len
    # 1 x T
    row = torch.arange(0, max_len_v).type_as(lens).view(1, -1)
    # B x 1
    col = lens.view(-1, 1)
    # Broadcast to B x T, compares increasing number to max
    mask = row < col
    return mask


def vec_log_sum_exp(vec: torch.Tensor, dim: int) -> torch.Tensor:
    """Vectorized version of log-sum-exp

    :param vec: Vector
    :param dim: What dimension to operate on
    :return:
    """
    max_scores, idx = torch.max(vec, dim, keepdim=True)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), dim, keepdim=True))


def unsort_batch(batch: torch.Tensor, perm_idx: torch.Tensor) -> torch.Tensor:
    """Undo the sort on a batch of tensors done for packing the data in the RNN.

    :param batch: The batch of data batch first `[B, ...]`
    :param perm_idx: The permutation index returned from the torch.sort.

    :returns: The batch in the original order.
    """
    # Add ones to the shape of the perm_idx until it can broadcast to the batch
    perm_idx = perm_idx.to(batch.device)
    diff = len(batch.shape) - len(perm_idx.shape)
    extra_dims = [1] * diff
    perm_idx = perm_idx.view([-1] + extra_dims)
    return torch.scatter(torch.zeros_like(batch), 0, perm_idx.expand_as(batch), batch)


def infer_lengths(tensor, dim=1):
    """Infer the lengths of an input based on the idea the Offsets.PAD was used as the padding token.

    :param tensor: The data to infer the length of, should be either [B, T] or [T, B]
    :param dim: The dimension which contains the sequential signal

    :returns: A Tensor of shape `[B]` that has the lengths for example item in the batch
    """
    if len(tensor.shape) != 2:
        raise ValueError(f"infer_lengths only works with tensors wit two dims right now, got {len(tensor.shape)}")
    offsets = torch.arange(1, tensor.shape[dim] + 1, device=tensor.device, dtype=tensor.dtype).unsqueeze(1 - dim)
    non_pad_loc = (tensor != Offsets.PAD).to(tensor.dtype)
    return torch.argmax(non_pad_loc * offsets, dim=dim) + 1


def tensor_and_lengths(inputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return either the unpacked inputs (2), or a `Tuple` of the input with None

    TODO: this function should probably be changed to always return the lengths second.
    To do this, we just need a sentinel value, e.g. <PAD> (0).  The problem with doing this is
    that it might be possible to generate <PAD> in the middle of the tensor which would make that
    length invalid.

    :param inputs: Either a sequence of the `(tensor, length)` or just the `tensor`
    :return: A `Tuple` of `(tensor, length)` or `(tensor, None)`
    """
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None

    return in_tensor, lengths


class VariationalDropout(nn.Module):
    """Inverted dropout that applies the same mask at each time step."""

    def __init__(self, pdrop: float = 0.5, batch_first: bool = False):
        """Variational Dropout

        :param pdrop: the percentage to drop
        """
        super().__init__()
        self.pdrop = pdrop
        self.batch_first = batch_first

    def extra_repr(self):
        return "p=%.1f" % self.pdrop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input
        # Create a mask that covers a single time step
        if self.batch_first:
            dim0 = input.size(0)
            dim1 = 1
        else:
            dim0 = 1
            dim1 = input.size(1)
        mask = torch.zeros(dim0, dim1, input.size(2)).bernoulli_(1 - self.pdrop).to(input.device)
        mask = mask / self.pdrop
        # Broadcast the mask over the sequence
        return mask * input


class SequenceLoss(nn.Module):
    """Computes the loss over a sequence"""

    def __init__(self, LossFn: nn.Module = nn.NLLLoss, avg: str = "token"):
        """A class that applies a Loss function to sequence via the folding trick.

        :param LossFn: A loss function to apply (defaults to `nn.NLLLoss`)
        :param avg: A divisor to apply, valid values are `token` and `batch`
        """
        super().__init__()
        self.avg = avg
        if avg == "token":
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction="mean")
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction="sum")
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate some loss over a sequence.
        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.
        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)

    def extra_repr(self):
        return f"reduction={self.avg}"


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, ignore_index=0, reduction="none"):
        """Use Label smoothing from `Szegedy et. al., 2015`_ to temper model confidence.

        Implements add-gamma smoothing where the probability mass of the gold label distribution
        is smoothed across classes.

        This implementation is based on `OpenNMT-py`_ but has been adapted to not require the
        vocabulary size up front.

        .. _Szegedy et. al., 2015: https://arxiv.org/abs/1512.00567
        .. _OpenNMY-py: https://github.com/OpenNMT/OpenNMT-py/blob/938a4f561b07f4d468647823fab761cfb51f21da/onmt/utils/loss.py#L194
        """
        if not (0.0 < label_smoothing <= 1.0):
            raise ValueError(f"`label_smoothing` must be between 0.0 and 1.0, got {label_smoothing}")
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing
        self.reduction = reduction if reduction != "mean" else "batchmean"

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param output: The model outputs, [B, V]
        :param target: The target labels, [B]
        """
        B, V = output.size()
        smoothed = torch.full((B, V), self.label_smoothing / (V - 2))
        smoothed[:, self.ignore_index] = 0
        smoothed = torch.scatter(smoothed, 1, target.unsqueeze(1), self.confidence)
        smoothed = smoothed.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        return F.kl_div(output, smoothed, reduction=self.reduction)

    def extra_repr(self):
        return f"label_smoothing={self.label_smoothing}"

class MeanPool1D(nn.Module):
    """Do a mean pool while accounting for the length of a sequence
    """

    def __init__(self, outsz, batch_first=True):
        """Set up pooling module

        :param outsz: The output dim, for dowstream access
        :param batch_first: Is this module batch first or time first?
        """
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz
        self.requires_length = True

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply mean pooling on the valid inputs

        :param inputs: A tuple of `(input, lengths)`
        :return: Pooled output
        """
        tensor, lengths = tensor_and_lengths(inputs)
        # Regardless of whether the input is `[B, T, H]` or `[T, B, H]` the shape after
        # the sum is `[B, H]` so the lengths (of shape `[B]`) should be unsqueezed to
        # `[B, 1]` in order to broadcast
        return torch.sum(tensor, self.reduction_dim, keepdim=False) / torch.unsqueeze(lengths, -1).to(tensor.dtype).to(
            tensor.device
        )

    def extra_repr(self):
        return f"batch_first={self.batch_first}"


class MaxPool1D(nn.Module):
    """Do a max-pooling operation with or without a length given
    """

    def __init__(self, outsz, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """If we are given a tuple as input, we will use the length, otherwise we will do an operation without masking

        :param inputs: either a tuple of `(input, lengths)` or a tensor `input`
        :return: A pooled tensor
        """
        tensor, lengths = tensor_and_lengths(inputs)
        if lengths is not None:
            # If tensor = `[B, T, H]`
            #    mask = `[B, T, 1]`
            # If tensor = `[T, B, H]`
            #    mask = `[T, B, 1]`
            # So it will mask all the values in H past the right length
            mask = sequence_mask(lengths).to(tensor.device)
            mask = mask if self.batch_first else bth2tbh(mask)
            # Fill masked with very negative so it never gets selected
            tensor = tensor.masked_fill(mask.unsqueeze(-1) == MASK_FALSE, -1e4)
        dmax, _ = torch.max(tensor, self.reduction_dim, keepdim=False)
        return dmax

    def extra_repr(self) -> str:
        return f"batch_first={self.batch_first}"


# Torch only added this module in 1.4.0, shim
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


def get_activation(name: str = "relu") -> nn.Module:
    """Get back an `nn.Module` by string name of the activation operator

    :param name: A string name of the operation
    :return: A module associated with that string
    """
    if name is None or name == "ident":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return GeLU()
    if name == "hardtanh":
        return nn.Hardtanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "prelu":
        return nn.PReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "log_sigmoid":
        return nn.LogSigmoid()
    if name == "log_softmax":
        return nn.LogSoftmax(dim=-1)
    if name == "softmax":
        return nn.Softmax(dim=-1)
    return nn.ReLU()


def _cat_dir(h: torch.Tensor) -> torch.Tensor:
    """Concat forward and backword state vectors.

    The shape of the hidden is `[#layers * #dirs, B, H]`. The docs say you can
    separate directions with `h.view(#l, #dirs, B, H)` with the forward dir being
    index 0 and backwards dir being 1.

    This means that before separating with the view the forward dir are the even
    indices in the first dim while the backwards dirs are the odd ones. Here we select
    the even and odd values and concatenate them

    :param h: The hidden shape as it comes back from PyTorch modules
    """
    return torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], dim=-1)


def concat_state_dirs(state):
    """Convert the bidirectional out of an RNN so the forward and backward values are a single vector."""
    if isinstance(state, tuple):
        return tuple(_cat_dir(h) for h in state)
    return _cat_dir(state)


class Conv1DSame(nn.Module):
    """Perform a 1D convolution with output size same as input size

    To make this operation work as expected, we cannot just use `padding=kernel_size//2` inside
    of the convolution operation.  Instead, we zeropad the input using the `ConstantPad1d` module

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, groups: int = 1):
        """Create a 1D conv to produce the same output size as input

        :param in_channels: The number of input feature maps
        :param out_channels: The number of output feature maps
        :param kernel_size: The kernel size
        :param bias: Is bias on?
        :param groups: Number of conv groups

        """
        super().__init__()
        end_pad = kernel_size // 2
        start_pad = end_pad - 1 if kernel_size % 2 == 0 else end_pad
        self.conv = nn.Sequential(
            nn.ConstantPad1d((start_pad, end_pad), 0.),
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, groups=groups)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do convolution1d on an input tensor, `[B, C, T]`

        :param x: The input tensor of shape `[B, C, T]`
        :return: The output tensor of shape `[B, H, T]`
        """
        return self.conv(x)


class ConvEncoder(nn.Module):
    """1D Convolutional layer encoder with given activation function, optional dropout

    This module takes in a temporal signal of either shape `[B, C, T]` or `[B, T, C]`, depending on the constructor
    and produces an output signal of the same orientation (`[B, H, T]` or `[B, T, H]`, respectively).  We default
    to `[B, T, H]` orientation to make it more convenient for typical layout, but this requires transposing the last
    2 dims before and after the convolution operation.

    """

    def __init__(self, insz: int, outsz: int, filtsz: int, pdrop: float = 0.0, activation: str = "relu", bias: bool = True, groups: int = 1, hidden_last=True):

        """Construct the encoder with optional dropout, given activation, and orientation

        :param insz: The number of input feature maps
        :param outsz: The number of output feature maps (or hidden size)
        :param filtsz: The kernel size
        :param pdrop: The amount of dropout to apply, this defaults to 0
        :param activation: The activation function by name, defaults to `relu`
        :param bias: Use bias?
        :param groups: How many conv groups. Defaults to 1
        :param hidden_last: PyTorch only! If `True` the orientatiation is `[B, T, H]`, o.w. `[B, H, T]` expected
        """
        super().__init__()
        self.output_dim = outsz
        conv = Conv1DSame(insz, outsz, filtsz, bias=bias, groups=groups)
        act = get_activation(activation)
        dropout = nn.Dropout(pdrop)

        if hidden_last:
            self.conv = nn.Sequential(BTH2BHT(), conv, act, dropout, BHT2BTH())
        else:
            self.conv = nn.Sequential(conv, act, dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class ConvEncoderStack(nn.Module):
    """Create a stack of convolutional encoders with residual connections between, using the `ConvEncoder` underneath

    This creates an encoder stack of convolutions, finally returning the last temporal output.  Each layer uses zero-padding
    which causes the output of the convolution at each layer to be the same length.

    As in the `ConvEncoder` we support input tensor shapes of `[B, C, T]` or `[B, T, C]` depending on the constructor
    initialization, and transpose underneath the input and output of the stack if the orientation is defaulted to
    `[B, T, C]`
    """

    def __init__(self, insz: int, outsz: int, filtsz: int, nlayers: int = 1, pdrop: float = 0.0, activation: str = "relu", bias: bool = True, groups: int = 1, hidden_last=True):
        """Construct the encoder stack

        :param insz: The input number of feature maps
        :param outsz: The output number of feature maps
        :param filtsz: The kernel size
        :param nlayers: The number of layers in the stack (defaults to a single layer)
        :param pdrop: The amount of dropout to apply (defaults to `0`)
        :param activation: The activation function to use as a string, defaults to `relu`
        :param bias: Use bias?
        :param groups: How many conv groups. Defaults to 1
        :param hidden_last: PyTorch only! If `True` the orientatiation is `[B, T, H]`, o.w. `[B, H, T]` expected
        """
        super().__init__()

        if hidden_last:
            first_layer = nn.Sequential(BTH2BHT(), ConvEncoder(insz, outsz, filtsz, pdrop, activation, bias, groups, hidden_last=False))
        else:
            first_layer = ConvEncoder(insz, outsz, filtsz, pdrop, activation, bias, groups, hidden_last=False)

        subsequent_layer = ResidualBlock(ConvEncoder(outsz, outsz, filtsz, pdrop, activation, bias, groups, hidden_last=False))

        self.layers = nn.ModuleList([first_layer] + [copy.deepcopy(subsequent_layer) for _ in range(nlayers - 1)])
        if hidden_last:
            self.layers.append(BHT2BTH())
        self.output_dim = outsz

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply a stack of 1D convolutions with residual connections between them

        :param input: A tensor of shape `[B, T, C]` or `[B, C, T]` depending on value of `hidden_last`
        :return: A tensor of shape `[B, T, H]` or `[B, H, T]` depending on the value of `hidden_last`
        """
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


def bth2bht(t: torch.Tensor) -> torch.Tensor:
    """Transpose the 2nd and 3rd dim of a tensor"""
    return t.transpose(1, 2).contiguous()


class BTH2BHT(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[B, H, T]`
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return bth2bht(t)


def tbh2bht(t: torch.Tensor) -> torch.Tensor:
    """Permute the dimensions, first goes to third, second goes to first, last moves to second"""
    return t.permute(1, 2, 0).contiguous()


class TBH2BHT(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, H, T]`
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return tbh2bht(t)


def tbh2bth(t: torch.Tensor) -> torch.Tensor:
    """Transpose the first 2 dims"""
    return t.transpose(0, 1).contiguous()


class TBH2BTH(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, T, H]`
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return tbh2bth(t)


def bth2tbh(t: torch.Tensor) -> torch.Tensor:
    """Transpose the first 2 dims"""
    return t.transpose(0, 1).contiguous()


class BTH2TBH(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[T, B, H]`
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return bth2tbh(t)


def bht2bth(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2).contiguous()


class BHT2BTH(nn.Module):
    """Utility layer to convert from `[B, H, T]` to `[B, T, H]`
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return bht2bth(t)


class ParallelConv(nn.Module):
    """Layer of parallel convolutions with varying filter sizes followed by max over time pooling

    This module takes an input tensor of any orientation based on its constructor, and pools its
    output to shape `[B, H]`, where `H` is `outsz * len(filtsz)`
    """

    def __init__(self, insz: int, outsz: int, filtsz: List[int], activation: str = "relu", input_fmt: str = "bth"):
        """
        Constructor for a parallel convolution from any orientation tensor input

        :param insz: The number of input feature maps
        :param outsz: The number of output feature maps
        :param filtsz: The kernel size as a list of parallel filters to apply, e.g. `[3, 4, 5]`
        :param activation: An activation function by name to apply
        :param input_fmt: A string for the orientation.  Valid values are `bth` or `btc` meaning hidden units last,
        `bht` or `bct` meaning the temporal dim last or `tbh` or `tbc` meaning the hidden units last and the temporal dim
        first
        """
        super().__init__()
        self.requires_length = False
        convs = []
        outsz_filts = outsz
        self.input_fmt = input_fmt.lower()

        if type(outsz) == int:
            outsz_filts = len(filtsz) * [outsz]

        self.output_dim = sum(outsz_filts)
        for i, fsz in enumerate(filtsz):
            if fsz % 2 == 0:
                conv = Conv1DSame(insz, outsz_filts[i], fsz)
            else:
                pad = fsz // 2
                conv = nn.Conv1d(insz, outsz_filts[i], fsz, padding=pad)
            conv = nn.Sequential(
                conv,
                get_activation(activation)
            )
            convs.append(conv)
            # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)

    def transform_input(self, t: torch.Tensor) -> torch.Tensor:

        if self.input_fmt == "bth" or self.input_fmt == "btc":
            return bth2bht(t)
        elif self.input_fmt == "tbh" or self.input_fmt == "tbc":
            return tbh2bht(t)
        else:
            return t

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the input to `[B, C, T]` from any orientation and perform parallel 1D convs and max over time pool

        :param inputs: An input tensor of any format specified in the constructor
        :return: A `[B, H]` tensor representing the pooled outputs
        """
        mots = []
        input_bct = self.transform_input(inputs)

        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return mots  # self.conv_drop(mots)


class Highway(nn.Module):
    """Highway layer as defined in https://arxiv.org/abs/1505.00387

    """

    def __init__(self, input_size: int, **kwargs):
        """Highway layer constructor

        :param input_size: The input hidden size
        :param kwargs:
        """
        super().__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)
        self.output_dim = input_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Take a tensor in and produce the highway layer output

        :param input: Input tensor
        :return: output tensor
        """
        proj_result = torch.relu(self.proj(input))
        proj_gate = torch.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


def pytorch_linear(in_sz: int, out_sz: int, unif: float = 0, initializer: str = None, bias: bool = True):
    """Utility function that wraps a linear (AKA dense) layer creation, with options for weight init and bias"""
    l = nn.Linear(in_sz, out_sz, bias=bias)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(l.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(l.weight)
    else:
        nn.init.xavier_uniform_(l.weight)
    if bias:
        l.bias.data.zero_()
    return l


class StackedLSTMCell(nn.Module):
    """A stacked LSTM cells applied at a timestep
    """

    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=rnn_size, bias=False))
            input_size = rnn_size

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        """Apply a stack of LSTMs

        :param input: The input to the first LSTM `[B, H]`
        :param hidden: The previous `(h, c)` where `h=(h_0, h_1,..)`, `c=(c_0, c_1,..)`
        :return: The output and hidden `(h, c)` where `h=(h_0, h_1,..)`, `c=(c_0, c_1,..)`
        """
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs.append(h_i)
            cs.append(c_i)

        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return input, (hs, cs)


class StackedGRUCell(nn.Module):
    """A stacked GRU cells applied at a timestep
    """

    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size=input_size, hidden_size=rnn_size))
            input_size = rnn_size

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a stack of GRUs

        :param input: The input to the first LSTM `[B, H]`
        :param hidden: The previous `h` where `h=(h_0, h_1,..)`
        :return: The output and hidden `h` where `h=(h_0, h_1,..)`
        """
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            h_i = layer(input, (h_0[i]))
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            hs.append(h_i)

        hs = torch.stack(hs)

        return input, hs


class Dense(nn.Module):
    """Dense (Linear) layer with optional activation given

    This module is the equivalent of the tf.keras.layer.Dense, module with optional activations applied
    """

    def __init__(
        self,
        insz: int,
        outsz: int,
        activation: Optional[str] = None,
        unif: float = 0,
        initializer: Optional[str] = None,
    ):
        """Constructor for "dense" or "linear" layer, with optional activation applied

        :param insz: The number of hidden units in the input
        :param outsz: The number of hidden units in the output
        :param activation: The activation function by name, defaults to `None`, meaning no activation is applied
        :param unif: An optional initialization value which can set the linear weights.  If given, biases will init to 0
        :param initializer: An initialization scheme by string name: `ortho`, `kaiming` or `he`, `xavier` or `glorot`
        """
        super().__init__()
        self.layer = pytorch_linear(insz, outsz, unif, initializer)
        self.activation = get_activation(activation)
        self.output_dim = outsz

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run a linear projection over the input, followed by an optional activation given by constructor

        :param input: the input tensor
        :return: the transformed output
        """
        return self.activation(self.layer(input))


class WeightTieDense(nn.Module):
    """Do weight tying from the input parameter

    This module never copies the weight pointer, it lazily accesses to allow the tied variable to reset its parameters
    after initialization.  This is helpful for cases where we have LMs and are reloading them after they have been
    initially created
    """

    def __init__(self, tie: nn.Module, bias=False):
        super().__init__()
        self.tie = tie
        self.transform = self._get_transform(tie)
        if bias:
            bias = torch.nn.Parameter(torch.zeros(self.transform(self.weight.shape[0])))
        else:
            bias = None
        self.register_parameter("bias", bias)

    def _get_transform(self, tie: nn.Module):
        emb = getattr(tie, "embeddings", None)
        if emb is not None:
            return self._identity
        return self._transpose

    @property
    def weight(self):
        emb = getattr(self.tie, "embeddings", None)
        if emb is not None:
            return getattr(emb, "weight")
        return getattr(self.tie, "weight")

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _transpose(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.transform(self.weight), self.bias)


class ResidualBlock(nn.Module):
    """Create a residual block by wrapping an layer with a residual connection"""

    def __init__(self, layer: Optional[nn.Module] = None, **kwargs):
        """Wrap an layer with a residual connection

        :param layer: This layer will be applied to the input and added to the input
        :param kwargs:
        """
        super().__init__()
        self.layer = layer
        if self.layer is not None and hasattr(layer, "output_dim"):
            self.output_dim = layer.output_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply a residual block

        :param input: A tensor to use as input and to add to output
        :return: The residual connection output
        """
        return input + self.layer(input)


class SkipConnection(ResidualBlock):

    """Subclass of ResidualBlock(Dense) with an activation function given
    """

    def __init__(self, input_size: int, activation: str = "relu"):
        """Create a `SkipConnection`

        :param input_size: The input dimension size
        :param activation: A string activation name
        """
        super().__init__(None)
        self.layer = Dense(input_size, input_size, activation=activation)
        self.output_dim = input_size


def rnn_cell(insz: int, hsz: int, rnntype: str, nlayers: int, dropout: float):
    """This is a wrapper function around a stacked RNN cell

    :param insz: The input dimensions
    :param hsz: The hidden dimensions
    :param rnntype: An RNN type `gru` or `lstm`
    :param nlayers: The number of layers to stack
    :param dropout: The amount of dropout
    :return:
    """
    if rnntype == "gru":
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


def pytorch_lstm(
    insz: int,
    hsz: int,
    rnntype: str,
    nlayers: int,
    dropout: float,
    unif: float = 0,
    batch_first: bool = False,
    initializer: str = None,
) -> torch.nn.LSTM:
    """Wrapper around `torch.nn.LSTM`, mainly for weight initialization options

    :param insz: The input dimension
    :param hsz: The number of hidden units
    :param rnntype: A string description of the type of LSTM: `bi?lstm` or `lstm`
    :param nlayers: The number of layers
    :param dropout: How much dropout to apply
    :param unif: if uniform initialization, what range?
    :param batch_first: Should we do the RNN batch first or time first
    :param initializer: An optional string representing a style of initialization `ortho`, `he`/`kaiming`, `xavier`/`glorot`
    :return: An LSTM
    """
    if nlayers == 1:
        dropout = 0.0
    ndir = 2 if rnntype.startswith("b") else 1
    layer_hsz = hsz // ndir
    rnn = torch.nn.LSTM(
        insz, layer_hsz, nlayers, dropout=dropout, bidirectional=True if ndir > 1 else False, batch_first=batch_first
    )  # , bias=False)
    if initializer == "ortho":
        nn.init.orthogonal(rnn.weight_hh_l0)
        nn.init.orthogonal(rnn.weight_ih_l0)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(rnn.weight_hh_l0)
        nn.init.kaiming_uniform(rnn.weight_ih_l0)
    elif unif > 0:
        for weight in rnn.parameters():
            weight.data.uniform_(-unif, unif)
    else:
        nn.init.xavier_uniform_(rnn.weight_hh_l0)
        nn.init.xavier_uniform_(rnn.weight_ih_l0)

    return rnn


class LSTMEncoderBase(nn.Module):
    """The LSTM encoder is a base for a set of encoders producing various outputs.

    All LSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `LSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `LSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(
        self,
        insz: int,
        hsz: int,
        nlayers: int,
        pdrop: float = 0.0,
        requires_length: bool = True,
        batch_first: bool = False,
        unif: float = 0,
        initializer: str = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    # def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    #    tbc, lengths = tensor_and_lengths(inputs)
    #    packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
    #    output, hidden = self.rnn(packed)
    #    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
    #    return self.output_fn(output, hidden)

    # def output_fn(self, output, state):
    #    return output, self.extract_top_state(state)

    def extract_top_state(self, state: Tuple[torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        """Get a view of the top state of shape [B, H]`

        :param state:
        :return:
        """
        # Select the topmost state with -1 and the only direction is forward (select with 0)
        top = []
        for s in state:
            top.append(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0])

        return top


class LSTMEncoderSequence(LSTMEncoderBase):

    """LSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Take in a tuple of `(sequence, lengths)` and produce and output tensor of the last layer of LSTMs

        The value `S` here is defined as `max(lengths)`, `S <= T`

        :param inputs: sequence of shapes `[B, T, C]` or `[T, B, C]` and a lengths of shape `[B]`
        :return: A tensor of shape `[B, S, H]` or `[S, B, H]` depending on setting of `batch_first`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class LSTMEncoderWithState(nn.Module):

    """LSTM encoder producing the hidden state and the output, where the input doesnt require any padding

    PyTorch note: This type of encoder doesnt inherit the `LSTMEncoderWithState` base
    """

    def __init__(
        self,
        insz: int,
        hsz: int,
        nlayers: int,
        pdrop: float = 0.0,
        batch_first: bool = False,
        unif: float = 0,
        initializer: str = None,
        **kwargs,
    ):
        """
        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param batch_first: PyTorch only! do batch first or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN

        """
        super().__init__()
        self.requires_length = False
        self.requires_state = True
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def forward(self, input_and_prev_h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input_and_prev_h: The input at this timestep and the previous hidden unit or `None`
        :return: Raw `torch.nn.LSTM` output
        """
        inputs, hidden = input_and_prev_h
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden  ##concat_state_dirs(hidden)


class LSTMEncoderAll(LSTMEncoderBase):
    """LSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H]` or `[B, H, S]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, hidden


class LSTMEncoderHidden(LSTMEncoderBase):

    """LSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(hidden)[0]


# TODO: this module only exists in pytorch.  Do we eliminate it or put it in both?
class LSTMEncoderSequenceHiddenContext(LSTMEncoderBase):
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(hidden)


class BiLSTMEncoderBase(nn.Module):
    """BiLSTM encoder base for a set of encoders producing various outputs.

    All BiLSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiLSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiLSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(
        self,
        insz: int,
        hsz: int,
        nlayers: int,
        pdrop: float = 0.0,
        requires_length: bool = True,
        batch_first: bool = False,
        unif: float = 0,
        initializer: str = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz // 2, nlayers, dropout=pdrop, bidirectional=True, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state):
        # Select the topmost state with -1 and the only direction is forward (select with 0)
        return tuple(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0] for s in state)


# TODO: this module only exists in pytorch.  Do we eliminate it or put it in both?
class BiLSTMEncoderSequenceHiddenContext(BiLSTMEncoderBase):
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(concat_state_dirs(hidden))


class BiLSTMEncoderAll(BiLSTMEncoderBase):
    """BiLSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, concat_state_dirs(hidden)


class BiLSTMEncoderSequence(BiLSTMEncoderBase):

    """BiLSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.


    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Take in a tuple of `(sequence, lengths)` and produce and output tensor of the last layer of LSTMs

        The value `S` here is defined as `max(lengths)`, `S <= T`

        :param inputs: sequence of shapes `[B, T, C]` or `[T, B, C]` and a lengths of shape `[B]`
        :return: A tensor of shape `[B, S, H]` or `[S, B, H]` depending on setting of `batch_first`
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class BiLSTMEncoderHidden(BiLSTMEncoderBase):

    """BiLSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """
    def forward(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(concat_state_dirs(hidden))[0]

# TODO: Add this to TF or remove
class BiLSTMEncoderHiddenContext(BiLSTMEncoderBase):
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(concat_state_dirs(hidden))


class GRUEncoderBase(nn.Module):
    """The GRU encoder is a base for a set of encoders producing various outputs.

    All GRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `GRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `GRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(
            self,
            insz: int,
            hsz: int,
            nlayers: int,
            pdrop: float = 0.0,
            requires_length: bool = True,
            batch_first: bool = False,
            unif: float = 0,
            initializer: str = None,
            **kwargs,
    ):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
            nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) -> torch.Tensor:
        return state[-1]


class GRUEncoderSequence(GRUEncoderBase):

    """GRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Take in a tuple of the sequence tensor `[T, B, H]` or `[B, T, H]` and its length, produce output sequence

        :param inputs: A tuple of the sequence tensor and its length
        :return: A sequence tensor of shape `[T, B, H]` or `[B, T, H]`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class GRUEncoderAll(GRUEncoderBase):
    """GRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H]` or `[B, H, S]` , and a hidden tensor `[L, B, H]`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, hidden


class GRUEncoderHidden(GRUEncoderBase):

    """GRU encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(hidden)


class BiGRUEncoderBase(nn.Module):
    """BiGRU encoder base for a set of encoders producing various outputs.

    All BiGRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiGRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiGRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """
    def __init__(
            self,
            insz: int,
            hsz: int,
            nlayers: int,
            pdrop: float = 0.0,
            requires_length: bool = True,
            batch_first: bool = False,
            unif: float = 0,
            initializer: str = None,
            **kwargs,
    ):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiGRU (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiGRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """

        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz // 2, nlayers, dropout=pdrop, bidirectional=True, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) -> torch.Tensor:
        # Select the topmost state with -1 and the only direction is forward (select with 0)
        return state[-1]

# TODO: normalize across backends or remove
class BiGRUEncoderSequenceHiddenContext(BiGRUEncoderBase):
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(_cat_dir(hidden))


class BiGRUEncoderAll(BiGRUEncoderBase):
    """BiGRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and a hidden vector `[L, B, H]`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, _cat_dir(hidden)


class BiGRUEncoderSequence(BiGRUEncoderBase):

    """BiGRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Take in a tuple of `(sequence, lengths)` and produce and output tensor of the last layer of GRUs

        The value `S` here is defined as `max(lengths)`, `S <= T`

        :param inputs: sequence of shapes `[B, T, C]` or `[T, B, C]` and a lengths of shape `[B]`
        :return: A tensor of shape `[B, S, H]` or `[S, B, H]` depending on setting of `batch_first`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class BiGRUEncoderHidden(BiGRUEncoderBase):

    """GRU encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """
    def forward(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(_cat_dir(hidden))

class Reduction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        pass


class ConcatReduction(Reduction):
    def __init__(self, output_dims: List[int], axis=-1):
        super().__init__()
        self.axis = axis
        self.output_dim = sum(output_dims)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, self.axis)


class SumReduction(Reduction):
    def __init__(self, output_dims: List[int]):
        super().__init__()
        # We could actually project if we needed, or at least should validate
        self.output_dim = output_dims[0]

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return sum(inputs)


class SumLayerNormReduction(Reduction):

    def __init__(self, output_dims: List[int], layer_norm_eps: float = 1.0e-12):
        super().__init__()
        self.output_dim = output_dims[0]
        self.ln = nn.LayerNorm(self.output_dim, eps=layer_norm_eps)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        output = sum(inputs)
        return self.ln(output)


class EmbeddingsStack(nn.Module):
    def __init__(
        self,
        embeddings_dict: Dict[str, nn.Embedding],
        dropout_rate: float = 0.0,
        requires_length: bool = False,
        reduction: Optional[Union[str, nn.Module]] = 'concat',
        **kwargs,
    ):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings
        :param embeddings_dict: dictionary of each feature embedding
        :param dropout_rate: The dropout rate (0.0 means no dropout, 1.0 means complete)
        """

        super().__init__()

        self._keys: List[str] = []
        embeddings_list = []
        output_dims = []
        for k, embedding in embeddings_dict.items():

            embeddings_list.append(embedding)
            self._keys.append(k)
            output_dims += [embedding.get_dsz()]

        self.embeddings: nn.ModuleList = nn.ModuleList(embeddings_list)
        # TODO: should we make a registry of options?
        if isinstance(reduction, str):
            if reduction == 'sum':
                self.reduction = SumReduction(output_dims)
            elif reduction == 'sum-layer-norm':
                self.reduction = SumLayerNormReduction(output_dims, layer_norm_eps=kwargs.get('layer_norm_eps', 1.0e-12))
            else:
                self.reduction = ConcatReduction(output_dims)
        else:
            self.reduction = reduction
        self.dsz = self.reduction.output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.requires_length = requires_length

    def __getitem__(self, item: str) -> nn.Module:
        idx = self._keys.index(item)
        if idx < 0:
            raise Exception(f"Invalid item ({item})")
        return self.embeddings[idx]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings
        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        i = 0
        for embedding in self.embeddings:
            k = self._keys[i]
            x = inputs[k]
            # Its a hair faster to do this than using isinstance
            if x.__class__ == tuple:
                embeddings_out = embedding(*x)
            else:
                embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
            i += 1
        word_embeddings = self.reduction(all_embeddings_out)
        return self.dropout(word_embeddings)

    def keys(self):
        return self._keys

    @property
    def output_dim(self):
        return self.dsz

    def items(self):
        for k, v in zip(self.keys(), self.embeddings):
            yield k, v


class DenseStack(nn.Module):
    """A stack of one or more hidden layers
    """

    def __init__(
        self,
        insz: int,
        hsz: Union[int, List[int]],
        activation: Union[str, List[str]] = "relu",
        pdrop_value: float = 0.5,
        init=None,
        skip_connect=False,
        layer_norm=False,
        **kwargs,
    ):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param insz: The number of input units
        :param hsz: The number of hidden units
        :param activation: The name of the activation function to use
        :param pdrop_value: The dropout probability
        :param init: The initializer
        :param skip_connect: whether use skip connection when insz is equal to outsz for a layer
        :param layer_norm: whether use layer norm in each layer

        """
        super().__init__()
        hszs = listify(hsz)
        self.output_dim = hsz[-1]
        activations = listify(activation)
        if len(activations) == 1:
            activations = activations * len(hszs)
        if len(activations) != len(hszs):
            raise ValueError("Number of activations must match number of hidden sizes in a stack!")
        current = insz
        layer_stack = []
        if layer_norm:
            layer_norm_eps = kwargs.get('layer_norm_eps', 1e-6)
        for hsz, activation in zip(hszs, activations):
            if skip_connect and current == hsz:
                layer = SkipConnection(current, activation)
            else:
                layer = Dense(current, hsz, activation)
            if layer_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(hsz, eps=layer_norm_eps))
            layer_stack.append(WithDropout(layer, pdrop_value))
            current = hsz
        self.layer_stack = nn.Sequential(*layer_stack)
        self.requires_length = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param inputs: The fixed representation of the model

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """
        return self.layer_stack(inputs)


class VectorSequenceAttention(nn.Module):
    def __init__(self, hsz: int):
        super().__init__()
        self.hsz = hsz
        self.W_c = nn.Linear(2 * self.hsz, hsz, bias=False)

    def forward(self, query_t, keys_bth, values_bth, keys_mask=None):
        # Output(t) = B x H x 1
        # Keys = B x T x H
        # a = B x T x 1
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)

        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        pass

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = torch.bmm(a, values_bth).squeeze(1)

        attended = torch.cat([c_t, query_t], -1)
        attended = torch.tanh(self.W_c(attended))
        return attended


def dot_product_attention_weights(query_t: torch.Tensor,
                                  keys_bth: torch.Tensor,
                                  keys_mask: torch.Tensor) -> torch.Tensor:
    a = keys_bth @ query_t.unsqueeze(2)
    a = a.squeeze(2).masked_fill(keys_mask == MASK_FALSE, -1e9)
    a = F.softmax(a, dim=-1)
    return a


def dot_product_attention_weights_lengths(query_t: torch.Tensor,
                                          keys_bth: torch.Tensor,
                                          keys_lengths: torch.Tensor) -> torch.Tensor:
    mask = sequence_mask(keys_lengths, keys_bth.shape[1]).to(keys_bth.device)
    return dot_product_attention_weights(query_t, keys_bth, mask)


class LuongDotProductAttention(VectorSequenceAttention):
    def __init__(self, hsz):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        return dot_product_attention_weights(query_t, keys_bth, keys_mask)


class ScaledDotProductAttention(VectorSequenceAttention):
    def __init__(self, hsz):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = (keys_bth @ query_t.unsqueeze(2)) / math.sqrt(self.hsz)
        a = a.squeeze(2).masked_fill(keys_mask == MASK_FALSE, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class LuongGeneralAttention(VectorSequenceAttention):
    def __init__(self, hsz):
        super().__init__(hsz)
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ self.W_a(query_t).unsqueeze(2)
        a = a.squeeze(2).masked_fill(keys_mask == MASK_FALSE, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class BahdanauAttention(VectorSequenceAttention):
    def __init__(self, hsz):
        super().__init__(hsz)
        self.hsz = hsz
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.E_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.v = nn.Linear(self.hsz, 1, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        B, T, H = keys_bth.shape
        q = self.W_a(query_t.view(-1, self.hsz)).view(B, 1, H)
        u = self.E_a(keys_bth).view(B, T, H)
        z = torch.tanh(q + u)
        a = self.v(z.view(-1, self.hsz)).view(B, T)
        a = a.masked_fill(keys_mask == MASK_FALSE, -1e9)
        a = F.softmax(a, dim=-1)
        return a

    def _update(self, a, query_t, values_bth):
        query_t = query_t.view(-1, self.hsz)
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H) -> (B x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = (a @ values_bth).squeeze(1)
        # (B x 2H)
        attended = torch.cat([c_t, query_t], -1)
        attended = self.W_c(attended)
        return attended


class FineTuneModel(nn.Module):
    def __init__(self, nc, embeddings, stack_model=None):
        super().__init__()
        if isinstance(embeddings, dict):
            self.finetuned = EmbeddingsStack(embeddings)
        else:
            self.finetuned = embeddings
        self.stack_model = stack_model
        output_dim = self.finetuned.output_dim if stack_model is None else stack_model.output_dim
        self.output_layer = Dense(output_dim, nc, activation="log_softmax")

    def forward(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class CompositePooling(nn.Module):
    """Composite pooling allows for multiple sub-modules during pooling to be used in parallel
    """

    def __init__(self, models):
        """
        Note, this currently requires that each submodel is an eight_mile model with an `output_dim` attr
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.output_dim = sum(m.output_dim for m in self.models)
        self.requires_length = any(getattr(m, "requires_length", False) for m in self.models)

    def forward(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        pooled = []
        for sub_model in self.models:
            if getattr(sub_model, "requires_length", False):
                pooled.append(sub_model((inputs, lengths)))
            else:
                pooled.append(sub_model(inputs))
        return torch.cat(pooled, -1)


class EmbedPoolStackModel(nn.Module):
    """This provides an idiom for classification consisting of multiple phases

    In the first phase, we embed the input tensors, and subsequently pool them to
    a fixed width representation.  Finally, we allow multiple hidden "stacking"
    layers, ultimately ending in a projection to the output space

    """

    def __init__(
        self,
        nc: int,
        embeddings: nn.Module,
        pool_model: nn.Module,
        stack_model: Optional[nn.Module] = None,
        output_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_model = embeddings
        self.pool_model = pool_model
        self.stack_model = stack_model if stack_model else nn.Identity()
        output_dim = self.pool_model.output_dim if stack_model is None else stack_model.output_dim
        self.output_layer = Dense(output_dim, nc, activation="log_softmax") if output_model is None else output_model

    def forward(self, inputs: Dict[str, torch.Tensor]):
        lengths = inputs["lengths"]
        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


class PassThru(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class WithoutLength(nn.Module):
    """Wrapper layer to remove lengths from the input
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.output_dim = self.layer.output_dim if hasattr(self.layer, "output_dim") else 0

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.layer(inputs[0])


class WithDropout(nn.Module):
    """Wrapper for any layer that surrounds it with dropout"""

    def __init__(self, layer: nn.Module, pdrop: float = 0.5, variational=False, batch_first=False):
        """Create a dropout wrapper around the given layer

        :param layer: Some sort of layer
        :param pdrop: A dropout value
        """
        super().__init__()
        self.layer = layer
        self.dropout = VariationalDropout(pdrop, batch_first=batch_first) if variational else nn.Dropout(pdrop)
        self.output_dim = self.layer.output_dim if hasattr(self.layer, "output_dim") else 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the layer followed by dropout

        :param inputs: input tensor
        :return: output transformed by the held layer and subsequent dropout
        """
        return self.dropout(self.layer(inputs))


class WithDropoutOnFirst(nn.Module):
    """Wrapper for any layer that surrounds it with dropout

    This exists primarily for the LSTMEncoderWithState to allow dropout on the output while
    passing back the hidden state
    """

    def __init__(self, layer: nn.Module, pdrop: float = 0.5, variational=False):
        """Create a dropout wrapper around the given layer

        :param layer: Some sort of layer
        :param pdrop: A dropout value
        """
        super().__init__()
        self.layer = layer
        self.dropout = VariationalDropout(pdrop) if variational else nn.Dropout(pdrop)
        self.output_dim = self.layer.output_dim if hasattr(self.layer, "output_dim") else 0

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Apply the layer followed by dropout

        :param inputs: input tensor
        :return: output transformed by the held layer and subsequent dropout
        """
        outputs = self.layer(inputs)
        return self.dropout(outputs[0]), outputs[1]


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a mask to enforce span sequence transition constraints.

    Returns a Tensor with valid transitions as a 0 and invalid as a 1 for easy use with `masked_fill`
    """
    np_mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx=pad_idx)
    return torch.from_numpy(np_mask) == 0


@torch.jit.script
def inplace_assign(data: torch.Tensor, index: torch.Tensor, new_data: torch.Tensor) -> torch.Tensor:
    new_data = new_data.unsqueeze(0)
    index = index.expand(1, new_data.size(1))
    data.scatter_(0, index, new_data)
    return data


@torch.jit.script
def i2t(i: int) -> torch.Tensor:
    return torch.tensor(i).unsqueeze(0)


@torch.jit.script
def script_viterbi(
    unary: torch.Tensor, trans: torch.Tensor, start_idx: int, end_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    seq_len: int = unary.size(0)
    num_tags: int = unary.size(1)
    fill_value: float = -1e4
    alphas = torch.full((num_tags,), fill_value, dtype=unary.dtype, device=unary.device)
    broadcast_idx = torch.full((num_tags,), start_idx, dtype=torch.long)
    alphas = alphas.scatter(0, broadcast_idx, torch.zeros((num_tags,)))
    alphas = alphas.unsqueeze(0)
    backpointers: torch.Tensor = torch.zeros(num_tags, dtype=torch.long).unsqueeze(0)
    for i in range(seq_len):
        unary_t = unary[i, :]
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 1)
        backpointers = torch.cat([backpointers, best_tag_ids.unsqueeze(0)], 0)
        alphas = (viterbi + unary_t).unsqueeze(0)

    terminal_vars = alphas.squeeze(0) + trans[end_idx, :]
    path_score, best_tag_id = torch.max(terminal_vars, 0)
    best_path = best_tag_id.unsqueeze(0)

    for i in range(unary.size(0)):
        t = seq_len - i - 1
        best_tag_id = backpointers[t + 1, best_tag_id]
        best_path = torch.cat([best_path, best_tag_id.unsqueeze(0)], -1)

    new_path_vec = best_path.flip(0)
    return new_path_vec[1:], path_score


class ViterbiBatchSize1(nn.Module):
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, _: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unary = unary.squeeze(1)
        trans = trans.squeeze(0)
        path, score = script_viterbi(unary, trans, self.start_idx, self.end_idx)
        return path.unsqueeze(1), score


class Viterbi(nn.Module):
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        # r, start_idx: int, end_idx: int, norm = lambda x, y: x

    def forward(
        self, unary: torch.Tensor, trans: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []

        # Alphas: [B, 1, N]
        alphas = torch.full((batch_size, 1, tag_size), -1e4, device=unary.device)
        alphas[:, 0, self.start_idx] = 0
        # alphas = self.norm(alphas)

        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            # This part generates a warning
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas

        # Add end tag
        terminal_var = alphas.squeeze(1) + trans[:, self.end_idx, :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        # Flip lengths
        rev_len = seq_len - lengths - 1

        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            # Get new best tag candidate
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            # We are going backwards now, if flipped length was passed
            # these you aren't in your real results yet
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        # Mask out the extra tags (This might be pointless given thathatt anything that
        # will use this as a dense tensor downstream will mask it itself?)
        seq_mask = sequence_mask(lengths, seq_len).to(best_path.device).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


@torch.jit.script
def script_viterbi_log_softmax_norm(
    unary: torch.Tensor, trans: torch.Tensor, start_idx: int, end_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    seq_len: int = unary.size(0)
    num_tags: int = unary.size(1)
    fill_value: float = -1e4
    alphas = torch.full((num_tags,), fill_value, dtype=unary.dtype, device=unary.device)
    broadcast_idx = torch.full((num_tags,), start_idx, dtype=torch.long)
    alphas = alphas.scatter(0, broadcast_idx, torch.zeros((num_tags,)))
    alphas = alphas.unsqueeze(0)
    alphas = torch.log(F.softmax(alphas, dim=-1))
    backpointers: torch.Tensor = torch.zeros(num_tags, dtype=torch.long).unsqueeze(0)
    for i in range(seq_len):
        unary_t = unary[i, :]
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 1)
        backpointers = torch.cat([backpointers, best_tag_ids.unsqueeze(0)], 0)
        alphas = (viterbi + unary_t).unsqueeze(0)

    terminal_vars = alphas.squeeze(0) + trans[end_idx, :]
    path_score, best_tag_id = torch.max(terminal_vars, 0)
    best_path = best_tag_id.unsqueeze(0)

    for i in range(unary.size(0)):
        t = seq_len - i - 1
        best_tag_id = backpointers[t + 1, best_tag_id]
        best_path = torch.cat([best_path, best_tag_id.unsqueeze(0)], -1)

    new_path_vec = best_path.flip(0)
    return new_path_vec[1:], path_score


class ViterbiLogSoftmaxNormBatchSize1(nn.Module):
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, _: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unary = unary.squeeze(1)
        trans = trans.squeeze(0)
        path, score = script_viterbi_log_softmax_norm(unary, trans, self.start_idx, self.end_idx)
        return path.unsqueeze(1), score


class ViterbiLogSoftmaxNorm(Viterbi):
    def forward(
        self, unary: torch.Tensor, trans: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []

        # Alphas: [B, 1, N]
        alphas = torch.full((batch_size, 1, tag_size), -1e4, device=unary.device)
        alphas[:, 0, self.start_idx] = 0
        alphas = F.log_softmax(alphas, dim=-1)

        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas

        # Add end tag
        terminal_var = alphas.squeeze(1) + trans[:, self.end_idx, :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        # Flip lengths
        rev_len = seq_len - lengths - 1

        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            # Get new best tag candidate
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            # We are going backwards now, if flipped length was passed
            # these you aren't in your real results yet
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        # Mask out the extra tags (This might be pointless given that anything that
        # will use this as a dense tensor downstream will mask it itself?)
        seq_mask = sequence_mask(lengths).to(best_path.device).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


def ident(x):
    return x


class TaggerGreedyDecoder(nn.Module):
    def __init__(
        self,
        num_tags: int,
        constraint_mask: Optional[torch.Tensor] = None,
        batch_first: bool = True,
        reduction: str = "batch",
    ):
        """A Greedy decoder and loss module for taggers.

        :param num_tags: `int` The number of output classes
        :param constraint_mask: `Tensor[1, N, N]` A mask with valid transitions as 1 and invalid as 0
        :param batch_first: `bool` Should the batch dimensions be first?
        :param reduction: `str` Should the loss be calculated at the token level or batch level
        """
        super().__init__()
        self.num_tags = num_tags

        if constraint_mask is not None:
            constraint_mask = F.log_softmax(
                torch.zeros(constraint_mask.shape).masked_fill(constraint_mask, -1e4), dim=1
            )
            self.register_buffer("constraint_mask", constraint_mask)
        else:
            self.constraint_mask = None
        # FIXME: we cant do it like this if using TorchScript
        self.to_batch_first = ident if batch_first else tbh2bth
        self.to_time_first = bth2tbh if batch_first else ident
        self.batch_first = batch_first
        self.loss = SequenceLoss(LossFn=nn.CrossEntropyLoss, avg=reduction)
        self.viterbi = ViterbiLogSoftmaxNorm(Offsets.GO, Offsets.EOS)

    @property
    def transitions(self):
        return self.constraint_mask

    def neg_log_loss(self, inputs, tags, lengths):
        unaries = self.to_batch_first(inputs)
        tags = self.to_batch_first(tags)
        return self.loss(unaries, tags)

    def forward(self, inputs) -> torch.Tensor:
        unaries, lengths = tensor_and_lengths(inputs)
        # If there is a constraint mask do a masked viterbi
        if self.constraint_mask is not None:
            probv = self.to_time_first(unaries)
            probv = F.log_softmax(probv, dim=-1)
            preds, scores = self.viterbi(probv, self.constraint_mask, lengths)
            if self.batch_first:
                return tbh2bth(preds)  # , scores
            else:
                return preds
        else:
            # Decoding doesn't care about batch/time first
            _, preds = torch.max(unaries, -1)
            mask = sequence_mask(lengths, unaries.shape[1]).to(preds.device)
            # The mask gets generated as batch first
            mask = mask if self.batch_first else mask.transpose(0, 1)
            preds = preds.masked_fill(mask == MASK_FALSE, 0)
        return preds  # , None

    def extra_repr(self) -> str:
        str_ = f"n_tags={self.num_tags}, batch_first={self.batch_first}"
        if self.constraint_mask is not None:
            str_ += ", constrained=True"
        return str_


class CRF(nn.Module):
    def __init__(
        self,
        num_tags: int,
        constraint_mask: Optional[torch.Tensor] = None,
        batch_first: bool = True,
        idxs: Tuple[int, int] = (Offsets.GO, Offsets.EOS),
    ):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint: torch.ByteTensor, Constraints on the transitions [1, N, N]
        :param idxs: Tuple(int. int), The index of the start and stop symbol
            in emissions.
        :param batch_first: bool, if the input [B, T, ...] or [T, B, ...]

        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
        """
        super().__init__()
        self.start_idx, self.end_idx = idxs
        self.num_tags = num_tags
        if constraint_mask is not None:
            self.register_buffer("constraint_mask", constraint_mask)
        else:
            self.constraint_mask = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.num_tags, self.num_tags).zero_())
        self.batch_first = batch_first
        self.viterbi = Viterbi(self.start_idx, self.end_idx)

    def extra_repr(self) -> str:
        str_ = "n_tags=%d, batch_first=%s" % (self.num_tags, self.batch_first)
        if self.constraint_mask is not None:
            str_ += ", constrained=True"
        return str_

    @property
    def transitions(self):
        if self.constraint_mask is not None:
            return self.transitions_p.masked_fill(self.constraint_mask, -1e4)
        return self.transitions_p

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param tags: torch.LongTensor: [T, B] or [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # Convert from [B, T, N] -> [T, B, N]
        if self.batch_first:
            unary = unary.transpose(0, 1)
            tags = tags.transpose(0, 1)
        _, batch_size, _ = unary.size()
        fwd_score = self._forward_alg(unary, lengths)
        gold_score = self.score_sentence(unary, tags, lengths)

        loss = fwd_score - gold_score
        batch_loss = torch.mean(loss)
        return batch_loss

    def score_sentence(self, unary: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        batch_size = lengths.shape[0]
        assert lengths.shape[0] == unary.shape[1]

        trans = self.transitions.squeeze(0)  # [N, N]
        start = torch.full((1, batch_size), self.start_idx, dtype=tags.dtype, device=tags.device)  # [1, B]
        tags = torch.cat([start, tags], 0)  # [T + 1, B]

        # Unfold gives me all slices of size 2 (this tag next tag) from dimension T
        tag_pairs = tags.unfold(0, 2, 1)
        # Move the pair dim to the front and split it into two
        indices = tag_pairs.permute(2, 0, 1).chunk(2)
        trans_score = trans[[indices[1], indices[0]]].squeeze(0)
        # Pull out the values of the tags from the unary scores.
        unary_score = unary.gather(2, tags[1:].unsqueeze(-1)).squeeze(-1)
        mask = sequence_mask(lengths).transpose(0, 1).to(tags.device)
        scores = unary_score + trans_score
        scores = scores.masked_fill(mask == MASK_FALSE, 0)
        scores = scores.sum(0)

        eos_scores = trans[self.end_idx, tags.gather(0, lengths.unsqueeze(0)).squeeze(0)]
        scores = scores + eos_scores
        return scores

    def _forward_alg(self, unary: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # alphas: [B, 1, N]
        min_length = torch.min(lengths)
        batch_size = lengths.shape[0]
        lengths.shape[0] == unary.shape[1]
        alphas = torch.full((batch_size, 1, self.num_tags), -1e4, device=unary.device)
        alphas[:, 0, self.start_idx] = 0.0
        # alphas.requires_grad = True

        trans = self.transitions  # [1, N, N]

        for i, unary_t in enumerate(unary):
            # unary_t: [B, N]
            unary_t = unary_t.unsqueeze(2)  # [B, N, 1]
            # Broadcast alphas along the rows of trans
            # Broadcast trans along the batch of alphas
            # [B, 1, N] + [1, N, N] -> [B, N, N]
            # Broadcast unary_t along the cols of result
            # [B, N, N] + [B, N, 1] -> [B, N, N]
            scores = alphas + trans + unary_t
            new_alphas = vec_log_sum_exp(scores, 2).transpose(1, 2)
            # If we haven't reached your length zero out old alpha and take new one.
            # If we are past your length, zero out new_alpha and keep old one.

            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas

        terminal_vars = alphas + trans[:, self.end_idx]
        alphas = vec_log_sum_exp(terminal_vars, 2)
        return alphas.view(batch_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        unary, lengths = inputs
        if self.training:
            if self.batch_first:
                unary = unary.transpose(0, 1)
            forward = self._forward_alg(unary, lengths)
            # if self.batch_first:
            #    forward = forward.transpose(0, 1)
            return forward
        with torch.no_grad():
            return self.decode(unary, lengths)[0]

    @jit.export
    def decode(self, unary: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: torch.LongTensor: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        trans = self.transitions  # [1, N, N]
        path, score = self.viterbi(unary, trans, lengths)
        if self.batch_first:
            path = path.transpose(0, 1)
        return path, score


class SequenceModel(nn.Module):
    def __init__(self, nc: int, embeddings: nn.Module, transducer: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.embed_model = embeddings
        self.transducer_model = transducer
        # TODO: make this a separate model!
        if transducer.output_dim != nc:
            self.proj_layer = Dense(transducer.output_dim, nc)
        else:
            self.proj_layer = nn.Identity()
        self.decoder_model = decoder

    def transduce(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        lengths = inputs["lengths"]

        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        # transduced = self.transducer_model(embedded)
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.decoder_model((transduced, lengths))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass


class TagSequenceModel(SequenceModel):
    def __init__(self, nc: int, embeddings: nn.Module, transducer: nn.Module, decoder: Optional[nn.Module] = None):
        decoder_model = CRF(nc, batch_first=True) if decoder is None else decoder
        super().__init__(nc, embeddings, transducer, decoder_model)

    def neg_log_loss(self, unary: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.decoder_model.neg_log_loss(unary, tags, lengths)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        transduced = self.transduce(inputs)
        path = self.decode(transduced, inputs["lengths"])
        return path


class LangSequenceModel(nn.Module):
    def __init__(
        self,
        nc: int,
        embeddings: nn.Module,
        transducer: nn.Module,
        decoder: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.embed_model = embeddings
        self.transducer_model = transducer
        if hasattr(transducer, "requires_state") and transducer.requires_state:
            self._call = self._call_with_state
            self.requires_state = True
        else:
            self._call = self._call_without_state
            self.requires_state = False
        self.output_layer = nn.Linear(self.transducer_model.output_dim, nc)
        self.decoder_model = decoder

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._call(inputs)

    def _call_with_state(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        h = inputs["h"]

        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        transduced = self.output_layer(transduced)
        return transduced, hidden

    def _call_without_state(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.embed_model(inputs)
        transduced = self.transducer_model((embedded, None))
        transduced = self.output_layer(transduced)
        return transduced, None


def pytorch_embedding(weights: torch.Tensor, finetune: bool = True) -> nn.Embedding:
    """Creation function for making an nn.Embedding with the given weights

    :param weights: The weights to use
    :param finetune: Should we fine-tune the embeddings or freeze them
    """
    lut = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=Offsets.PAD)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=finetune)
    return lut


def subsequent_mask(size: int):
    """
    Creates a lower triangular mask to mask future

    :param size: Temporal length
    :return: A tensor of type `uint8` that is 1s along diagonals and below, zero  o.w
    """
    attn_shape = (1, 1, size, size)
    sub_mask = np.tril(np.ones(attn_shape)).astype("uint8")
    return torch.from_numpy(sub_mask)


class SequenceSequenceAttention(nn.Module):
    def __init__(self, hsz: int = None, pdrop: float = 0.1, **kwargs):
        super().__init__()
        self.hsz = hsz
        self.dropout = nn.Dropout(pdrop)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        query, key, value, mask = qkvm
        a = self._attention(query, key, mask)
        self.attn = a
        a = self.dropout(a)
        return self._update(a, value)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def _update(self, a: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param values: The values [B, H, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        return torch.matmul(a, value)


class SeqScaledDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop: float = 0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: A tensor that is (BxHxTxT)
        """
        # (., H, T, T) = (., H, T, D) x (., H, D, T)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1e9)
        return F.softmax(scores, dim=-1)


class SeqDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop: float = 0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1e9)
        return F.softmax(scores, dim=-1)


class SequenceSequenceRelativeAttention(nn.Module):
    """This form of attention is specified in Shaw et al 2018: https://www.aclweb.org/anthology/N18-2074.pdf

    """

    def __init__(self, hsz: int = None, pdrop: float = 0.1, **kwargs):
        super().__init__()
        self.hsz = hsz
        self.dropout = nn.Dropout(pdrop)
        self.attn = None

    def forward(
        self, q_k_v_ek_ev_m: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Take in a tuple of tensors corresponding to the query, key, value, edges_key, edges_value and mask variables

        :param q_k_v_ek_ev_m: A tuple consisting of query, key, value, `edges_key`, `edges_value` and `mask` respectively
        :return: An updated value Tensor
        """
        query, key, value, edges_key, edges_value, mask = q_k_v_ek_ev_m
        a = self._attention(query, key, edges_key, mask)
        self.attn = a
        a = self.dropout(a)
        return self._update(a, value, edges_value)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, edges_key: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    def _update(self, a: torch.Tensor, value: torch.Tensor, edges_value: torch.Tensor) -> torch.Tensor:
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param value: The values [B, H, T, D]
        :param edge_value: The edge values [T, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        B, H, T, D = value.shape
        updated_values = torch.matmul(a, value)
        if edges_value is not None:
            a = a.view(B * H, T, T).transpose(0, 1)  # (T, BxH, T)
            t = torch.matmul(a, edges_value)  # (T, BxH, D)
            update_edge_values = t.transpose(0, 1).view(B, H, T, D)
            return updated_values + update_edge_values
        else:
            return updated_values


class SeqScaledDotProductRelativeAttention(SequenceSequenceRelativeAttention):
    def __init__(self, pdrop: float = 0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, edges_key: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [TxTxD]
        :return: A tensor that is (BxHxTxT)
        """
        # (., H, T, T) = (., H, T, D) x (., H, D, T)
        B, H, T, d_k = query.shape
        scores_qk = torch.matmul(query, key.transpose(-2, -1))
        tbhd = query.reshape(B * H, T, d_k).transpose(0, 1)
        scores_qek = torch.matmul(tbhd, edges_key.transpose(-2, -1))
        scores_qek = scores_qek.transpose(0, 1).view(B, H, T, T)
        scores = (scores_qk + scores_qek) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1e9)
        return F.softmax(scores, dim=-1)


class SeqDotProductRelativeAttention(SequenceSequenceRelativeAttention):
    def __init__(self, pdrop: float = 0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, edges_key: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, H, T, d_k = query.shape
        scores_qk = torch.matmul(query, key.transpose(-2, -1))
        tbhd = query.reshape(B * H, T, d_k).transpose(0, 1)
        scores_qek = torch.matmul(tbhd, edges_key.transpose(-2, -1))
        scores_qek = scores_qek.transpose(0, 1).view(B, H, T, T)
        scores = scores_qk + scores_qek
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1e9)
        return F.softmax(scores, dim=-1)


def unfold_tensor(tensor, dim, window_sz):
    """Unfold a tensor by applying a sliding window on a certain dimension with step 1 and padding of 0's. The window
    dimension is added as the last dimension

    :param tensor: the tensor to be unfolded, with shape [d_1, d_2, ..., T, ..., d_n]
    :param dim: the dimension along which unfolding is applied
    :param window_sz: sliding window size, need to be an odd number

    :return: the unfolded tensor with shape [d_1, d_2, ..., T, ..., d_n, window_sz]
    """
    half_window = (window_sz - 1) // 2
    if dim < 0:
        dim = len(tensor.shape) + dim
    # torch.nn.functional.pad apply backwardly from the last dimension
    padding = [0, 0] * (len(tensor.shape) - dim - 1) + [half_window, half_window]
    return F.pad(tensor, padding).unfold(dim, window_sz, 1)


class SeqScaledWindowedRelativeAttention(SequenceSequenceRelativeAttention):
    """This class implements windowed relative attention, i.e. preventing attention beyond rpr_k. For efficiency,
    _attention and _update are implemented in a different way."""
    def __init__(self, pdrop: float = 0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _unfold_mask(self, mask, batchsz, rpr_k):
        """Transform mask into the unfolded format."""
        window_sz = 2 * rpr_k + 1
        T = mask.shape[3]
        if mask.shape[2] > 1:  # mask is from a subsequent mask, with [1, 1, T, T] or [B, 1, T, T]
            logger.warning("Using subsequent mask with long sequence may cause OOM error.")
            mask = mask.expand(batchsz, 1, T, T)  # expand sequence/subsequent mask into a uniform dim
            mask = F.pad(mask, [rpr_k, rpr_k])  # pad both sides with rpr_k, [B, 1, T, T + 2*rpr_k]
            seq = torch.arange(T + 2 * rpr_k)
            indices = seq.unfold(0, window_sz, 1)  # indices of a sliding window, [T, W]
            indices = indices.unsqueeze(0).unsqueeze(0).expand(batchsz, 1, T, window_sz).to(mask.device)
            return torch.gather(mask, -1, indices)  # [B, 1, T, W]):
        else:  # mask is a sequence mask [B, 1, 1, T]
            unfolded = unfold_tensor(mask, dim=-1, window_sz=window_sz)  # [B, 1, 1, T, W]
            return unfolded.squeeze(1)  # [B, 1, T, W]

    def _attention(
            self, query: torch.Tensor, key: torch.Tensor, rpr_key: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Implementation of attention considering RA masking: using torch.Tensor.unfold to create an extra dimension
        representing the sliding window. Then when applying matmul, Q, K, V share the same T dimension.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param rpr_key: tensor of the rpr_key embeddings [W, d_k]
        :return: A tensor that is [B, H, T, 1, W] to be matmul with values
        """
        B, H, T, d_k = query.shape
        window_sz = rpr_key.shape[0]
        rpr_k = (window_sz - 1) // 2
        query = query.unsqueeze(-2)  # [B, H, T, 1, d_k]
        key = unfold_tensor(key, dim=2, window_sz=window_sz)  # [B, H, T, d_k, W]
        rpr_key = rpr_key.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, d_k, W]

        scores_qk = torch.matmul(query, key)  # [B, H, T, 1, W]
        scores_qrk = torch.matmul(query, rpr_key)  # [B, H, T, 1, W]
        scores = (scores_qk + scores_qrk) / math.sqrt(d_k)
        if mask is not None:
            mask = self._unfold_mask(mask, B, rpr_k).unsqueeze(-2)  # [B, 1, T, 1, W]
            scores = scores.masked_fill(mask == False, -1e9)
        return F.softmax(scores, dim=-1)

    def _update(self, a: torch.Tensor, value: torch.Tensor, rpr_value: torch.Tensor) -> torch.Tensor:
        # a has dim [B, H, T, 1, W]
        window_sz = a.shape[-1]
        value = unfold_tensor(value, dim=2, window_sz=window_sz).transpose(-1, -2)  # [B, H, T, W, d_value]
        updated_values = torch.matmul(a, value)  # [B, H, T, 1, d_value]
        if rpr_value is not None:
            rpr_value = rpr_value.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, W, d_value]
            update_rpr_values = torch.matmul(a, rpr_value)  # [B, H, T, 1, d_value]
            return (updated_values + update_rpr_values).squeeze(3)  # [B, H, T, d_value]
        else:
            return updated_values.squeeze(3)


class SeqBahdanauAttention(SequenceSequenceAttention):
    def __init__(self, hsz: int, pdrop: float = 0.1, **kwargs):
        super().__init__(hsz, pdrop=pdrop, **kwargs)
        self.V = pytorch_linear(self.hsz, 1, bias=False)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # [B, H, T, 1, D] + [B, H, 1, T, D] = [B, H, T, T, D]
        additive = query.unsqueeze(-2) + key.unsqueeze(-3)
        non_linear = torch.tanh(additive)
        # [B, H, T, T, D] @ [D, 1] = [B, H, T, T, 1]
        scores = self.V(non_linear)
        # [B, H, T, T]
        scores = scores.squeeze(-1)
        return F.softmax(scores, dim=-1)


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """

    def __init__(
        self, num_heads: int, d_model: int, dropout: float = 0.1, scale: bool = False, d_k: Optional[int] = None
    ):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})")
        else:
            self.d_k = d_k
        self.h = num_heads
        # for multi-headed attention, w_V projects to h heads, each head has dim d_k; for single headed attention, w_V
        # project to 1 head with dim d_model
        if self.h > 1:
            self.d_value = self.d_k
        else:
            self.d_value = d_model
        self.w_Q = Dense(d_model, self.d_k * self.h)
        self.w_K = Dense(d_model, self.d_k * self.h)
        self.w_V = Dense(d_model, self.d_value * self.h)
        if self.h > 1:  # w_O is not needed for sinlge headed attention
            self.w_O = Dense(self.d_k * self.h, d_model)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_value).transpose(1, 2)

        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn

        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_value)
        if self.h > 1:
            return self.w_O(x)
        else:
            return x


class MultiHeadedRelativeAttention(nn.Module):
    """
    Multi-headed relative attention from Shaw et al 2018 (https://www.aclweb.org/anthology/N18-2074.pdf)

    This method follows the same approach of MultiHeadedAttention, but it computes Relative Position Representations (RPR)
    which are used as part of the attention computations.  To facilitate this, the model has its own internal
    embeddings lookup table, and it has an updated computation for both the attention weights and the application
    of those weights to follow them.

    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        rpr_k: int,
        dropout: float = 0.1,
        scale: bool = False,
        d_k: Optional[int] = None,
        windowed_ra: bool = False,
        rpr_value_on: bool = True
    ):
        """Constructor for multi-headed attention

        :param num_heads: The number of heads
        :param d_model: The model hidden size
        :param rpr_k: distance within which relative positional embedding will be considered
        :param windowed_ra: whether prevent attention beyond rpr_k
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()

        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})")
        else:
            self.d_k = d_k

        self.h = num_heads
        # for multi-headed attention, w_V projects to h heads, each head has dim d_k; for single headed attention, w_V
        # project to 1 head with dim d_model
        if self.h > 1:
            self.d_value = self.d_k
        else:
            self.d_value = d_model
        self.rpr_k = rpr_k
        self.rpr_value_on = rpr_value_on
        self.rpr_key = nn.Embedding(2 * rpr_k + 1, self.d_k)
        if self.rpr_value_on:
            self.rpr_value = nn.Embedding(2 * rpr_k + 1, self.d_value)
        self.windowed_ra = windowed_ra
        self.w_Q = Dense(d_model, self.d_k * self.h)
        self.w_K = Dense(d_model, self.d_k * self.h)
        self.w_V = Dense(d_model, self.d_value * self.h)
        if self.h > 1:  # w_O is not needed for sinlge headed attention
            self.w_O = Dense(self.d_k * self.h, d_model)
        if scale:
            if windowed_ra:
                self.attn_fn = SeqScaledWindowedRelativeAttention(dropout)
            else:
                self.attn_fn = SeqScaledDotProductRelativeAttention(dropout)
        else:
            self.attn_fn = SeqDotProductRelativeAttention(dropout)
        self.attn = None

    def make_rpr(self, seq_len, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a matrix shifted by self.rpr_k and bounded between 0 and 2*self.rpr_k to provide 0-based indexing for embedding
        """
        seq = torch.arange(seq_len).to(device)
        window_len = 2 * self.rpr_k
        edges = seq.view(1, -1) - seq.view(-1, 1) + self.rpr_k
        edges = torch.clamp(edges, 0, window_len)
        if self.rpr_value_on:
            return self.rpr_key(edges), self.rpr_value(edges)
        else:
            return self.rpr_key(edges), None

    def make_windowed_rpr(self, device):
        window_len = 2 * self.rpr_k + 1
        window = torch.arange(window_len).to(device)
        if self.rpr_value_on:
            return self.rpr_key(window), self.rpr_value(window)
        else:
            return self.rpr_key(window), None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_len = query.size(1)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_value).transpose(1, 2)

        if self.windowed_ra:
            rpr_key, rpr_value = self.make_windowed_rpr(query.device)
        else:
            rpr_key, rpr_value = self.make_rpr(seq_len, query.device)
        x = self.attn_fn((query, key, value, rpr_key, rpr_value, mask))
        self.attn = self.attn_fn.attn

        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_value)
        if self.h > 1:
            return self.w_O(x)
        else:
            return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[int] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: Optional[bool] = False,
        rpr_value_on: bool = True
    ):
        super().__init__()
        # to properly execute BERT models, we have to follow T2T and do layer norms after
        self.layer_norms_after = layer_norms_after
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k,
                                                          windowed_ra=windowed_ra, rpr_value_on=rpr_value_on)
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale, d_k=d_k)
        self.ffn = nn.Sequential(
            Dense(self.d_model, self.d_ff),
            get_activation(activation_type),
            nn.Dropout(ffn_pdrop),
            Dense(self.d_ff, self.d_model),
        )
        # Slightly late for a name change
        # LN1 = ln_x
        # LN2 = ln_attn_output
        self.ln1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param inputs: `(x, mask)`
        :return: The output tensor
        """
        x, mask = inputs

        if not self.layer_norms_after:
            x = self.ln1(x)
        h = self.self_attn((x, x, x, mask))
        x = x + self.dropout(h)
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[int] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_norms_after = layer_norms_after
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k)
            self.src_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k)

        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k)
            self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k)

        self.ffn = nn.Sequential(
            Dense(self.d_model, self.d_ff),
            nn.Dropout(ffn_pdrop),
            get_activation(activation_type),
            Dense(self.d_ff, self.d_model),
        )

        self.ln1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln3 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        x, memory, src_mask, tgt_mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x, tgt_mask)))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory, src_mask)))

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: Optional[bool] = False,
        rpr_value_on: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.ln = nn.Identity() if layer_norms_after else nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.output_dim = d_model

        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers

        for i in range(layers):
            self.encoders.append(
                TransformerEncoder(
                    num_heads, d_model, pdrop, scale, activation, d_ff, d_k,
                    rpr_k=rpr_k[i], ffn_pdrop=ffn_pdrop, layer_norms_after=layer_norms_after,
                    layer_norm_eps=layer_norm_eps, windowed_ra=windowed_ra, rpr_value_on=rpr_value_on
                )
            )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, mask = inputs
        for layer in self.encoders:
            x = layer((x, mask))
        return self.ln(x)


class TransformerEncoderStackWithLengths(TransformerEncoderStack):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: bool,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        input_sz: Optional[int] = None,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        **kwargs,
    ):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k,
                         layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps)
        self.proj = WithDropout(pytorch_linear(input_sz, d_model), pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        x, lengths = inputs
        x = self.proj(x)
        max_seqlen = x.shape[1]
        mask = sequence_mask(lengths, max_seqlen).to(x.device)
        return super().forward((x, mask.unsqueeze(1).unsqueeze(1)))


class TransformerEncoderStackWithTimeMask(TransformerEncoderStack):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: bool,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        input_sz: Optional[int] = None,
        layer_norms_after = False,
        **kwargs,
    ):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k, layer_norms_after=layer_norms_after)
        self.proj = WithDropout(pytorch_linear(input_sz, d_model), pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, lengths = inputs
        x = self.proj(x)
        max_seqlen = x.shape[1]
        mask = subsequent_mask(max_seqlen).to(x.device)
        return super().forward((x, mask.unsqueeze(1).unsqueeze(1)))


class TransformerDecoderStack(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        layers: int = 1,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6

    ):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.ln = nn.Identity() if layer_norms_after else nn.LayerNorm(d_model, eps=layer_norm_eps)

        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers

        for i in range(layers):
            self.decoders.append(
                TransformerDecoder(num_heads, d_model, pdrop, scale, activation_type, d_ff,
                                   d_k=d_k, rpr_k=rpr_k[i], ffn_pdrop=ffn_pdrop,
                                   layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps)
            )

    def forward(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        for layer in self.decoders:
            x = layer((x, memory, src_mask, tgt_mask))
        return self.ln(x)


def update_lengths(lengths, eoses, idx):
    """Update the length of a generated tensor based on the first EOS found.

    This is useful for a decoding situation where tokens after an EOS
    can be something other than EOS. This also makes sure that a second
    generated EOS doesn't affect the lengths.

    :param lengths: `torch.LongTensor`: The lengths where zero means an
        unfinished sequence.
    :param eoses:  `torch.ByteTensor`: A mask that has 1 for sequences that
        generated an EOS.
    :param idx: `int`: What value to fill the finished lengths with (normally
        the current decoding timestep).

    :returns: `torch.Tensor`: The updated lengths tensor (same shape and type).
    """
    # If a length is 0 it has never had a length set so it is eligible to have
    # this EOS be the length.
    updatable_lengths = lengths == 0
    # If this length can be updated AND this token is an eos
    lengths_mask = updatable_lengths & eoses
    return lengths.masked_fill(lengths_mask, idx)


def gnmt_length_penalty(lengths, alpha=0.8):
    """Calculate a length penalty from https://arxiv.org/pdf/1609.08144.pdf

    The paper states the penalty as (5 + |Y|)^a / (5 + 1)^a. This is implemented
    as ((5 + |Y|) / 6)^a for a (very) tiny performance boost

    :param lengths: `torch.LongTensor`: [B, K] The lengths of the beams.
    :param alpha: `float`: A hyperparameter. See Table 2 for a search on this
        parameter.

    :returns:
        `torch.FloatTensor`: [B, K, 1] The penalties.
    """
    lengths = lengths.to(torch.float)
    penalty = torch.pow(((5 + lengths) / 6), alpha)
    return penalty.unsqueeze(-1)


def no_length_penalty(lengths):
    """A dummy function that returns a no penalty (1)."""
    return torch.ones_like(lengths).to(torch.float).unsqueeze(-1)


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = torch.arange(10).view(2, -1)
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> a.repeat(2, 1)
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = t.shape
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = t.unsqueeze(dim + 1).repeat(tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim + 1 :])
    return tiled.view(new_shape)


class BeamSearchBase:
    def __init__(self, beam=1, length_penalty=None, **kwargs):
        self.length_penalty = length_penalty if length_penalty else no_length_penalty
        self.K = beam

    def init(self, encoder_outputs):
        pass

    def step(self, paths, extra):
        pass

    def update(self, beams, extra):
        pass

    def __call__(self, encoder_outputs, **kwargs):
        """Perform batched Beam Search.

        Note:
            The paths and lengths generated do not include the <GO> token.

        :param encoder_outputs: `namedtuple` The outputs of the encoder class.
        :param init: `Callable(ecnoder_outputs: encoder_outputs, K: int)` -> Any: A
            callable that is called once at the start of the search to initialize
            things. This returns a blob that is passed to other callables.
        :param step: `Callable(paths: torch.LongTensor, extra) -> (probs: torch.FloatTensor, extra):
            A callable that is does a single decoding step. It returns the log
            probabilities over the vocabulary in the last dimension. It also returns
            any state the decoding process needs.
        :param update: `Callable(beams: torch.LongTensor, extra) -> extra:
            A callable that is called to edit the decoding state based on the selected
            best beams.
        :param length_penalty: `Callable(lengths: torch.LongTensor) -> torch.floatTensor
            A callable that generates a penalty based on the lengths. Lengths is
            [B, K] and the returned penalty should be [B, K, 1] (or [B, K, V] to
            have token based penalties?)

        :Keyword Arguments:
        * *beam* -- `int`: The number of beams to use.
        * *mxlen* -- `int`: The max number of steps to run the search for.

        :returns:
            tuple(preds: torch.LongTensor, lengths: torch.LongTensor, scores: torch.FloatTensor)
            preds: The predicted values: [B, K, max(lengths)]
            lengths: The length of each prediction [B, K]
            scores: The score of each path [B, K]
        """
        mxlen = kwargs.get("mxlen", 100)
        bsz = encoder_outputs.output.shape[0]
        device = encoder_outputs.output.device
        with torch.no_grad():
            extra = self.init(encoder_outputs)
            paths = torch.full((bsz, self.K, 1), Offsets.GO, dtype=torch.long, device=device)
            # This tracks the log prob of each beam. This is distinct from score which
            # is based on the log prob and penalties.
            log_probs = torch.zeros((bsz, self.K), dtype=torch.float, device=device)
            # Tracks the lengths of the beams, unfinished beams have lengths of zero.
            lengths = torch.zeros((bsz, self.K), dtype=torch.long, device=device)

            for i in range(mxlen - 1):
                probs, extra = self.step(paths, extra)
                V = probs.shape[-1]
                probs = probs.view((bsz, self.K, V))  # [B, K, V]
                if i > 0:
                    # This mask is for all beams that are done.
                    done_mask = (lengths != 0).unsqueeze(-1)  # [B, K, 1]
                    # Can creating this mask be moved out of the loop? It never changes but we don't have V
                    # This mask selects the EOS token
                    eos_mask = torch.zeros((1, 1, V), dtype=done_mask.dtype, device=device)
                    eos_mask[:, :, Offsets.EOS] = 1
                    # This mask selects the EOS token of only the beams that are done.
                    mask = done_mask & eos_mask
                    # Put all probability mass on the EOS token for finished beams.
                    # Otherwise as the other beams get longer they will all give
                    # up and eventually select this beam and all outputs become
                    # the same.
                    probs = probs.masked_fill(done_mask, -np.inf)
                    probs = probs.masked_fill(mask, 0)
                    probs = log_probs.unsqueeze(-1) + probs  # [B, K, V]
                    # Calculate the score of the beam based on the current length.
                    path_scores = probs / self.length_penalty(lengths.masked_fill(lengths == 0, i + 1))
                else:
                    # On the first step we only look at probabilities for the first beam.
                    # If we don't then the probs will be the same for each beam
                    # This means the same token will be selected for each beam
                    # And we won't get any diversity.
                    # Using only the first beam ensures K different starting points.
                    path_scores = probs[:, 0, :]

                flat_scores = path_scores.view(bsz, -1)  # [B, K * V]
                best_scores, best_idx = flat_scores.topk(self.K, 1)
                # Get the log_probs of the best scoring beams
                log_probs = probs.view(bsz, -1).gather(1, best_idx).view(bsz, self.K)

                best_beams = best_idx // V  # Get which beam it came from
                best_idx = best_idx % V  # Get the index of the word regardless of which beam it is.

                # Best Beam index is relative within the batch (only [0, K)).
                # This makes the index global (e.g. best beams for the second
                # batch example is in [K, 2*K)).
                offsets = torch.arange(bsz, dtype=torch.long, device=device) * self.K
                offset_beams = best_beams + offsets.unsqueeze(-1)
                flat_beams = offset_beams.view(bsz * self.K)
                # Select the paths to extend based on the best beams
                flat_paths = paths.view(bsz * self.K, -1)
                new_paths = flat_paths[flat_beams, :].view(bsz, self.K, -1)
                # Add the selected outputs to the paths
                paths = torch.cat([new_paths, best_idx.unsqueeze(-1)], dim=2)

                # Select the lengths to keep tracking based on the valid beams left.
                lengths = lengths.view(-1)[flat_beams].view((bsz, self.K))

                extra = self.update(flat_beams, extra)

                # Updated lengths based on if we hit EOS
                last = paths[:, :, -1]
                eoses = last == Offsets.EOS
                lengths = update_lengths(lengths, eoses, i + 1)
                if (lengths != 0).all():
                    break
            else:
                # This runs if the loop didn't break meaning one beam hit the max len
                # Add an EOS to anything that hasn't hit the end. This makes the scores real.
                probs, extra = self.step(paths, extra)

                V = probs.size(-1)
                probs = probs.view((bsz, self.K, V))
                probs = probs[:, :, Offsets.EOS]  # Select the score of EOS
                # If any of the beams are done mask out the score of this EOS (they already had an EOS)
                probs = probs.masked_fill((lengths != 0), 0)
                log_probs = log_probs + probs
                end_tokens = torch.full((bsz, self.K, 1), Offsets.EOS, device=device, dtype=paths.dtype)
                paths = torch.cat([paths, end_tokens], dim=2)
                lengths = update_lengths(lengths, torch.ones_like(lengths) == 1, mxlen)
                lengths = update_lengths(lengths, torch.ones_like(lengths) == 1, mxlen)
                best_scores = log_probs / self.length_penalty(lengths).squeeze(-1)

        # Slice off the Offsets.GO token
        paths = paths[:, :, 1:]
        return paths, lengths, best_scores


def checkpoint_for(model_base, epoch, tick_type='epoch'):
    return '{}-{}-{}'.format(model_base, tick_type, epoch+1)


def rm_old_checkpoints(base_path, current_epoch, last_n=10):
    for i in range(0, current_epoch-last_n):
        checkpoint_i = checkpoint_for(base_path, i)
        for extension in ('.pth', '.npz'):
            checkpoint_name = checkpoint_i + extension
            if os.path.exists(checkpoint_name):
                os.remove(checkpoint_name)


def find_latest_checkpoint(checkpoint_dir: str, wildcard="checkpoint") -> str:
    step_num = 0
    for f in glob.glob(os.path.join(checkpoint_dir, f"{wildcard}*")):
        last = f.split("-")[-1]
        for x in ('.pth', '.npz'):
            last = last.replace(x, '', -1)
        this_step_num = int(last)
        if this_step_num > step_num:
            checkpoint = f
            step_num = this_step_num
    return checkpoint, step_num

def save_checkpoint(model: torch.nn.Module, model_base: str, count: int, tick_type: str = 'epoch', save_npz: bool = False):
    from eight_mile.pytorch.serialize import save_tlm_npz
    checkpoint_name = checkpoint_for(model_base, count, tick_type=tick_type)
    # Its possible due to how its called that we might save the same checkpoint twice if we dont check first
    if os.path.exists(checkpoint_name):
        logger.info("Checkpoint already exists: %s", checkpoint_name)
        return
    logger.info("Creating checkpoint: %s", checkpoint_name)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_name+'.pth')
        if save_npz:
            save_tlm_npz(model.module, checkpoint_name+'.npz')
    else:
        torch.save(model.state_dict(), checkpoint_name+'.pth')
        if save_npz:
            save_tlm_npz(model, checkpoint_name+'.npz')
    if tick_type == 'epoch':
        rm_old_checkpoints(model_base, count)


def init_distributed(local_rank):
    if local_rank == -1:
        # https://github.com/kubeflow/pytorch-operator/issues/128
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        logger.info("Setting local rank to RANK env variable")
        local_rank = int(os.environ['RANK'])
    logger.warning("Local rank (%d)", local_rank)
    # In an env like k8s with kubeflow each worker will only see a single gpu
    # with an id of 0. If the gpu count is 1 then we are probably in an env like
    # that so we should just use the first (and only) gpu avaiable
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    # This program assumes multiprocess/multi-device on a single node. Each
    # process gets a rank (via cli or ENV variable) and uses that rank to select
    # which gpu to use. This only makes sense on a single node, if you had 4
    # processes on 2 nodes where each node has 2 GPUs then the ranks would be
    # 0, 1, 2, 3 but the gpus numbers would be node 0: 0, 1 and node 1: 0, 1
    # and this assignment to gpu 3 would fail. On a single node with 4 processes
    # and 4 gpus the rank and gpu ids will align and this will work
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return device, local_rank


class SingleHeadReduction(nn.Module):
    """
    Implementation of the "self_attention_head" layer from the conveRT paper (https://arxiv.org/pdf/1911.03688.pdf)
    """
    def __init__(
            self, d_model: int, dropout: float = 0.0, scale: bool = True, d_k: Optional[int] = None
    ):
        """
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        self.d_model = d_model
        if d_k is None:
            self.d_k = d_model
        else:
            self.d_k = d_k
        self.w_Q = Dense(d_model, self.d_k)
        self.w_K = Dense(d_model, self.d_k)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """According to conveRT model's graph, they project token encodings to lower-dimensional query and key in single
        head, use them to calculate the attention score matrix that has dim [B, T, T], then sum over the query dim to
        get a tensor with [B, 1, T] (meaning the amount of attentions each token gets from all other tokens), scale it
        by sqrt of sequence lengths, then use it as the weight to weighted sum the token encoding to get the sentence
        encoding. we implement it in an equivalent way that can best make use of the eight_mile codes: do the matrix
        multiply with value first, then sum over the query dimension.
        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: sentence-level encoding with dim [B, d_model]
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_mask = mask.squeeze()  # [B, T]
        seq_lengths = seq_mask.sum(dim=1)

        # (B, H, T, D), still have num_heads = 1 to use the attention function defined in eight_miles
        query = self.w_Q(query).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        value = value.view(batchsz, -1, 1, self.d_model).transpose(1, 2)
        x = self.attn_fn((query, key, value, mask))  # [B, 1, T, D]
        self.attn = self.attn_fn.attn

        x = x.squeeze(1)  # [B, T, D]
        x = x * seq_mask.unsqueeze(-1)
        x = x.sum(dim=1)  # [B, D]
        x = x * seq_lengths.float().sqrt().unsqueeze(-1)
        return x


class TransformerDiscriminator(nn.Module):
    """A Transformer model that tries to predict if each token is real or fake


    This model is based on [ELECTRA: Pre-Training Text Encoders as Discriminators Rather Than Generators,
    Clark et al. 2019](https://openreview.net/pdf?id=r1xMH1BtvB).

    """

    def __init__(
            self,
            embeddings,
            num_heads: int,
            d_model: int,
            dropout: bool,
            layers: int = 1,
            activation: str = "relu",
            d_ff: Optional[int] = None,
            d_k: Optional[int] = None,
            rpr_k: Optional[Union[int, List[int]]] = None,
            layer_norms_after: bool = False,
            layer_norm_eps: float = 1.0e-6,
            **kwargs,
    ):
        super().__init__()
        self.embeddings = EmbeddingsStack(embeddings, dropout)
        self.weight_std = kwargs.get('weight_std', 0.02)
        assert self.embeddings.dsz == d_model
        self.transformer = TransformerEncoderStack(
            num_heads, d_model=d_model, pdrop=dropout, scale=True,
            layers=layers, activation=activation, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
            layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps
        )
        self.proj_to_output = pytorch_linear(d_model, 1)
        self.apply(self.init_layer_weights)
        self.lengths_feature = kwargs.get('lengths_feature', list(self.embeddings.keys())[0])

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, features):
        embedded = self.embeddings(features)
        x = features[self.lengths_feature]
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        transformer_out = self.transformer((embedded, input_mask))
        binary = self.proj_to_output(transformer_out)
        return torch.sigmoid(binary)

    def create_loss(self):
        return nn.BCELoss()



class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        super().__init__()
        if avg == 'token':
            # self.crit = LossFn(ignore_index=Offsets.PAD, reduction='elementwise-mean')
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=True)
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=False)
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)


def pytorch_conv1d(in_channels, out_channels, fsz, unif=0, padding=0, initializer=None):
    c = nn.Conv1d(in_channels, out_channels, fsz, padding=padding)
    if unif > 0:
        c.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(c.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(c.weight)
    else:
        nn.init.xavier_uniform_(c.weight)
    return c


def tie_weight(to_layer, from_layer):
    """Assigns a weight object to the layer weights.

    This method exists to duplicate baseline functionality across packages.

    :param to_layer: the pytorch layer to assign weights to
    :param from_layer: pytorch layer to retrieve weights from
    """
    to_layer.weight = from_layer.weight


class BilinearAttention(nn.Module):

    def __init__(self, in_hsz, out_hsz=1, bias_x=True, bias_y=True):
        super().__init__()

        self.in_hsz = in_hsz
        self.out_hsz = out_hsz
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(out_hsz, in_hsz + bias_x, in_hsz + bias_y))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        #nn.init.orthogonal_(self.weight)

    def forward(self, x, y, mask):
        r"""
        Args:
            x: ``[B, T, H]``.
            y: ``[B, T, H]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """
        if self.bias_x:
            ones = torch.ones(x.shape[:-1] + (1,), device=x.device)
            x = torch.cat([x, ones], -1)
        if self.bias_y:
            ones = torch.ones(x.shape[:-1] + (1,), device=y.device)
            y = torch.cat([y, ones], -1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        u = x @ self.weight
        s = u @ y.transpose(-2, -1)
        s = s.squeeze(1)
        s = s.masked_fill((mask == MASK_FALSE).unsqueeze(1), -1e9)
        return s
