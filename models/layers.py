import torch 
import torch.nn as nn
import torch.nn.functional as F 


class BlockDiagonalLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=4,bias=True):
        super(BlockDiagonalLinear, self).__init__()

        assert input_size % num_blocks == 0, "Input size must be divisible by the number of blocks"
        assert output_size % num_blocks == 0, "Output size must be divisible by the number of blocks"

        self.num_blocks = num_blocks
        self.block_in_size = input_size // num_blocks  # Input size of each block
        self.block_out_size = output_size // num_blocks  # Output size of each block

        # Create weight matrices for each block and bias
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(self.block_out_size, self.block_in_size)) 
                                         for _ in range(num_blocks)])
        self.need_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        block_diag = torch.block_diag(*self.weights)
        out = torch.matmul(x, block_diag) 
        if self.need_bias:
            out = out + self.bias
        return out

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Causal padding: we pad only on the left side so that the convolution doesn't look ahead
        self.padding = (kernel_size - 1) * dilation
        # Causal convolution layer
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # Apply convolution and then trim the extra padding on the right (causal behavior)
        out = self.conv1d(x)
        return out[:, :, :-self.padding]  # Trimming the output to maintain causality

class SwishActivation(nn.Module):
    def __init__(self, beta =1.0, learnable=False):
        super(SwishActivation, self).__init__()
        # Define a learnable parameter if `learnable` is True
        if learnable:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            # Use a fixed parameter if not learnable
            self.beta = beta

    def forward(self, x):

        return x * torch.sigmoid(self.beta * x)