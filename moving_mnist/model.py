import torch
from torch import nn
from stackrnn import StackRNN

class Model(nn.Module):
    def __init__(self, in_shape, num_layers, num_hidden, filter_size, stride, layer_norm, patch_size):
        super(Model, self).__init__()
        self.stackrnn = StackRNN(in_shape=in_shape,
                                 num_layers=num_layers,
                                 num_hidden=num_hidden,
                                 filter_size=filter_size,
                                 stride=stride,
                                 layer_norm=layer_norm,
                                 patch_size=patch_size)

    def forward(self, x, delta_c_list, delta_m_list, memory, h_t, c_t):
        y, delta_c_list, delta_m_list, memory, h_t, c_t = self.stackrnn(x, delta_c_list, delta_m_list, memory, h_t, c_t)
        return y, delta_c_list, delta_m_list, memory, h_t, c_t
