import torch
from torch import nn
from stcell import STCell
from cell import BasicConv2d
import torch.nn.functional as F

class StackRNN(nn.Module):
    def __init__(self, in_shape, num_layers, num_hidden, filter_size, stride, layer_norm, patch_size):
        super(StackRNN, self).__init__()

        _, C, H, W = in_shape
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.channel = patch_size * patch_size * C
        self.height = H // patch_size
        self.width = W // patch_size


        cell_list = []

        for i in range(self.num_layers):
            in_channel = self.channel if i==0 else self.num_hidden
            cell_list.append(
                STCell(in_channel=in_channel,
                       num_hidden=self.num_hidden,
                       height=self.height,
                       width=self.width,
                       filter_size=self.filter_size,
                       stride=self.stride,
                       layer_norm=self.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.merge = BasicConv2d(in_channels=self.num_hidden*2,
                                 out_channels=self.num_hidden,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.conv_last = nn.Conv2d(in_channels=self.num_hidden,
                                     out_channels=self.channel,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.adapter = nn.Conv2d(in_channels=self.num_hidden,
                                 out_channels=self.num_hidden,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

    def forward(self, in_tensor, delta_c_list, delta_m_list, memory, h_t, c_t):

        h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](in_tensor, h_t[0], c_t[0], memory)
        copy = h_t[0]

        delta_c_list[0] = F.normalize(
            self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
        )
        delta_m_list[0] = F.normalize(
            self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
        )

        for i in range(1, self.num_layers):
            h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)
            delta_c_list[i] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
            )
            delta_m_list[i] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
            )

        concat = torch.cat((copy, h_t[self.num_layers-1]), dim=1)
        concat_out = self.merge(concat)

        out_tensor = self.conv_last(concat_out)

        return out_tensor, delta_c_list, delta_m_list, memory, h_t, c_t

