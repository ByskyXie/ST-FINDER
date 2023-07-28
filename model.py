import torch
import torch.nn as nn
import random
import torchcde
from utils.embedding import PositionalEmbedding, TemporalEmbedding
from cc_attention.check3d import CrissCrossAttention3D
from partialconv.partialconv3d import PartialConv3d
from matplotlib import pyplot as plt


class ST_Finder(nn.Module):

    def get_predictor(self):
        self.predictor = nn.Sequential(
            nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=(self.input_matrix_num, 3, 3)
                      , padding=(0, 1, 1), bias=False),  # ...cks_num*4 4 is encoder_block conv kernel
            nn.Conv3d(self.output_matrix_num, self.output_matrix_num
                      , kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.ReLU()
        )

    # 初始化参数：Encoder,InterpolateDecoder,predictor
    def __init__(self, config, model_cfg):  # freq_of_minute: sampling number per hour
        super(ST_Finder, self).__init__()
        assert config.input_matrix_num % 2 == 0

        self.multi_head_channels = config.multi_head_channels
        self.input_matrix_num = config.input_matrix_num
        self.output_matrix_num = config.predict_matrix_num + 1  # the 1 denote the left middle input matrix
        self.in_channels = config.in_channels
        self.blocks_num = model_cfg.blocks_num
        self.matrix_row = model_cfg.matrix_row
        self.matrix_column = model_cfg.matrix_column

        self.encoder = Encoder(self.input_matrix_num, self.blocks_num, self.in_channels, model_cfg.encoder_block_kernels,
                               model_cfg.encoder_block_paddings, model_cfg.downsample_kernel, model_cfg.downsample_padding)
        decoder_in_channels = self.encoder.channel_list[-1]
        self.Inter_decoder = InterpolateDecoder(self.input_matrix_num, self.blocks_num, decoder_in_channels,
                                                model_cfg.decoder_block_kernels, model_cfg.decoder_block_paddings)

        self.cca1 = CrissCrossAttention3D()  # it should not use at encoder
        self.cca2 = CrissCrossAttention3D()
        self.cca3 = CrissCrossAttention3D()
        self.norm1 = torch.nn.LayerNorm(normalized_shape=[self.output_matrix_num, self.input_matrix_num, self.matrix_row, self.matrix_column])
        self.norm2 = torch.nn.LayerNorm(normalized_shape=[self.output_matrix_num, self.input_matrix_num, self.matrix_row, self.matrix_column])
        self.norm3 = torch.nn.LayerNorm(normalized_shape=[self.output_matrix_num, self.input_matrix_num, self.matrix_row, self.matrix_column])
        self.gap1_q, self.gap1_k, self.gap1_v = nn.Conv3d(self.in_channels, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.in_channels, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.in_channels, self.output_matrix_num, kernel_size=1)
        self.gap2_q, self.gap2_k, self.gap2_v = nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1)
        self.gap3_q, self.gap3_k, self.gap3_v = nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1), nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1)
        self.head_conv0 = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1)
        self.head_conv = nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1)
        self.head_conv2 = nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1)
        self.head_conv3 = nn.Conv3d(self.output_matrix_num, self.output_matrix_num, kernel_size=1)
        self.get_predictor()

        self.pos_embed = PositionalEmbedding(d_model=model_cfg.embed_dim)
        self.temporal_embed = TemporalEmbedding(d_model=model_cfg.embed_dim, freq='h', freq_of_minute=model_cfg.freq_of_minute)


    def forward(self, x, time_seq, target_time):  # [8, 1, 4, 23, 23] and [8, 4]
        res = self.encoder(x)
        out, skips = res[0], res[1:]

        # ST-Finder part
        out = self.Inter_decoder(out, skips)  # torch.Size([8, 1, 8, 12, 12])

        # add pos embedding
        out = self.head_conv0(out)  # torch.Size([8, 8, 8, 12, 12])
        temp_embed = self.temporal_embed(time_seq)  # torch.Size([8, 8, 144])
        temp_embed = temp_embed.reshape([temp_embed.shape[0], 1, self.input_matrix_num, self.matrix_row, self.matrix_column])

        tembed = self.temporal_embed(target_time).reshape([temp_embed.shape[0], 1, self.output_matrix_num, self.matrix_row, self.matrix_column])
        tembed = torch.unsqueeze(torch.squeeze(tembed), dim=2)

        # out += tembed+temp_embed
        q1, k1, v1 = self.gap1_q(out), self.gap1_k(out), self.gap1_v(out)
        out_ = out*self.norm1(self.cca1(q1+tembed+temp_embed, k1+tembed+temp_embed, v1))  # batchsize c height width depth
        out = self.head_conv(out_)

        # predict
        out = self.predictor(out)

        return out



class Encoder_block(nn.Module):
    def __init__(self, input_matrix_num, in_channels, block_kernels, block_paddings, ds_kernel, ds_padding):
        super(Encoder_block, self).__init__()
        self.input_matrix_num = input_matrix_num

        # Partial Conv
        self.base_conv1 = PartialConv3d(in_channels, 4 * in_channels, kernel_size=block_kernels[0], padding=block_paddings[0], bias=False, multi_channel=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.base_conv2 = PartialConv3d(4 * in_channels, 4 * in_channels, kernel_size=block_kernels[1], padding=block_paddings[1], bias=False, multi_channel=True)

        self.block_gating = nn.Sequential(
            nn.AdaptiveAvgPool3d((input_matrix_num, 1, 1)),
            nn.Conv3d(4*in_channels, 4*in_channels, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.downsample = PartialConv3d(in_channels, 4*in_channels, kernel_size=ds_kernel, padding=ds_padding, bias=False, multi_channel=True)

    def forward(self, x):
        xx = self.base_conv1(x, get_not_zero_position(x))  # torch.Size([8, 2, 8, 10, 10]) torch.Size([8, 2, 8, 10, 10])
        xx = self.relu1(xx)
        xx = self.base_conv2(xx, get_not_zero_position(xx))

        # print('Encode:', x.shape, xx.shape, self.downsample(x).shape)
        xx = xx + self.downsample(x, get_not_zero_position(x))

        residual = self.block_gating(xx)  # 尺寸一样直接相加
        return xx + residual

# 设计编码模块
class Encoder(nn.Module):

    def __init__(self, input_matrix_num, blocks_num, in_channels, block_kernels, block_paddings, ds_kernel, ds_padding):
        super(Encoder, self).__init__()
        self.input_matrix_num = input_matrix_num

        # create blocks
        channel_list = [in_channels]
        blocks = []
        for i in range(blocks_num):
            blocks.append(Encoder_block(input_matrix_num, channel_list[-1],
                                        block_kernels, block_paddings, ds_kernel, ds_padding))
            channel_list.append(channel_list[-1]*4)
        self.channel_list = channel_list

        assert len(blocks) > 0
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        self.skips = [x]  # residual skip connect
        for i in range(len(self.blocks) - 1):
            x = self.blocks[i](x)
            self.skips.append(x)

        res = [self.blocks[len(self.blocks) - 1](x)]
        res += self.skips
        return res


class Decoder_block(nn.Module):
    def __init__(self, input_matrix_num, in_channels, block_kernels, block_paddings):
        super(Decoder_block, self).__init__()
        self.input_matrix_num = input_matrix_num
        self.block_conv_T = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels//4, kernel_size=block_kernels[0], stride=1, padding=block_paddings[0], output_padding=0, bias=False),
            nn.ConvTranspose3d(in_channels//4, in_channels//4, kernel_size=block_kernels[1], stride=1, padding=block_paddings[1], output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//4, in_channels//4, kernel_size=block_kernels[2], padding=block_paddings[2], bias=False),
        )
        self.block_gating = nn.Sequential(
            nn.AdaptiveAvgPool3d((input_matrix_num, 1, 1)),
            nn.Conv3d(in_channels//4, in_channels//4, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )


    def forward(self, x, skip=None):
        # print('DeConv:', x.shape, self.block_conv_T(x).shape)
        x = self.block_conv_T(x)  # 尺寸放大 torch.Size([8, 3, 16, 11, 11])

        xx = self.block_gating(x) + x
        if skip is not None:
            # print(xx.shape, skip.shape)
            assert xx.shape == skip.shape
            xx += skip
        return xx


class InterpolateDecoder(nn.Module):
    def __init__(self, input_matrix_num, blocks_num, in_channels, block_kernels, block_paddings):
        super(InterpolateDecoder, self).__init__()
        self.input_matrix_num = input_matrix_num
        self.residual_weight = PartialConv3d(1, 1, kernel_size=1)

        # create blocks
        channel_list = [in_channels]
        blocks = []
        for i in range(blocks_num, 0, -1):
            blocks.append(Decoder_block(input_matrix_num, channel_list[-1], block_kernels, block_paddings))
            channel_list.append(channel_list[-1]//4)
        self.channel_list = channel_list

        assert len(blocks) > 0
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        # TODO: assert len(skips) == len(self.blocks) - 1
        skips = skips[:: -1]  # 反向便于操作[1,2,3]->[3,2,1]

        for i in range(0, len(self.blocks)-1):
            skip = skips[i]
            x = self.blocks[i](x, skip)
        x = self.blocks[-1](x)

        return x+self.residual_weight(skips[-1], get_not_zero_position(skips[-1]))

def get_not_zero_position(inputs):
    return torch.clamp(torch.clamp(torch.abs(inputs), 0, 1e-32) * 1e36, 0, 1).detach()


if __name__ == '__main__':
    pass


