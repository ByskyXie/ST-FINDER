import torch
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model%2==0 else torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # torch.Size([1, 5000, 512])
        self.register_buffer('pe', pe)

    def forward(self, x):  # torch.Size([1, 96, 512])
        # print(x.shape, self.pe.shape, self.pe[:, :x.size(1)].shape)  # [8, 1, 8, 12, 12] [1, 5000, 144] [1, 1, 144]
        return self.pe[:, :x.size(1)]



class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='h', freq_of_minute=15):
        super(TemporalEmbedding, self).__init__()
        self.d_model = d_model

        minute_size = int(60 / freq_of_minute)  # minute=4 cause sampling per 15 minute
        hour_size = 24;
        weekday_size = 7;
        day_size = 32;
        month_size = 13

        Embed = nn.Embedding
        if freq == 't':  # [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):  # x_mark: torch.Size([32, 96, 4])  batch, all_seq_len, y-m-d-t
        # x = x.long()
        #
        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:, :, 3])
        # weekday_x = self.weekday_embed(x[:, :, 2])
        # day_x = self.day_embed(x[:, :, 1])
        # month_x = self.month_embed(x[:, :, 0])
        #
        # # print(x.shape, (hour_x + weekday_x + day_x + month_x + minute_x).shape)  # [8, 8, 4] [8, 8, 144]
        # return hour_x + weekday_x + day_x + month_x + minute_x
        x_shape = list(x.unsqueeze(-1).shape)  # [8, 8, 1]
        x_shape[-1] = self.d_model  # [8, 8, d_model]
        x = x.view(-1)  # [64]

        pe = torch.zeros(len(x), self.d_model).float().to(torch.device('cuda:0'))  # torch.Size([64, 144])
        pe.require_grad = False

        position = x.float().unsqueeze(1)
        div_term = (torch.arange(0, self.d_model, 2).float() * -(math.log(1000000.0) / self.d_model)).exp().to(
            torch.device('cuda:0'))  # period=1M

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if self.d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.view(x_shape)

        return pe

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [batch, long, head, embed] [batch, s, head, embed]
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

