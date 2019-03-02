import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import pickle

class SentenceMath(nn.Module):
    def __init__(self, emb_dim, vacb_ch, hidden_dim, drop_out_flag=True):
        super(SentenceMath, self).__init__()
        self.emb_dim = emb_dim
        self.drop_out_p = 0.5
        self.drop_out_flag = drop_out_flag
        self.max_seq_len = 120
        self.hidden_dim = hidden_dim

        self.ch_Embedding = nn.Embedding(len(vacb_ch) + 1, emb_dim)
        self.ch_Embedding.weight = nn.Parameter(torch.FloatTensor(pickle.load(open("data/embed.pkl", 'rb'))))
        self.attention=ScaledDotProductAttention(emb_dim)

        # self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1,
        #                      bidirectional=True, batch_first=True)
        # self.rnns = resconnetc(3, emb_dim, hidden_dim, drop_out_flag, self.drop_out_p)
        self.rnn1 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)

        self.rnn2 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.rnn3 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)

        # self.seq_hidden2vec = nn.Linear(self.hidden_dim * 2, self.emb_dim)

        self.out2tag = nn.Linear(self.hidden_dim * 2, 2)

    def init_weigths(self):
        for weights in self.rnn.all_weights:
            for weight in weights:
                weight.data.uniform_(-0.1, 0.1)

        self.ch_Embedding.weight.data.uniform_(-0.1, 0.1)

        # self.seq_hidden2vec.weight.data.uniform_(-0.1, 0.1)
        # self.seq_hidden2vec.bias.data.zero_()
        #
        self.out2tag.weight.data.uniform_(-0.1, 0.1)
        self.out2tag.bias.data.zero_()

    def encoder(self, input_ch):
        rnninput = torch.zeros(1 * 2, input_ch.size(0), self.hidden_dim).cuda()
        input_x = self.ch_Embedding(input_ch)
        input_x = F.dropout(input_x, p=self.drop_out_p, training=self.drop_out_flag)
        # hn = self.rnns(input_x)
        output, hn = self.rnn1(input_x, rnninput)
        # output, hn = self.rnn2(output + input_x, rnninput)
        # output=self.attention(output,output,output)[0]
        # output, hn = self.rnn3(output + input_x, rnninput)
        return hn
        # return input_x
    def forward(self, input_ch1, input_ch2):

        rnn_out1 = self.encoder(input_ch1)
        rnn_out2 = self.encoder(input_ch2)
        rnn_out1 = rnn_out1.transpose(1, 0).contiguous().reshape(rnn_out1.size(1), -1)
        rnn_out2 = rnn_out2.transpose(1, 0).contiguous().reshape(rnn_out2.size(1), -1)
        out = rnn_out1 * rnn_out2
        # out = torch.cat([rnn_out1, rnn_out2], 1)
        return self.out2tag_logsoftmax(out)

    def out2tag_logsoftmax(self, out):
        out = F.dropout(out, p=self.drop_out_p, training=self.drop_out_flag)
        out = F.leaky_relu(self.out2tag(out))
        out = F.log_softmax(out, dim=1)
        return out

def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nh = 1
        self.D = 200

    def forward(self, queries):
        memory = queries
        query = queries
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.Nh)
        K, V = [self.split_last_dim(tensor, self.Nh) for tensor in torch.split(memory, self.D, dim=2)]

        key_depth_per_head = self.D // self.Nh
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=0.5, training=self.training)
        return torch.matmul(weights, v)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=0.5, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)
    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret
class resconnetc(nn.Module):
    def __init__(self, num_layers, emb_dim, hidden_dim, training=False, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.training = training
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.rnns = nn.ModuleList(
            [nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                    batch_first=True) for _ in range(self.num_layers)])
    def forward(self, x):
        # x: shape [batch_size,length, input_size, ]
        hn = 0
        for i in range(self.num_layers):
            rnninput = torch.zeros(1 * 2, x.size(0), self.hidden_dim).cuda()
            outs, hns = self.rnns[i](x, rnninput)
            outs = F.dropout(outs, p=self.dropout, training=self.training)
            x += outs
            hn += hns
        return hn
