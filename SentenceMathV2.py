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
        self.drop_out_p = 0.1
        self.drop_out_flag = drop_out_flag
        self.max_seq_len = 120
        self.hidden_dim = hidden_dim

        self.ch_Embedding = nn.Embedding(len(vacb_ch) + 1, emb_dim)
        self.ch_Embedding.weight = nn.Parameter(torch.FloatTensor(pickle.load(open("data/embed.pkl", 'rb'))))

        self.embencoder = Embedding()

        # self.attention = ScaledDotProductAttention(emb_dim)

        # self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1,
        #                      bidirectional=True, batch_first=True)
        # self.rnns = resconnetc(3, emb_dim, hidden_dim, drop_out_flag, self.drop_out_p)
        self.rnn1 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)

        self.rnn2 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.rnn3 = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)

        self.mhattn = MultiHeadAttention(2, 200, 200, 200)
        D = 200
        L_q = 120
        self.cqattn = CQAttention(D, L_q, L_q, self.drop_out_p)

        self.out2tag = nn.Linear(self.hidden_dim * 2, 2)
        # self.out2tag = nn.Linear(120, 2)
        self.init_weigths()
    def init_weigths(self):
        for weights in self.rnn1.all_weights:
            for weight in weights:
                weight.data.uniform_(-0.1, 0.1)

        # self.ch_Embedding.weight.data.uniform_(-0.1, 0.1)

        # self.seq_hidden2vec.weight.data.uniform_(-0.1, 0.1)
        # self.seq_hidden2vec.bias.data.zero_()
        #
        self.out2tag.weight.data.uniform_(-0.1, 0.1)
        self.out2tag.bias.data.zero_()

    def encoder(self, input_ch, mask, word_ch):
        # rnninput = torch.zeros(1 * 2, input_ch.size(0), self.hidden_dim).cuda()
        input_ch = self.ch_Embedding(input_ch)
        input_word_ch = self.ch_Embedding(word_ch)
        input_ch = F.dropout(input_ch, p=self.drop_out_p, training=self.drop_out_flag)
        input_word_ch = F.dropout(input_word_ch, p=self.drop_out_p, training=self.drop_out_flag)
        # hn = self.rnns(input_x)
        # input_x = self.embencoder(input_word_ch, input_ch)

        input_x=input_ch

        output, hn = self.rnn1(input_x)
        # output, hn = self.rnn2(output + input_x)

        # input_x1 = output
        # input_x1 = self.mhattn(input_x1, input_x1, input_x1, mask.reshape(mask.size(0), 1, -1).byte())
        # input_x1 = input_x1[0]
        # output, hn = self.rnn3(input_x1 + input_x)

        return hn
        # return output
    def forward(self, input_ch1, mask1, input_ch2, mask2, word_ch_a, word_ch_b):

        rnn_out1 = self.encoder(input_ch1, mask1, word_ch_a)
        rnn_out2 = self.encoder(input_ch2, mask2, word_ch_b)
        rnn_out1 = rnn_out1.transpose(1, 0).contiguous().reshape(rnn_out1.size(1), -1)
        rnn_out2 = rnn_out2.transpose(1, 0).contiguous().reshape(rnn_out2.size(1), -1)

        out = rnn_out1 * rnn_out2
        # out = self.cqattn(rnn_out1, rnn_out2, mask1, mask2)
        # out = out.transpose(1, 2).max(2)[0]
        return self.out2tag_logsoftmax(out)

    def out2tag_logsoftmax(self, out):
        out = F.dropout(out, p=self.drop_out_p, training=self.drop_out_flag)
        out = F.leaky_relu(self.out2tag(out))
        out = F.log_softmax(out, dim=1)
        return out

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        Dchar = 120
        D = 120
        Dword = 120
        self.conv2d = nn.Conv2d(Dchar, D, kernel_size=(1, 2), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(Dword + D, D, bias=False)
        self.high = Highway(2,  Dword)

    def forward(self, ch_emb, wd_emb):
        N = ch_emb.size()[0]  # 64*120*10*200
        ch_emb = ch_emb.permute(0, 1, 3, 2)  # 16*120*200*10
        ch_emb = F.dropout(ch_emb, p=0.1, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)  # 16*120*200*9
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()  # 16*120*200

        wd_emb = F.dropout(wd_emb, p=0.1, training=self.training)
        # wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        #         return q,k,v
        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))  # n个head view 到一起后再线性变换到原始维度。
        output = self.layer_norm(output + residual)  # 加殘差。

        return output, attn

def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)

class CQAttention(nn.Module):
    def __init__(self, D, Lc, Lq, dropout):
        super().__init__()
        self.dropout = dropout
        self.Lc = Lc
        self.Lq = Lq
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        # C = C.transpose(1, 2)
        # Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, self.Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, self.Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        #         print(C.shape,Q.shape)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, self.Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, self.Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        #         print(subres0.shape,subres1.shape,subres2.shape)
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

class Highway(nn.Module):
    def __init__(self, layer_num: int, size, dropout=0.1):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = dropout

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x
