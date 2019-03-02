import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import pandas as pd
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
        # self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1,
        #                      bidirectional=True, batch_first=True)

        self.rnn = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                          batch_first=True)

        # self.seq_hidden2vec = nn.Linear(self.hidden_dim * 2, self.emb_dim)

        self.out2tag = nn.Linear(self.hidden_dim*2 , 2)

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

    def forward(self, input_ch1, input_ch2):

        rnn_out1 = self.encoder(input_ch1)
        rnn_out2 = self.encoder(input_ch2)
        # rnn_out1 = rnn_out1.transpose(1, 0).contiguous().reshape(rnn_out1.size(1), -1)
        # rnn_out2 = rnn_out2.transpose(1, 0).contiguous().reshape(rnn_out2.size(1), -1)
        # out = rnn_out1 + rnn_out2
        out = torch.cat([rnn_out1, rnn_out2], 1)
        return self.out2tag_logsoftmax(out)

    def encoder(self, input_ch):
        rnninput = torch.zeros(1 * 2, input_ch.size(0), self.hidden_dim).cuda()
        input_x = self.ch_Embedding(input_ch)
        input_x = F.dropout(input_x, p=self.drop_out_p, training=self.drop_out_flag)
        # output, hn = self.rnn(input_x, rnninput)
        hn=torch.mean(input_x,1)


        return hn
        # return input_x

    def out2tag_logsoftmax(self, out):
        out = F.dropout(out, p=self.drop_out_p, training=self.drop_out_flag)
        out = F.leaky_relu(self.out2tag(out))
        out = F.log_softmax(out, dim=1)
        return out
