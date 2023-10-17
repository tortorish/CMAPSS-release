import torch
from torch import nn


class AdditiveAttentionForSeq(nn.Module):
    def __init__(self, num_hidden, attention_size, seq_len):
        super(AdditiveAttentionForSeq, self).__init__()
        self.attention = nn.Linear(attention_size, 1)
        self.linear1 = nn.Linear(num_hidden, attention_size, bias=False)
        self.linear2 = nn.Linear(num_hidden, attention_size, bias=False)
        self.seq_len = seq_len
        self.attention_weight = 0


    def forward(self, encoder_output, encoder_state):
        average_state = (torch.mean(encoder_state, dim=0)).unsqueeze(1)  # (bs,1,n_h)
        average_state = average_state.repeat(1, self.seq_len, 1)  # (bs,30,n_h)
        average_state = self.linear1(average_state)  # (bs,30,a_s)
        keys = self.linear2(encoder_output)  # (bs,30,a_s)
        added = average_state + keys  # (bs,30,a_s)
        scores = torch.tanh(added)  # (bs,30,a_s)
        scores = (self.attention(scores)).squeeze(-1)  # (bs,30)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        self.attention_weight = attention_weights
        return output, attention_weights


class SelfConcatAttentionForSeq(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfConcatAttentionForSeq, self).__init__()
        self.attention = nn.Linear(input_size*2, attention_size)
        self.linear = nn.Linear(attention_size, 1, bias=False)
        self.attention_weights = 0
    def forward(self, key, query, values):  # (bs, 30, 14)
        concat = torch.cat((key, query), dim=-1)  # (bs,30,28)
        scores = torch.tanh(self.attention(concat))  # (bs,30,20)
        scores = (self.linear(scores)).squeeze(-1)  # (bs,30)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights.unsqueeze(1), values)
        self.attention_weights = attention_weights
        #self.sum_attention_weights += attention_weights
        return output, attention_weights

