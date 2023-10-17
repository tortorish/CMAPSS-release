from .Attention_modules import *


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, num_layers, num_hiddem):
        super(Seq2SeqEncoder, self).__init__()
        self.nn_lstm = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=num_hiddem, batch_first=True)
        self.lstm = self.nn_lstm

    def forward(self, x):
        output = self.lstm(x)
        return output


class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size, num_layers, num_hidden, seq_len, attention_size):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size+num_hidden,
                                  num_layers=num_layers,
                                  hidden_size=num_hidden,
                                  batch_first=True)
        self.attention = AdditiveAttentionForSeq(num_hidden=num_hidden,
                                                 attention_size=attention_size,
                                                 seq_len=seq_len)
        self.Linear = nn.Linear(num_hidden, 1)

    def forward(self, decoder_x, encoder_output, encoder_state):
        output, attention_weights = self.attention(encoder_output, encoder_state[0])  # encoder_state[0]表示h
        output = torch.cat((output, decoder_x), dim=-1)
        output, _ = self.lstm(output, encoder_state)
        output = output.squeeze(1)
        output = self.Linear(output)
        return output, attention_weights


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, feature_attention_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_attention = SelfConcatAttentionForSeq(input_size=14, attention_size=4)

    def forward(self, encoder_x):
        encoder_output, encoder_state = self.encoder(encoder_x)
        decoder_x, attention_weight_feature = self.feature_attention(encoder_x, encoder_x, encoder_x)
        output, attention_weights = self.decoder(decoder_x, encoder_output, encoder_state)
        return output, attention_weights
