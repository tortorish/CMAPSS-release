from model import *
from thop import profile
from utils.functions import count_parameters


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


encoder = Seq2SeqEncoder(input_size=14, num_layers=2, num_hiddem=8)
decoder = Seq2SeqDecoder(input_size=14, num_layers=2, num_hidden=8, seq_len=30, attention_size=28)
model = EncoderDecoder(encoder=encoder, decoder=decoder, feature_attention_size=4)

encoder_x = torch.ones((128, 30, 14))

print('parameters_count:', count_parameters(model))
