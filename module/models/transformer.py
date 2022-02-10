import math

from torch import nn
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, input_dim, enc_num_layers=6, dec_num_layers=6, n_head=8, dropout=0.2, out_dim=1,
                 length_prediction=24, dec_head=1, enc_length=240):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_head,
                                                                        dropout=dropout,
                                                                        dim_feedforward=4*input_dim),
                                             num_layers=enc_num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=out_dim, nhead=dec_head,
                                                                        dim_feedforward=4*dec_head,
                                                                        dropout=dropout),
                                             num_layers=dec_num_layers)
        self.length_prediction = length_prediction
        self.fc = nn.Linear(enc_length*input_dim, length_prediction*out_dim)
        self.out_fc = nn.Linear(length_prediction * out_dim, length_prediction)

    def forward(self, x, y):
        x = self.pos_encoder(x.transpose(0, 1))
        x = self.encoder(x)
        x = self.fc(x.transpose(0, 1).flatten(start_dim=1)).transpose(0, 1).unsqueeze(-1)
        y = self.decoder(y.transpose(0, 1), x)
        y = y.transpose(0, 1)
        y = self.out_fc(y.flatten(start_dim=1))
        return y.unsqueeze(-1)


if __name__ == '__main__':
    import numpy as np

    a = torch.from_numpy(np.random.uniform(-1., 1., size=(128, 240, 14))).float()
    b = torch.from_numpy(np.random.uniform(-1., 1., size=(128, 24, 1))).float()
    m = Transformer(input_dim=14, enc_num_layers=6, dec_num_layers=6, n_head=7, out_dim=1)(a, b)
    print(m.shape)
