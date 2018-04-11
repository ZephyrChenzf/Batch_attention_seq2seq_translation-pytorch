import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        input_seqs: B,T
        hidden: initial state of GRU
        '''
        input_seqs=input_seqs.permute(1,0)
        embedded = self.embedding(input_seqs)#T,B,H
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.data.tolist())
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # T,B,H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0., std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        hidden: layers*directions,B,H
        encoder_outputs: T,B,H
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)#T,B,H->B,T,H
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score#B,T
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax#B,1,T

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]采用连接的方法
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size,embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p,bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        word_input:  B,1
        last_hidden: layers*direction,B,H
        encoder_outputs: T,B,H
        '''
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)解码器输入单个
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)#B,1,T
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,H)
        context = context.transpose(0, 1)  # (1,B,H)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)#1,B,V+H
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,H*2)->(B,H*2)
        output = F.log_softmax(self.out(output))#B,V
        # Return final output, hidden state
        return output, hidden

    def my_batch_nllloss(self,input,target):
        batch_loss=-torch.gather(input,1,target)
        return batch_loss
