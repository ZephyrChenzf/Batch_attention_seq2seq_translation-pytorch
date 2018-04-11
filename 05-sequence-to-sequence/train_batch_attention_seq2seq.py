import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.dataset import TextDataset
from Batch_Attention_Seq2seq.attentionRNN import EncoderRNN, AttnDecoderRNN

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_cuda = torch.cuda.is_available()

SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10
lang_dataset = TextDataset()
lang_dataloader = DataLoader(lang_dataset, batch_size=16, shuffle=True)

in_len_dic = lang_dataset.input_lang_words
out_len_dic = lang_dataset.output_lang_words
emb_dim = 256
hidden_size = 256
num_epoches = 20
batch_size = 16

if use_cuda:
    encoderModel = EncoderRNN(in_len_dic, emb_dim, hidden_size).cuda()
    attentionDecoderModel = AttnDecoderRNN(hidden_size, emb_dim, out_len_dic).cuda()
else:
    encoderModel = EncoderRNN(in_len_dic, emb_dim, hidden_size)
    attentionDecoderModel = AttnDecoderRNN(hidden_size, emb_dim, out_len_dic)

param = list(encoderModel.parameters()) + list(attentionDecoderModel.parameters())
# criterion = nn.NLLLoss()#自定义了一个
optimzier = optim.Adam(param, lr=1e-3)

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    for i, data in enumerate(lang_dataloader):
        x, y , mask = data
        if use_cuda:
            x = Variable(x).cuda()  # b,t
            y = Variable(y).cuda()
            mask=Variable(mask).cuda()
        else:
            x = Variable(x)
            y = Variable(y)
            mask=Variable(mask)
        # 处理输入，使其按非0序列长度降序排序
        sort_value, sort_index = torch.sort(torch.sum(x != 0, 1), descending=True)
        x = x[sort_index]
        y = y[sort_index]
        mask=mask[sort_index]

        encoder_outputs, encoder_hidden = encoderModel(x, sort_value)
        decoder_input = Variable(torch.LongTensor([[SOS_token]*len(y)]))  # 注意转化为矩阵
        if use_cuda:
            decoder_input = decoder_input.cuda()
        decoder_hidden = encoder_hidden
        loss = 0
        acc_t = 0
        for di in range(y.size(1)):
            y_=y[:,di].contiguous().view(-1,1)
            mask_=mask[:,di].contiguous().view(-1,1)
            decoder_output,decoder_hidden=attentionDecoderModel(decoder_input,decoder_hidden,encoder_outputs)
            # loss_not_mask=criterion(decoder_output, y[:, di])#未mask损失
            loss_not_mask=attentionDecoderModel.my_batch_nllloss(decoder_output,y_)#未mask损失
            loss_mask=loss_not_mask*mask_.float()
            loss += loss_mask
            max_value,max_index=torch.max(decoder_output,1)  # 取最大可能的一个,B
            #准确次数
            #注意此时mask_，TOT
            if (mask_==0).data.sum()==len(mask_):
                break
            y_pre=torch.masked_select(max_index.view(-1,1),mask_)
            y_true=torch.masked_select(y_,mask_)
            acc_t+=(y_pre==y_true).data.sum()/len(y_pre)#加上每个序列的batch准确度

            ni = max_index.view(1,-1)
            decoder_input = ni  # 前面的输出作为后面的输入
        loss=torch.sum(loss)/len(y)
        # backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        train_loss += loss.data[0]
        train_acc += acc_t/(len(y))
        if (i + 1) % 10 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(lang_dataloader),
                                                                            train_loss / (i +1),
                                                                            train_acc / (i +1)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(lang_dataloader)),
                                                                     train_acc / (len(lang_dataloader))))

torch.save(EncoderRNN.state_dict(), './model/encoder_model.pth')
torch.save(AttnDecoderRNN.state_dict(), './model/decoder_model.pth')
