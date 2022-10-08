import pickle
import timeit
import numpy as np
from math import sqrt
import math
import torch
import torch.optim as optim
import os
from torch import nn, einsum
from torch.nn import Parameter
# from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.nn import init


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(5)

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * self.softmax(x) + x

class spatt(nn.Module):
    def __init__(self, padding = 3):
        super(spatt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(2*padding+1),padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xc = x.unsqueeze(1)
        avg = torch.mean(xc, dim=1, keepdim=True)
        max_x, _ = torch.max(xc, dim=1, keepdim=True)
        xt = torch.cat((avg,max_x),dim=1)
        att = self.sigmoid(self.conv1(xt))
        # print(att.squeeze(1).shape)
        return x * (att.squeeze(1))


class CNN_MLP(nn.Module):
    def __init__(self, Affine, patch, channel, output_size, dr, down=False, last=False):
        super(CNN_MLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.cross_patch_linear0 = nn.Linear(patch, patch)
        self.cross_patch_linear1 = nn.Linear(patch, patch)
        self.cross_patch_linear = nn.Linear(patch, patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.cnn1 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=15, padding=7, groups=patch)
        self.bn1 = nn.BatchNorm1d(patch)
        self.cnn2 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=31, padding=15, groups=patch)
        self.bn2 = nn.BatchNorm1d(patch)
        self.cnn3 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
        self.bn3 = nn.BatchNorm1d(patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.self_attention = self_attention(channel)
        self.bnp1 = nn.BatchNorm1d(channel)

        self.cross_channel_linear1 = nn.Linear(channel, channel)
        self.cross_channel_linear2 = nn.Linear(channel, channel)

        self.att = spatt(3)
        self.att_sp = spatial_attention(3)
        # self.att = ExternalAtt(patch, channel)
        self.attention_channel_linear2 = nn.Linear(channel, channel)
        self.last_linear = nn.Linear(channel, output_size)
        self.bnp = nn.BatchNorm1d(patch)
        self.act = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(0.05)
        self.down = down

    def forward(self, x):
        # print(x.shape)
        x_cp = self.Afine_p1(x).permute(0, 2, 1)
        x_cp = self.act(self.cross_patch_linear0(x_cp))
        x_cp = self.act(self.cross_patch_linear1(x_cp))
        x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
        x_cc = x + self.Afine_p2(x_cp)
        x_cc2 = self.Afine_p3(x_cc)
        x_cc2 = self.act(self.bn1(self.cnn1(x_cc2)))
        x_cc2 = self.act(self.bn2(self.cnn2(x_cc2)))
        x_cc2 = self.act(self.bn3(self.cnn3(x_cc2)))
        x_cc2 = self.Afine_p4(x_cc2)
        # x_cc2 = self.act(x_cc2)
        x_cc2 = self.att(x_cc2)
        # print(atten)
        x_out = x_cc + self.dropout(x_cc2)
        # x_out = x_cc + atten * x_cc2
        if self.last == True:
            x_out = self.last_linear(x_out)
        return x_out


class ResMLP(nn.Module):
    def __init__(self, Affine, patch, channel, output_size,dr, down=False, last=False):
        super(ResMLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.cross_patch_linear0 = nn.Linear(patch, patch)
        self.cross_patch_linear1 = nn.Linear(patch, patch)
        self.cross_patch_linear = nn.Linear(patch, patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.bnp1 = nn.BatchNorm1d(channel)

        self.cross_channel_linear1 = nn.Linear(channel, channel)
        self.cross_channel_linear2 = nn.Linear(channel, channel)

        self.cross_channel_linear1_down = nn.Linear(channel, channel//5)
        self.cross_channel_linear2_down = nn.Linear(channel//5, channel)

        self.att = spatt(3)
        # self.att = ExternalAtt(patch, channel)
        self.attention_channel_linear2 = nn.Linear(channel, channel)
        self.last_linear = nn.Linear(channel, output_size)
        self.bnp = nn.BatchNorm1d(patch)
        self.activation = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(dr)
        self.down = down

    def forward(self, x):
        x_cp = self.Afine_p1(x).permute(0, 2, 1)
        x_cp = self.activation(self.cross_patch_linear0(x_cp))
        x_cp = self.activation(self.cross_patch_linear1(x_cp))
        x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
        x_cc = x + self.Afine_p2(x_cp)
        x_cc2 = self.Afine_p3(x_cc)

        x_cc2 = self.activation(self.cross_channel_linear1(x_cc2))
        x_cc2 = self.cross_channel_linear2(x_cc2)
        x_cc2 = self.Afine_p4(x_cc2)
        x_cc2 = self.activation(x_cc2)
        x_cc2 = self.att(x_cc2)
        # x_cc2 = self.dropout(x_cc2)

        # atten = self.attention_channel_linear2(x_cc2)
        # atten = self.bnp(atten)

        x_out = x_cc + self.dropout(x_cc2)
        # x_out = x_cc + atten * x_cc2
        if self.last == True:
            x_out = self.last_linear(x_out)
        return x_out

class WingLoss(nn.Module):
    def __init__(self, om=10, ep=2):
        super(WingLoss, self).__init__()
        self.om = om
        self.ep = ep

    def forward(self, pred, tar):
        y = tar
        y_hat = pred
        de_y = (y - y_hat).abs()
        de_y1 = de_y[de_y < self.om]
        de_y2 = de_y[de_y >= self.om]
        loss1 = self.om * torch.log(1 + de_y1 / self.ep)
        C = self.om - self.om * math.log(1 + self.om / self.ep)
        loss2 = de_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class RWingLoss(nn.Module):
    def __init__(self, om=10, ep=2, r=0.5):
        super(RWingLoss, self).__init__()
        self.om = om
        self.ep = ep
        self.r = r

    def forward(self, pred, tar):
        y = tar
        y_hat = pred
        de_y = (y - y_hat).abs()
        de_y_0 = de_y[de_y < self.r]
        de_y0 = de_y[de_y >= self.r]
        de_y1 = de_y0[de_y0 < self.om]
        de_y2 = de_y[de_y >= self.om]
        loss0 = 0 * de_y_0
        loss1 = self.om * torch.log(1 + (de_y1-self.r) / self.ep)
        C = self.om - self.om * math.log(1 + (self.om-self.r) / self.ep)
        loss2 = de_y2 - C
        return (loss0.sum() + loss1.sum() + loss2.sum()) / (len(loss0) + len(loss1) + len(loss2))

class atten(nn.Module):
    def __init__(self, padding=3):
        super(atten, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xc = x.unsqueeze(1)
        xt = torch.cat((xc, self.dropout(xc)), dim=1)
        att = self.sigmoid(self.conv1(xt))
        return att.squeeze(1)


class cross_att(nn.Module):
    def __init__(self, channel, length):
        super(cross_att, self).__init__()
        self.linear_channel1 = nn.Linear(channel, channel)
        self.linear_channel2 = nn.Linear(channel, channel)
        self.linear_length1 = nn.Linear(length, length)
        self.linear_length2 = nn.Linear(length, length)
        self.softmax = nn.Softmax(dim=-1)
        self.atten0 = AttentionLayer(FullAttention(None, 5, 0.1), 75, 5)
        # self.dropout = nn.Dropout(0.05)

    def forward(self, mtrA, mtrB):
        # cross_ab = self.atten0(mtrB, mtrA, mtrA, True)
        mtrAB = torch.cat((mtrA, mtrB), 1)
        mtrABs = self.linear_channel1(mtrAB)
        mtrABs = self.linear_length1(mtrABs.permute(0, 2, 1)).permute(0, 2, 1)
        att1 = self.linear_channel2(mtrABs)
        # att2 = self.linear_length2(mtrABs.permute(0, 2, 1)).permute(0, 2, 1)
        att = att1# * att2
        scale = att.size(1) ** -0.5
        attention = self.softmax(att * scale)
        res = mtrAB + mtrABs * attention
        # pdi = torch.cat((res, cross_ab), 1)
        return res

class self_attention(nn.Module):
    def __init__(self, channel):
        super(self_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        scale = K.size(-1) ** -0.5
        att = self.softmax(Q * scale)
        ys = att * K
        return ys

class linear_attention(nn.Module):
    def __init__(self, channel):
        super(linear_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        V = self.linear_V(xs)
        weight = torch.matmul(K.permute(0, 2, 1), V)
        scale = K.size(-1) ** -0.5
        attention = self.softmax(weight * scale)
        ys = torch.matmul(Q, attention)
        return ys

class channel_attention(nn.Module):
    def __init__(self, channel):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels=channel, out_channels=channel//15, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels=channel//15, out_channels=channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        avg_ = self.fc2(self.relu(self.fc1(self.avg_pool(xs))))
        max_ = self.fc2(self.relu(self.fc1(self.max_pool(xs))))
        out = avg_ + max_
        return self.sigmoid(out).expand_as(xs)

class se_attention(nn.Module):
    def __init__(self, inputs, reduction=5):
        super(se_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inputs, inputs//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inputs//reduction, inputs, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(2)
        att = self.fc(avg)
        return att

class spatial_attention(nn.Module):
    def __init__(self, padding=3):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.dropout = nn.Dropout(0.1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        xt = torch.cat((avg, max_x), dim=1)
        att = self.Sigmoid((self.conv1(xt)))
        return att.expand_as(x)

class cross_layers(nn.Module):
    def __init__(self, channel, length, attention_dropout=0.05):
        super(cross_layers, self).__init__()
        self.self_atten0 = self_attention(channel)
        self.self_atten1 = self_attention(channel)
        self.cross_atten1 = cross_att(channel, 1251)
        self.cross_atten2 = ResMLP(Affine, 1251, 75, 75, 0)
        self.dropout = nn.Dropout(attention_dropout)
        self.linear = nn.Linear(150, 150)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        # self.downsample2 = nn.Linear(1326, 1251)
        self.act = nn.ReLU()
        # self.chatt = channel_attention(75)

    def forward(self, pro, com):

        pro1 = self.self_atten0(pro)
        com1 = self.self_atten1(com)
        # pc_fac1 = torch.cat((pro1, com1), dim=1)
        # pc_fac1 = pc_fac1 + self.cross_atten2(pc_fac1)
        pc_fac = self.cross_atten1(pro1, com1)
        # chatt = self.chatt(pc_fac1)
        # pc_fac = pc_fac1 + pc_fac1 * chatt
        dta1 = torch.mean(pc_fac, dim=2)
        dta2 = torch.mean(pc_fac, dim=1)
        # dta = torch.cat((dta1, dta2), dim=1)

        return dta1, dta2



class Mix_Decoder(nn.Module):

    def __init__(self, channel, patch, attention_dropout=0.1):
        super(Mix_Decoder, self).__init__()
        self.cross_layers = cross_layers(channel, patch, attention_dropout)
        self.sample = nn.Linear(150, 75)
        self.atten0 = AttentionLayer(FullAttention(None, 7, 0.0), 75, 5)
        self.bn = nn.BatchNorm1d(1200)
        self.ln = nn.LayerNorm(75)
        self.self_atten0 = self_attention(channel)
        self.self_atten1 = self_attention(channel)
        self.sample1 = nn.Linear(51, 75)
        self.sample_a = nn.Linear(51, 51)
        self.sample2 = nn.Linear(1200,225)
        self.sample3 = nn.Linear(1200,300)#nn.MaxPool1d(kernel_size=4)
        # self.sample_pf = nn.MaxPool1d(kernel_size=4)
        # self.sample_pb = nn.MaxPool1d(kernel_size=6)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()


    def forward(self, protein_vector, compound_vector, adjacency):

        compound_vectors = self.sample1(compound_vector.permute(0, 2, 1)).permute(0, 2, 1)
        # print(com_kernal.shape)
        protein_vectors, com_kernals = protein_vector.unsqueeze(1).permute(1, 0, 2, 3), compound_vectors.unsqueeze(1).permute(0, 1, 2, 3)
        rm = F.conv2d(protein_vectors, com_kernals, padding=(37,0), groups=com_kernals.shape[0]).squeeze(0).squeeze(2)
        att = rm.unsqueeze(2).expand(protein_vector.size())
        protein_vector = protein_vector * att + protein_vector

        # scale = compound_vector.size(-1) ** -1
        # compound_reg = torch.sum(torch.matmul(adjacency,compound_vector) * scale, dim=2)
        compound_reg = torch.sum(adjacency, dim=1)
        # print(compound_reg)
        att2 = compound_reg.unsqueeze(2).expand(compound_vector.size())
        compound_vector = compound_vector * att2 + compound_vector

        dta1, dta2 = self.cross_layers(protein_vector, compound_vector)
        # protein_drug_interaction = PDI + self.atten0(PDI, Respose_protein, Respose_protein, None)

        return dta1, dta2, rm


class one_hot(nn.Module):
    def __init__(self, input_len, output_shape):
        super(one_hot, self).__init__()
        self.matrix = torch.eye(output_shape)
        self.len = input_len
        self.out = output_shape

    def forward(self, xs):
        out_put = torch.zeros((self.len, self.out), device=device)
        for i in range(self.len):
            out_put[i] = self.matrix[xs[i]]
        return out_put


class Predictor(nn.Module):
    def __init__(self, Decoder, ResMLP, Affine, dim, window, window2, layer_gnn, layer_cnn, layer_output):
        super(Predictor, self).__init__()
        # self.embed_fingerprint = nn.Embedding(34, dim)
        self.embed_comg = nn.Embedding(16, 44)
        self.embed_word = nn.Embedding(264, 100)
        self.embed_ss = nn.Embedding(25, 100)
        self.sequence_embedding = sequence_embedding()
        self.embed_com_word = nn.Embedding(65, 100)
        self.layer_gnn = layer_gnn
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.W_gat = nn.Parameter(torch.ones(50, 50))
        self.WP_NN1 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN3 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN4 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN2 = CNN_MLP(Affine, 1200, 100, 75, 0, True, True)

        self.WP_FiNN1 = CNN_MLP(Affine, 1200, 100, 75, 0,True)
        self.WP_FiNN3 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_FiNN4 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_FiNN2 = CNN_MLP(Affine, 1200, 100, 75, 0,True, True)

        self.WC_struc1 = ResMLP(Affine, 50, 75, 75, 0)
        self.WC_struc2 = ResMLP(Affine, 50, 75, 75, 0,False, True)  # 1

        self.WC_fcfp1 = ResMLP(Affine, 1, 2048, 75, 0)
        self.WC_fcfp2 = ResMLP(Affine, 1, 2048, 75, 0,False, True)


        self.WC_words1 = CNN_MLP(Affine, 100, 100, 75, 0)
        self.WC_words2 = CNN_MLP(Affine, 100, 100, 75, 0,False, True)

        self.WPDI_len1 = ResMLP(Affine, 75, 600, 100, 0)

        self.WP_fixfea = CNN_MLP(Affine, 75, 2400, 1200, 0,False, True)


        self.Wpdi_cnn = nn.ModuleList([nn.Conv1d(
            in_channels=75, out_channels=75, kernel_size=2 * window2 + 1,
            stride=1, padding=window2) for _ in range(2)])
        self.Transcnn = nn.Linear((2048 // 16), 75)
        # self.dcnn = nn.Linear(2500, 75)

        self.WC = nn.Linear(75, 75)
        self.WC2 = nn.Linear(75, 75)
        self.WW = nn.Linear(75, 75)
        self.WM = nn.Linear(75, 75)
        self.WF = nn.Linear(75, 75)
        self.WS = nn.Linear(75, 75)
        self.WC3 = nn.Linear(75, 75)
        self.WM3 = nn.Linear(75, 75)
        self.merge_atten = atten(3)
        self.merge_atten2 = atten(3)
        self.down_sample1 = nn.Linear(150, 75)
        self.down_sample2 = nn.Linear(150, 75)


        self.bn = nn.BatchNorm1d(75)
        self.ln = nn.LayerNorm(75)

        self.layer_output = layer_output
        # self.W_out0 = nn.Linear(100 * 75, 10*75)
        self.Decoder = Mix_Decoder(75, 5, attention_dropout=0.05)
        self.apha = Parameter(torch.tensor([0.5]))

        self.W_out1_1 = nn.Linear(1251, 1024)
        self.bn_out1 = nn.BatchNorm1d(1)#nn.LayerNorm(1024)
        # nn.init.kaiming_normal(self.W_out1_1.weight,mode='fan_in')
        self.W_out1_2 = nn.Linear(1024, 1024)
        self.bn_out2 = nn.BatchNorm1d(1)
        # nn.init.kaiming_normal(self.W_out1_2.weight, mode='fan_in')
        self.W_out1_3 = nn.Linear(1024, 512)
        self.bn_out3 = nn.BatchNorm1d(1)
        # nn.init.kaiming_normal(self.W_out1_3.weight, mode='fan_in')
        self.W_out2_1 = nn.Linear(75, 128)
        self.W_out2_2 = nn.Linear(128, 128)
        self.W_out2_3 = nn.Linear(128, 128)

        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout1_1 = nn.Dropout(p=0.2)
        self.dropout1_2 = nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.activation2 = nn.GELU()
        self.act_norm = Swish(False)
        self.act_norm2 = Swish(True)
        self.act_norm3 = Swish(True)
        self.act_norm4 = Swish(False)
        self.sigmoid = nn.Sigmoid
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.softmax3 = nn.Softmax(dim=1)

        self.W_interaction = nn.Linear(640, 640)
        self.W_interaction1 = nn.Linear(640, 1)
        # nn.init.xavier_normal_(self.W_interaction1.weight)
        # nn.init.kaiming_normal(self.W_interaction1.weight, mode='fan_in')
        self.W_interaction2 = nn.Linear(64, 1)


    def gnn(self, xs, A, layer):
        # print(torch.min(xs))
        for i in range(layer):
            hs = self.activation(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def attention_PM(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC(xs))
        h = self.activation(self.WM(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def attention_PM2(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC3(xs))
        h = self.activation(self.WM3(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def Elem_feature_Fusion_D(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample1(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten(x_c)
        xs_ = self.activation(self.WC2(xs))
        x_ = self.activation(self.WW(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def Elem_feature_Fusion_P(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample2(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten2(x_c)
        xs_ = self.activation(self.WF(xs))
        x_ = self.activation(self.WS(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def forward(self, inputs):

        fingerprints, compounds_words, adjacency, morgan, words, protein_ss = inputs
        # print(fingerprints[0].shape)
        N = fingerprints.shape[0]
        L = adjacency.shape[1]
        """Compound vector with GNN."""
        compound_vector = torch.zeros((N, 50, 75), device=device)
        compound_vec = torch.zeros((fingerprints.shape[0], fingerprints.shape[1], 75), device=device)
        for i in range(N):
            fea = torch.zeros((fingerprints.shape[1], 75), device=device)
            atom_fea = fingerprints[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = fingerprints[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:44] = com
            fea[:, 44:75] = tf
            # com_vec = F.normalize(fea, dim=1)
            compound_vec[i, :, :] = fea
            # print(compound_vec)

        compound_vecs = self.gnn(compound_vec, adjacency, self.layer_gnn)
        t = compound_vecs.shape[1]
        # print(compound_vector.shape)
        compound_vector[:,0:t,:] = compound_vecs

        """Protein vector with attention-CNN."""
        # print(compound_vector.shape)
        word_vectors = torch.zeros((words.shape[0], words.shape[1], 100), device=device)

        for i in range(N):
            t = self.embed_word(torch.LongTensor(words[i].to('cpu').numpy()).cuda())
            tf = F.normalize(t, dim=1)
            word_vectors[i, :, :] = tf

        # protein_vector = self.Encoders(word_vectors)

        protein_fi_vector = torch.zeros((protein_ss.shape[0], protein_ss.shape[1], 100), device=device)
        for i in range(N):
            t = self.embed_word(torch.LongTensor(protein_ss[i].to('cpu').numpy()).cuda())
            # t2 = self.embed_ss(torch.LongTensor(protein_ss[i].to('cpu').numpy()).cuda())
            tf = F.normalize(t, dim=1)
            protein_fi_vector[i, :, :] = tf

        compound_vector = compound_vector + self.WC_struc1(compound_vector)
        compound_vector = self.WC_struc2(compound_vector)

        morgan_vector = morgan.view(N, 1, 2048)
        mol_vector = morgan_vector + self.WC_fcfp1(morgan_vector)
        mol_vector = self.WC_fcfp2(mol_vector)

        compound_vectors = self.dropout(self.attention_PM(compound_vector, mol_vector))  # .permute(0, 2, 1)
        mol_vectors = self.dropout(self.attention_PM2(mol_vector, compound_vector))

        compound_GNN_att = torch.cat((compound_vector, mol_vector), 1)
        mol_FCFPs_att = torch.cat((compound_vectors, mol_vectors), 1)

        drug_vectors = self.Elem_feature_Fusion_D(compound_GNN_att, mol_FCFPs_att)

        protein_vector = word_vectors + self.WP_NN1(word_vectors)
        protein_vector = protein_vector + self.WP_NN3(protein_vector)
        protein_vector = self.WP_NN2(protein_vector)

        protein_fi_vector = protein_fi_vector + self.WP_FiNN1(protein_fi_vector)
        protein_fi_vector = protein_fi_vector + self.WP_FiNN3(protein_fi_vector)
        protein_fi_vector = self.WP_FiNN2(protein_fi_vector)
        #
        protein_vectors = self.Elem_feature_Fusion_P(protein_vector, protein_fi_vector)
        adjacencys = torch.zeros((N, 51, 51), device=device)
        adjacencys[:, 0:L, 0:L] = adjacency

        dta1, dta2, response = self.Decoder(protein_vectors, drug_vectors, adjacencys)

        # PDI1 = self.activation(self.W_out1_1(protein_drug_interaction1))
        PDI1 = self.activation(self.W_out1_1(dta1))
        PDI1 = self.act_norm(self.W_out1_2(PDI1))
        PDI1 = self.activation(self.W_out1_3(PDI1))

        PDI2 = self.act_norm2(self.W_out2_1(dta2))
        PDI2 = self.activation(self.W_out2_2(PDI2))
        PDI2 = self.activation(self.W_out2_3(PDI2))

        PDI = torch.cat((PDI1, PDI2), dim=1)

        PDI = self.activation(self.W_interaction(PDI))
        interaction = self.W_interaction1(PDI)  # .squeeze(1)

        return interaction, response

    def __call__(self, data, train=True):
        # print(np.array(data).shape)
        inputs, correct_interaction, res_labels = data[:-2], data[-2], data[-1]
        correct_interaction = correct_interaction
        protein_drug_interaction, response = self.forward(inputs)
        protein_drug_interaction = protein_drug_interaction.squeeze(1)

        res_labels = res_labels * correct_interaction.unsqueeze(1).expand(res_labels.size())

        if train:

            criterion = nn.MSELoss()
            criterion_res = RWingLoss(om=1, ep=3, r=0.15)
            # criterion2 = nn.SmoothL1Loss()
            loss1 = criterion(protein_drug_interaction.to(torch.float32), correct_interaction.to(torch.float32))
            loss2 = criterion_res(response.to(torch.float32), res_labels.to(torch.float32))

            return loss1 + loss2, loss1
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            predicted_scores = protein_drug_interaction.to('cpu').data.numpy()
            return correct_labels, predicted_scores


class sequence_embedding(nn.Module):
    def __init__(self,):
        super(sequence_embedding, self).__init__()
        self.pool = nn.MaxPool1d(12)
        self.softmax = nn.Softmax(-1)

    def forward(self, p1, p2):
        protein_matrix = torch.matmul(p1, p2.permute(1, 0))
        scale = protein_matrix.size(-1) ** 0.5
        weight = self.softmax(protein_matrix * scale)
        protein_feture = torch.matmul(weight, p2)
        return protein_feture

from torch.nn.modules.loss import _Loss

class BMSE_Loss(nn.Module):
    def __init__(self, sigma, thr):
        super(BMSE_Loss, self).__init__()
        self.sigma = sigma
        self.thr = thr

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        loss1 = loss[loss <= self.thr]
        loss2 = loss[loss > self.thr]
        loss = (loss1.sum() + loss2) / (len(loss) + len(loss2))
        return loss


def pack(atoms, compounds_words, adjs, morgans, proteins, proteins_sss, labels, res_labels, device):
    proteins_len = 1200
    proteinss_len = 1200
    com_words_len = 100
    atoms_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    compounds_word_new = torch.zeros((N, com_words_len), device=device)
    i = 0
    for compounds_word in compounds_words:
        compounds_word_len = compounds_word.shape[0]
        # print(compounds_word.shape)
        if compounds_word_len <= 100:
            compounds_word_new[i, :compounds_word_len] = compounds_word
        else:
            compounds_word_new[i] = compounds_word[0:100]
        # compounds_word_new[i, :compounds_word_len] = compounds_word
        i += 1

    morgan_new = torch.zeros((N, 2048, 1), device=device)
    i = 0
    for morgan in morgans:
        morgan_new[i, :, :] = morgan.unsqueeze(1)
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1

    proteins_ss_new = torch.zeros((N, proteinss_len), device=device)
    i = 0
    for proteins_ss in proteins_sss:
        if proteins_ss.shape[0] > 1200:
            proteins_ss = proteins_ss[0:1200]
        a_len = proteins_ss.shape[0]
        proteins_ss_new[i, :a_len] = proteins_ss
        i += 1

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    res_labels_new = torch.zeros((N, 1200), device=device)
    i = 0
    for res_label in res_labels:
        res_labels_new[i] = res_label[0:1200]
        i += 1


    return atoms_new, compounds_word_new, adjs_new, morgan_new, proteins_new, proteins_ss_new, labels_new, res_labels_new  # , atom_num, protein_num


def pack_aug(atoms, compounds_words, adjs, morgans, proteins, proteins_sss, labels, res_labels, device):
    proteins_len = 1200
    proteinss_len = 1200
    com_words_len = 100
    atoms_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    compounds_word_new = torch.zeros((N, com_words_len), device=device)
    i = 0
    for compounds_word in compounds_words:
        compounds_word_len = compounds_word.shape[0]
        # print(compounds_word.shape)
        if compounds_word_len <= 100:
            compounds_word_new[i, :compounds_word_len] = compounds_word
        else:
            compounds_word_new[i] = compounds_word[0:100]
        # compounds_word_new[i, :compounds_word_len] = compounds_word
        i += 1

    morgan_new = torch.zeros((N, 2048, 1), device=device)
    i = 0
    for morgan in morgans:
        morgan_new[i, :, :] = morgan.unsqueeze(1)
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        nums = torch.randint(low=0, high=36, size=(a_len,))
        mask = (nums > 5).to(device)
        proteins_new[i, :a_len] = protein * mask
        i += 1

    proteins_ss_new = torch.zeros((N, proteinss_len), device=device)
    i = 0
    for proteins_ss in proteins_sss:
        if proteins_ss.shape[0] > 1200:
            proteins_ss = proteins_ss[0:1200]
        a_len = proteins_ss.shape[0]
        nums = torch.randint(low=0, high=36, size=(a_len,))
        mask = (nums > 5).to(device)
        proteins_ss_new[i, :a_len] = proteins_ss * mask
        i += 1

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    res_labels_new = torch.zeros((N, 1200), device=device)
    i = 0
    for res_label in res_labels:
        res_labels_new[i] = res_label[0:1200]
        i += 1

    return atoms_new, compounds_word_new, adjs_new, morgan_new, proteins_new, proteins_ss_new, labels_new, res_labels_new


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch = 16
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        loss_total2 = 0
        i = 0
        self.optimizer.zero_grad()

        fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            fingerprint, compounds_word, adjacency, morgan, word, protein_ss, label, res_label = data
            fingerprints.append(fingerprint)
            compounds_words.append(compounds_word)
            adjacencys.append(adjacency)
            morgans.append(morgan)
            words.append(word)
            protein_sss.append(protein_ss)
            labels.append(label)
            res_labels.append(res_label)
            if i % self.batch == 0 or i == N:
                # print(words[0])
                fingerprints1, compounds_words1, adjacencys1, morgans1, words1, protein_sss1, labels1, res_labels1 = pack(fingerprints,
                                                                                                      compounds_words,
                                                                                                      adjacencys,
                                                                                                      morgans, words,
                                                                                                      protein_sss,
                                                                                                      labels, res_labels, device)
                # print(words.shape)
                data = (fingerprints1, compounds_words1, adjacencys1, morgans1, words1, protein_sss1, labels1, res_labels1)
                loss, loss3 = self.model(data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=8)
                self.optimizer.step()
                self.optimizer.zero_grad()
                fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels = [], [], [], [], [], [], [], []
            else:
                continue


            loss_total += loss.item()
            loss_total2 += loss3.item()


        return loss_total, (loss_total2 * self.batch) / len(dataset)


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (float(down) + 0.00000001))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, S = [], []
        i = 0
        fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            fingerprint, compounds_word, adjacency, morgan, word, protein_ss, label, res_label = data
            fingerprints.append(fingerprint)
            compounds_words.append(compounds_word)
            adjacencys.append(adjacency)
            morgans.append(morgan)
            words.append(word)
            protein_sss.append(protein_ss)
            labels.append(label)
            res_labels.append(res_label)
            if i % 16 == 0 or i == N:
                # print(words[0])
                fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels = pack(
                    fingerprints,
                    compounds_words,
                    adjacencys,
                    morgans, words,
                    protein_sss,
                    labels, res_labels, device)
                # print(words.shape)
                data = (fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels)
                (correct_labels, predicted_scores) = self.model(data, train=False)
                for i in range(len(correct_labels)):
                    T.append(correct_labels[i])
                    S.append(predicted_scores[i])
                fingerprints, compounds_words, adjacencys, morgans, words, protein_sss, labels, res_labels = [], [], [], [], [], [], [], []
            else:
                continue

        CI = get_cindex(T, S)
        MSE = mean_squared_error(T, S)
        rm = get_rm2(T, S)
        return CI, MSE, rm, T, S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

from sklearn import metrics
def get_aupr(Y, P):
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y >= 7, 1, 0)
    P = np.where(P >= 7, 1, 0)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_tensor2(file_name, dtype):
    domains = np.load(file_name + '.npy', allow_pickle=True)
    Domain = []
    for d in domains:
        domain = []
        for j in d:
            domain.append(dtype(j).to(device))
        Domain.append(domain)
    return Domain


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def dataset(dir_input1, dir_input2):
    print(dir_input1 + 'compounds')
    compounds1 = load_tensor(dir_input1 + 'compounds', torch.LongTensor)
    compounds_word1 = load_tensor(dir_input1 + 'compound_words', torch.LongTensor)
    adjacencies1 = load_tensor(dir_input1 + 'adjacencies', torch.LongTensor)
    morgans = load_tensor(dir_input1 + 'morgan', torch.LongTensor)
    proteins1 = load_tensor(dir_input1 + 'proteins_feature', torch.LongTensor)
    protein_ss1 = load_tensor(dir_input1 + 'proteins', torch.LongTensor)
    interactions1 = load_tensor(dir_input1 + 'interactions', torch.FloatTensor)
    res_labels1 = load_tensor(dir_input1 + 'response_label15', torch.FloatTensor)
    train_dataset = list(zip(compounds1, compounds_word1, adjacencies1, morgans, proteins1, protein_ss1, interactions1, res_labels1))
    train_dataset = shuffle_dataset(train_dataset, 1234)

    print(dir_input2 + 'compounds')
    compounds2 = load_tensor(dir_input2 + 'compounds', torch.LongTensor)
    compounds_word2 = load_tensor(dir_input2 + 'compound_words', torch.LongTensor)
    adjacencies2 = load_tensor(dir_input2 + 'adjacencies', torch.LongTensor)
    morgans2 = load_tensor(dir_input2 + 'morgan', torch.LongTensor)
    proteins2 = load_tensor(dir_input2 + 'proteins_feature', torch.LongTensor)
    protein_ss2 = load_tensor(dir_input2 + 'proteins', torch.LongTensor)
    interactions2 = load_tensor(dir_input2 + 'interactions', torch.FloatTensor)
    res_labels2 = load_tensor(dir_input2 + 'response_label15', torch.FloatTensor)
    dev_dataset = list(zip(compounds2, compounds_word2, adjacencies2, morgans2, proteins2, protein_ss2, interactions2, res_labels2))
    dev_dataset = shuffle_dataset(dev_dataset, 1234)

    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    return train_dataset, dev_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    """Hyperparameters."""
    # (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
    # lr, lr_decay, decay_interval, weight_decay, iteration,
    # setting) = sys.argv[1:]
    DATASET = "/home/zhong/data_davis_kiba/DTA_data/davis_input/1"
    radius = 2
    ngram = 3
    dim = 75
    layer_gnn = 3
    side = 7
    side2 = 4
    window = (2 * side + 1)
    window2 = (2 * side2 + 1)
    layer_cnn = 3
    layer_output = 2
    lr = 5e-4
    lr_min = 1e-5
    lr_decay = 0.5
    decay_interval = 6
    weight_decay = 1e-3
    iteration = 380
    setting = "optimize1"

    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dir_input1 = DATASET + "/input_train/"
    dir_input2 = DATASET + "/input_test/"

    dataset_train, dataset_test = dataset(dir_input1, dir_input2)

    setup_seed(1234)
    model = Predictor(Decoder, ResMLP, Affine, dim, window, window2, layer_gnn, layer_cnn, layer_output).to(device)
    model = model.cuda()
    # model.load_state_dict(torch.load("output/model/optimize3"))
    trainer = Trainer(model, lr, weight_decay)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/AUCs--' + setting + '.txt'
    file_model = 'output/model/' + setting
    res_ = 'output/result/res--' + setting + '.txt'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'CI_test\tMSE_test\trm\tloss_total2')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    mse = 10
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] = trainer.optimizer.param_groups[0]['lr'] * lr_decay
            # trainer.optimizer.param_groups[0]['lr'] = [t, lr_min][t < lr_min]
        loss_train, loss_total2 = trainer.train(dataset_train)

        # CI_ = tester.test(dataset_test)[0]
        with torch.no_grad():
            CI, MSE, rm, T, S = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, CI, MSE, rm, loss_total2]
        tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        np_t = 'output/np/davis' + setting + str(epoch) + '_t.npy'
        np_s = 'output/np/davis' + setting + str(epoch) +'_s.npy'

        if mse > MSE:
            mse = MSE
            tester.save_model(model, file_model)  # print(Do_Dr_interact)
            # np.save(np_s, T)
            # np.save(np_t, S)
            files = open(res_, 'w')
            for i in range(len(T)):
                files.write("pre: " + str(S[i]) + "; lab: " + str(T[i]) + "\n")