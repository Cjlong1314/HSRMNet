import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import args
import scipy.sparse as sp


class link_prediction(nn.Module):
    def __init__(self, i_dim, h_dim, num_rels, num_times, use_cuda=False, dataset='YAGO'):
        super(link_prediction, self).__init__()
        self.dataset = dataset
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_times = num_times
        self.use_cuda = use_cuda
        self.ent_init_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, h_dim))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.global_mode = Global_mode(self.h_dim, self.i_dim, use_cuda)
        self.standard_mode = Standard_mode(self.h_dim, self.i_dim, use_cuda)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_init_embeds, gain=nn.init.calculate_gain('relu'))

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] / args.time_stamp
        init_tim = torch.Tensor(self.num_times, self.h_dim)
        for i in range(self.num_times):
            init_tim[i] = torch.Tensor(self.tim_init_embeds.cpu().detach().numpy().reshape(self.h_dim)) * (i + 1)
        init_tim = init_tim.to('cuda:{}'.format(args.gpu))
        T = init_tim[T_idx]
        return T

    # todo 新增
    def get_s_p_o(self, quadrupleList):
        s_idx = quadrupleList[:, 0]
        p_idx = quadrupleList[:, 1]
        o_idx = quadrupleList[:, 2]
        s = self.ent_init_embeds[s_idx]
        p = self.w_relation[p_idx]
        o = self.ent_init_embeds[o_idx]
        return s, p, o

    # todo  新增
    def forward(self, quadruple, copy_vocabulary, entity):
        s, p, o = self.get_s_p_o(quadruple)
        T = self.get_init_time(quadruple)
        score_g = self.global_mode(s, p, o, T, copy_vocabulary, entity)
        score_s = self.standard_mode(s, p, o, T, copy_vocabulary, entity)
        a = args.alpha
        score = score_s * a + score_g * (1 - a)
        # score = score_s
        # score = score_g
        score_end = torch.log(score)
        return score_end

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.w_relation.pow(2)) + torch.mean(self.ent_init_embeds.pow(2)) + torch.mean(
            self.tim_init_embeds.pow(2))
        return regularization_loss * reg_param

class Standard_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda):
        super(Standard_mode, self).__init__()
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda

    def forward(self, s_embed, rel_embed, o_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = torch.cat((s_embed, rel_embed, time_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, o_embed, time_embed), dim=1)
        q_s = self.relu(self.W_s(m_t))
        encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() != 0, dtype=float) * 1)
        encoded_mask += torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * (-10))
        if self.use_cuda:
            encoded_mask = encoded_mask.to('cuda:{}'.format(args.gpu))
        score_s = q_s + encoded_mask
        return F.softmax(score_s, dim=1)


class Global_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda):
        super(Global_mode, self).__init__()
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda

    def forward(self, s_embed, rel_embed, o_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = torch.cat((s_embed, rel_embed, time_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, o_embed, time_embed), dim=1)
        q_s = self.relu(self.W_s(m_t))
        cv = copy_vocabulary.cpu().numpy()
        encoded_mask = torch.Tensor(cv * (1))
        if self.use_cuda:
            encoded_mask = encoded_mask.to('cuda:{}'.format(args.gpu))
        score_g = q_s + encoded_mask
        # score_g = q_s + F.softmax(encoded_mask)
        return F.softmax(score_g, dim=1)
