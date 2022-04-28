import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
import torch
from torch import optim
import torch.nn as nn
torch.manual_seed(1)
import torch.nn.functional as F
from softnw import SoftNW, softnw_score
import AsMac_utility
import time
from _softnw import best_hit_fast, embed_value_fast

# class SoftNW(torch.autograd.Function):

class AsMac(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, gamma=0.01):
        super(AsMac, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.gap = torch.nn.Parameter(torch.ones([out_dim]))
        self.gamma = gamma
        self.weights = torch.nn.Parameter(torch.rand([out_dim, in_dim, kernel_size]) - 0.5 * torch.ones([out_dim, in_dim, kernel_size]))
        self.bias = torch.nn.Parameter(torch.rand([out_dim]) - 0.5 * torch.ones([out_dim]))
        self.softnw = SoftNW.apply
        self.cosine = nn.CosineSimilarity()

    def __repr__(self):
        model_str = 'AsMac('
        filter_str = '(weight): SoftNW1d(' + str(self.weights.shape[1]) + ', ' + str(
            self.weights.shape[0]) + ', kernel_size=' + str(self.weights.shape[2]) + ')'
        main_str = '\n'.join([model_str, filter_str, ')'])
        return main_str

    def state_dict(self):
        state = dict()
        state['weight'] = self.weights
        state['bias'] = self.bias
        state['gap'] = self.gap
        return state

    def load_state_dict(self, state):

        try:
            toy = torch.rand([self.out_dim, self.in_dim, self.kernel_size])
            toy[:, :, :] = torch.nn.Parameter(state['weight'])
        except:
            print("weight dimension not compatible")
            raise

        self.weights = torch.nn.Parameter(state['weight'])
        self.bias = torch.nn.Parameter(state['bias'])
        self.gap = torch.nn.Parameter(state['gap'])


    def forward_embed(self, seq):

        embed = torch.zeros([self.out_dim])
        for i in range(self.weights.shape[0]):

            weight = self.weights[i, :, :].detach().numpy().astype(np.float64)
            g = self.gap[i].detach().numpy().astype(np.float64)
            v, j_min, j_max = best_hit_fast(seq, weight, gap=g)
            seq_pick = torch.FloatTensor(seq[:, j_min:j_max])
            score = self.softnw(seq_pick, self.weights[i, :, :], self.gap[i])

            entry = score + self.bias[i]
            embed[i] = entry
        embed = F.normalize(input=F.relu(embed), p=2, dim=0)

        return embed

    def test_embed(self, seq):
        embed = torch.zeros([self.out_dim])
        for i in range(self.weights.shape[0]):
            weight = self.weights[i, :, :].detach().numpy().astype(np.float64)
            g = self.gap[i].detach().numpy().astype(np.float64)
            v = embed_value_fast(seq, weight, gap=g)
            entry = v + self.bias[i]

            embed[i] = entry

        embed = F.normalize(input=F.relu(embed), p=2, dim=0)
        return embed

    def test_forward(self, seq_oh):
        l = len(seq_oh)
        embeddings = torch.zeros([l, self.out_dim])
        for i, seq in enumerate(seq_oh):
            embeddings[i, :] = self.test_embed(seq)
        output = torch.ones(l, l) - torch.mm(embeddings, torch.transpose(embeddings, 0, 1))

        return output

    def forward(self, seq_oh):
        l = len(seq_oh)
        embeddings = torch.zeros([l, self.out_dim])
        for i, seq in enumerate(seq_oh):
            embeddings[i, :] = self.forward_embed(seq)
        output = torch.ones(l, l) - torch.mm(embeddings, torch.transpose(embeddings, 0, 1))

        return output


if __name__ == "__main__":

    net = AsMac(4, 150, 20)
    print(net)

    print(net.gap_regulate())
