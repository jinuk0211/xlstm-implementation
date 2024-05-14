import torch
import torch.nn as nn
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_dim):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bq = nn.Parameter(torch.randn(hidden_size, 1))
        self.Wk = nn.Parameter(torch.randn(memory_dim, input_size))
        self.bk = nn.Parameter(torch.randn(memory_dim, 1))
        self.Wv = nn.Parameter(torch.randn(memory_dim, input_size))
        self.bv = nn.Parameter(torch.randn(memory_dim, 1))
        self.wi = nn.Parameter(torch.randn(1, input_size))
        self.bi = nn.Parameter(torch.randn(1))
        self.wf = nn.Parameter(torch.randn(1, input_size))
        self.bf = nn.Parameter(torch.randn(1))
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bo = nn.Parameter(torch.randn(hidden_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, states):
        (C_prev, n_prev) = states
        qt = torch.matmul(self.Wq, x) + self.bq # query input 22번
        kt = (1 / math.sqrt(self.memory_dim)) * (torch.matmul(self.Wk, x) + self.bk)
        #key input 23번
        vt = torch.matmul(self.Wv, x) + self.bv
        #value input 24번

        it = torch.exp(torch.matmul(self.wi, x) + self.bi)
        #input gate 25번
        ft = torch.sigmoid(torch.matmul(self.wf, x) + self.bf)
        #forget gate 26번
        vt = vt.squeeze() #C(T) = C(T-1) + vt
        kt = kt.squeeze()

        C = ft * C_prev + it * torch.ger(vt, kt) #외적곱
        #cell state 19번
        n = ft * n_prev + it * kt.unsqueeze(1)
        #normalizer state 20번
        
        #21번 과정 #hidden state 
        max_nqt = torch.max(torch.abs(torch.matmul(n.T, qt)), torch.tensor(1.0))
        h_wave = torch.matmul(C, qt) / max_nqt  
        ot = torch.sigmoid(torch.matmul(self.Wo, x) + self.bo)
        #ot = output gate
        ht = ot * h_wave #

        return ht, (C, n)
