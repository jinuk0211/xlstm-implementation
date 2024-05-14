import torch
import torch.nn as nn
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_dim):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.W_q = nn.Parameter(torch.randn(hidden_size, input_size))
        self.b_q = nn.Parameter(torch.randn(hidden_size, 1))
        self.W_k = nn.Parameter(torch.randn(memory_dim, input_size))
        self.b_k = nn.Parameter(torch.randn(memory_dim, 1))
        self.W_v = nn.Parameter(torch.randn(memory_dim, input_size))
        self.b_v = nn.Parameter(torch.randn(memory_dim, 1))
        self.w_i = nn.Parameter(torch.randn(1, input_size))
        self.b_i = nn.Parameter(torch.randn(1))
        self.w_f = nn.Parameter(torch.randn(1, input_size))
        self.b_f = nn.Parameter(torch.randn(1))
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, states):
        (C_prev, n_prev) = states
        q_t = torch.matmul(self.W_q, x) + self.bq # query input 22번
        k_t = (1 / math.sqrt(self.memory_dim)) * (torch.matmul(self.W_k, x) + self.b_k)
        #key input 23번
        v_t = torch.matmul(self.W_v, x) + self.b_v
        #value input 24번

        i_t = torch.exp(torch.matmul(self.w_i, x) + self.b_i)
        #input gate 25번
        f_t = torch.sigmoid(torch.matmul(self.w_f, x) + self.b_f)
        #forget gate 26번
        v_t = v_t.squeeze() #C(T) = C(T-1) + vt
        k_t = k_t.squeeze()

        C = f_t * C_prev + i_t * torch.ger(v_t, k_t) #외적곱
        #cell state 19번
        n = f_t * n_prev + i_t * k_t.unsqueeze(1)
        #normalizer state 20번
        
        #21번 과정 #hidden state 
        max_nqt = torch.max(torch.abs(torch.matmul(n.T, qt)), torch.tensor(1.0))
        h_wave = torch.matmul(C, qt) / max_nqt  
        ot = torch.sigmoid(torch.matmul(self.Wo, x) + self.bo)
        #ot = output gate
        ht = ot * h_wave #

        return ht, (C, n)
