import torch
import torch.nn as nn
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        
        self.w_input = nn.Parameter(torch.Tensor(hidden_size, input_size)) #12번 inputgate
        self.w_forget = nn.Parameter(torch.Tensor(hidden_size, input_size)) #13번 forgetgate
        self.w_output = nn.Parameter(torch.Tensor(hidden_size, input_size)) #14번 output gate
        self.w_z = nn.Parameter(torch.Tensor(hidden_size, input_size)) #11번 cell input gate

        self.r_z = nn.Parameter(   
            torch.Tensor(hidden_size, hidden_size)  #11번 cell input
        )
        self.r_input = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size)  #12번 inputgate
        )
        self.r_forget = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size)  #13번 forgetgate
        )
        self.r_output = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size)  #14번 outputgate
        )
        
        self.b_z = nn.Parameter(torch.Tensor(hidden_size)) #11번 cell input
        self.b_input = nn.Parameter(torch.Tensor(hidden_size)) #12번 input gate
        self.b_forget = nn.Parameter(torch.Tensor(hidden_size)) #13번 forget gate 
        self.b_output = nn.Parameter(torch.Tensor(hidden_size)) #14번 output gate

        self.sigmoid = nn.Sigmoid()

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_input)
        nn.init.xavier_uniform_(self.w_forget)
        nn.init.xavier_uniform_(self.w_output)
        nn.init.xavier_uniform_(self.w_z)

        nn.init.orthogonal_(self.r_input)
        nn.init.orthogonal_(self.r_forget)
        nn.init.orthogonal_(self.r_output)
        nn.init.orthogonal_(self.r_z)

        nn.init.zeros_(self.b_input)
        nn.init.zeros_(self.b_forget)
        nn.init.zeros_(self.b_output)
        nn.init.zeros_(self.b_z)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states
        #hidden state, cell state, normalizer state, stabilizer state

        input_wave = ( #INPUT GATE 12번
            torch.matmul(self.w_input, x)
            + torch.matmul(self.r_input, h_prev)
            + self.b_input
        )
        forget_wave = ( #forget gate 13번
            torch.matmul(self.w_forget, x)
            + torch.matmul(self.r_forget, h_prev)
            + self.b_forget
        )
        output_wave = ( #output gate 14번
            torch.matmul(self.w_output, x)
            + torch.matmul(self.r_output, h_prev)
            + self.b_output
        )
        z_wave = ( #cell input 11번
            torch.matmul(self.w_z, x)
            + torch.matmul(self.r_z, h_prev)
            + self.b_z
        )
        #t 는 timestep
        input_t = torch.exp(input_wave) #input gate 12번
        forget_t = self.sigmoid( #forget gate 13번
            forget_wave
        )  # sigmoid 또는 exp based on context

        # Stabilizer state update
        m_t = torch.max(torch.log(forget_t) + m_prev, torch.log(input_t)) #15번

        # Stabilized gates
        input_prime = torch.exp(torch.log(input_t) - m_t) #16번 input gate(stabilizer)
        forget_prime = torch.exp(torch.log(forget_t) + m_prev - m_t) #17번 forget gate(stabilizer)

        c_t = forget_prime * c_prev + input_prime * torch.tanh(z_wave) #8번 cell state
        n_t = forget_prime * n_prev + forget_prime # 9번 normalizer forget_prime

        h_wave = c_t / n_t #10번 hidden state
        h_t = self.sigmoid(output_wave) * torch.tanh(h_wave) #10번

        return h_t, (h_t, c_t, n_t, m_t)


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(sLSTM, self).__init__()
        self.layers = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if i == 0 else hidden_size, hidden_size
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, initial_states=None):
        batch_size, seq_len, _ = x.size()
        if initial_states is None:
            initial_states = [
                (
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                )
                for _ in self.layers
            ]

        outputs = []
        current_states = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t  
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states

        outputs = torch.cat(
            outputs, dim=1
        ) #time dim에서 concat
        return outputs, current_states


x = torch.randn(1, 10, 64)
model = sLSTM(64, 128, 2)
output, states = model(x)
print(output.size())
