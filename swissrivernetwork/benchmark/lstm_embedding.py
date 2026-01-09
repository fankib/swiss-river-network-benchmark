import torch
import torch.nn as nn
import torch.nn.functional as f


'''
This file handles own created LSTM Cells and Networks
'''

class VanillaLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights for input and state
        self.W_x = nn.Parameter(torch.Tensor(input_size, 4*hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, 4*hidden_size))
        self.b = nn.Parameter(torch.zeros(4*hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x)        
        nn.init.orthogonal_(self.W_h)
    
    def forward(self, x, state):
        # x: (batch, input_size)
        # state: (h, c) @ (batch, hidden_size)

        h, c = state

        # Compute gates:
        gates = x @ self.W_x + h @ self.W_h + self.b
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i) # input gate
        f = torch.sigmoid(f) # forget gate
        g = torch.tanh(g) # candidate gate
        o = torch.sigmoid(o) # output gate

        c_next = f*c + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

class EmbeddingGateLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, embedd_memory):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size        
        self.embedding_size = embedding_size
        self.embedd_memory = embedd_memory # either embedd into the memory cell or the next hidden state

        # Weights for input and state
        self.W_x = nn.Parameter(torch.Tensor(input_size, 5*hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, 5*hidden_size))
        self.W_e = nn.Parameter(torch.Tensor(embedding_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(5*hidden_size))
        self.b_e = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.orthogonal_(self.W_h)
    
    def forward(self, x, emb, state):
        # x: (batch, input_size)
        # emb: (batch, hidden_size)
        # state: (h, c) @ (batch, hidden_size)

        h, c = state

        # Compute gates:
        gates = x @ self.W_x + h @ self.W_h + self.b
        i, f, g, o, e = gates.chunk(5, dim=-1)

        i = torch.sigmoid(i) # input gate
        f = torch.sigmoid(f) # forget gate
        g = torch.tanh(g) # candidate gate
        o = torch.sigmoid(o) # output gate
        e = torch.sigmoid(e) # embedding gate

        # embedding transformation:
        emb = torch.tanh(emb @ self.W_e + self.b_e)

        if self.embedd_memory:
            c_next = f*c + i*g + e*emb # Where to put the embedding?
            h_next = o*torch.tanh(c_next)
        else:
            # embedd hidden state
            c_next = f*c + i*g
            h_next = o*torch.tanh(c_next) + e*emb

        return h_next, c_next

#def columnwise_interpolate(A, B, alpha):
#    # ensure alpha is broadcastable across rows
#    alpha = alpha.unsqueeze(0)           # (1, n)
#    return alpha * A + (1 - alpha) * B   # broadcasting column-wise

class InterpolationLstmCell(nn.Module):
    # The weights are a convex combination of
    # W = aW1 + (1-a)W2, where a = Ve + b
    
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # Weights for input and state
        self.W_x1 = nn.Parameter(torch.Tensor(input_size, 4*hidden_size))
        self.W_x2 = nn.Parameter(torch.Tensor(input_size, 4*hidden_size))
        self.W_h1 = nn.Parameter(torch.Tensor(hidden_size, 4*hidden_size))
        self.W_h2 = nn.Parameter(torch.Tensor(hidden_size, 4*hidden_size))        
        self.b1 = nn.Parameter(torch.zeros(4*hidden_size))
        self.b2 = nn.Parameter(torch.zeros(4*hidden_size))
        self.W_e = nn.Parameter(torch.Tensor(embedding_size, hidden_size))
        self.b_e = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x1)
        nn.init.xavier_uniform_(self.W_x2)                
        nn.init.orthogonal_(self.W_h1)
        nn.init.orthogonal_(self.W_h2)
        nn.init.xavier_uniform_(self.W_e)        
    
    def forward(self, x, emb, state):
        # x: (batch, input_size)
        # state: (h, c) @ (batch, hidden_size)

        h, c = state

        # compute embedding:
        alpha = torch.sigmoid(emb @ self.W_e + self.b_e)
        alpha4 = alpha.repeat(1, 4)

        # do matmuls first, then interpolate results
        x_proj = x @ self.W_x1, x @ self.W_x2
        h_proj = h @ self.W_h1, h @ self.W_h2

        gates = alpha4 * (x_proj[0] + h_proj[0]) \
          + (1 - alpha4) * (x_proj[1] + h_proj[1]) \
          + alpha4 * self.b1 + (1 - alpha4) * self.b2
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i) # input gate
        f = torch.sigmoid(f) # forget gate
        g = torch.tanh(g) # candidate gate
        o = torch.sigmoid(o) # output gate

        c_next = f*c + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

class MultiInterpolationLstmCell(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_bases):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_bases = num_bases  # number of base matrices (N)

        # Base weight sets for input and hidden
        self.W_x = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
            for _ in range(num_bases)
        ])
        self.W_h = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
            for _ in range(num_bases)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(4 * hidden_size))
            for _ in range(num_bases)
        ])

        # Parameters for generating alphas
        self.W_e = nn.Parameter(torch.Tensor(embedding_size, num_bases*hidden_size))
        self.b_e = nn.Parameter(torch.zeros(num_bases))

        self.reset_parameters()
    
    def reset_parameters(self):
        for Wx in self.W_x:
            nn.init.xavier_uniform_(Wx)
        for Wh in self.W_h:
            nn.init.orthogonal_(Wh)
        nn.init.xavier_uniform_(self.W_e)
    
    def forward(self, x, emb, state):
        # x: (batch, input_size)
        # emb: (batch, embedding_size)
        # state: (h, c), each (batch, hidden_size)
        h, c = state

        batch_size = x.size(0)
        N = self.num_bases
        out_dim = 4 * self.hidden_size

        # Compute positive alphas, normalized (convex combination)
        raw_alpha = torch.sigmoid(emb @ self.W_e + self.b_e) # ensure > 0
        alpha = raw_alpha / raw_alpha.sum(dim=-1, keepdim=True)  # normalize
        alpha4 = alpha.unsqueeze(-1)  # shape (batch, N, 1)

        # Stack all base weights
        W_x_stack = torch.stack(list(self.W_x), dim=0)  # (N, in, 4h)
        W_h_stack = torch.stack(list(self.W_h), dim=0)  # (N, h, 4h)
        b_stack = torch.stack(list(self.b), dim=0)      # (N, 4h)

        # Compute interpolated weights
        W_x_eff = torch.einsum('bnj,nij->bij', alpha, W_x_stack)  # (batch, in, 4h)
        W_h_eff = torch.einsum('bnj,nij->bij', alpha, W_h_stack)  # (batch, h, 4h)
        b_eff   = torch.einsum('bnj,nj->bj', alpha, b_stack)      # (batch, 4h)

        # Compute gates
        gates = x @ W_x_eff + h @ W_h_eff + b_eff
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class VanillaLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaLSTMCell(input_size, hidden_size)
    
    def forward(self, x, state=None):
        # x: (batch, seq_len, input_size)
        # state: optional tuple (h0, c0)
        # returns: (output, (hn, cn))
        #   output: (batch, seq_len, hidden_size)

        batch, seq_len, _ = x.size()

        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = state
        
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], (h,c))
            outputs.append(h.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (h,c)

class EmbeddingCellLSTM(nn.Module):

    def __init__(self, hidden_size, cell):
        super().__init__()
        self.hidden_size = hidden_size        
        self.cell = cell
    
    def forward(self, x, emb, state=None):
        # x: (batch, seq_len, input_size)
        # state: optional tuple (h0, c0)
        # returns: (output, (hn, cn))
        #   output: (batch, seq_len, hidden_size)

        batch, seq_len, _ = x.size()

        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = state
        
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], emb[:, t, :], (h,c))
            outputs.append(h.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (h,c)

##################
# Model Variants #
##################

class VanillaLstmModel(nn.Module):

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()        
        self.lstm = VanillaLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)    
    
    def forward(self, e, x, emb=None):        
        out, hidden = self.lstm(x)
        target = self.linear(out)
        return target

class ConcatenationEmbeddingModel(nn.Module):

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = VanillaLSTM(input_size+embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2) # x in [batch x sequence x features]
        out, hidden = self.lstm(x)
        target = self.linear(out)
        return target

class ConcatenationEmbeddingOutputModel(nn.Module):
    
    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = VanillaLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size+embedding_size, 1)    
    
    def forward(self, e, x, emb=None):
        out, hidden = self.lstm(x)
        # concatenate the embedding to the output (Flashback Style!)
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out = torch.cat((emb, out), 2) # x in [batch x sequence x features]
        target = self.linear(out)
        return target


class EmbeddingGateMemoryModel(nn.Module):

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = EmbeddingGateLSTMCell(input_size, hidden_size, embedding_size, True)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, 1)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, hidden = self.lstm(x, emb)
        target = self.linear(out)
        return target

class EmbeddingGateHiddenModel(nn.Module):
    
    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = EmbeddingGateLSTMCell(input_size, hidden_size, embedding_size, False)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, 1)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, hidden = self.lstm(x, emb)
        target = self.linear(out)
        return target

class InterpolationEmbeddingModel(nn.Module):
    
    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = InterpolationLstmCell(input_size, hidden_size, embedding_size)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, 1)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, hidden = self.lstm(x, emb)
        target = self.linear(out)
        return target


### Model Registry ###

EMBEDDING_MODEL_FACTORY = {
    'vanilla': VanillaLstmModel,
    'concatenation_embedding': ConcatenationEmbeddingModel,
    'concatenation_embedding_output': ConcatenationEmbeddingOutputModel,
    'embedding_gate_memory': EmbeddingGateMemoryModel,
    'embedding_gate_hidden': EmbeddingGateHiddenModel,
    'interpolation_embedding': InterpolationEmbeddingModel
}


