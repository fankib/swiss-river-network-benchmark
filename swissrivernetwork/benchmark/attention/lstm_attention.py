from swissrivernetwork.benchmark.embedding.lstm_embedding import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationEmbeddingModel(nn.Module):
    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding_1 = nn.Embedding(num_embeddings, embedding_size)
        self.lstm_1 = VanillaLSTM(input_size + embedding_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, 1)

        self.attention_embedding = nn.Embedding(num_embeddings, hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        self.embedding_2 = nn.Embedding(num_embeddings, embedding_size)
        self.lstm_2 = VanillaLSTM(hidden_size + embedding_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, e, x, emb=None):        

        batch, stations, sequence, dimensions = x.shape

        # ----- First LSTM -----        
        emb = self.embedding_1(e)  # [batch x stations x seq x embedding]
        x_1 = torch.cat((emb, x), dim=3)       

        x_1 = x_1.reshape(batch*stations, sequence, dimensions+self.embedding_size)
        out_1, hidden = self.lstm_1(x_1)
        target_1 = self.linear_1(out_1)
        out_1 = out_1.reshape(batch, stations, sequence, self.hidden_size)
        target_1 = target_1.reshape(batch, stations, sequence, 1)
        

        # --- Multi-Head Attention over Stations:
        # preapare "station encoding"
        emb = self.attention_embedding(e)
        out_1 = out_1 + emb # superimpose embedding
        
        out_1 = out_1.permute(0, 2, 1, 3).contiguous() # [batch x seq x stations x embedding]
        out_1 = out_1.reshape(batch*sequence, stations, self.hidden_size)
        out_attention, weights = self.mha(out_1, out_1, out_1)
        out_attention = self.attention_norm(out_1 + out_attention) # skip connection + layer norm
        out_attention = out_attention.reshape(batch, sequence, stations, self.hidden_size)
        out_attention = out_attention.permute(0, 2, 1, 3).contiguous() # restore permutation

        # ----- Second LSTM -----
        emb = self.embedding_2(e)
        x_2 = torch.cat((emb, out_attention), dim=3)  # combine LSTM output after attention + new embedding
        x_2 = x_2.reshape(batch*stations, sequence, self.hidden_size+self.embedding_size)
        out_2, hidden = self.lstm_2(x_2)
        target_2 = self.linear_2(out_2)
        target_2 = target_2.reshape(batch, stations, sequence, 1)

        return target_1, target_2


### CHATGPT Improved:

class ImprovedConcatenationEmbeddingModel(nn.Module):
    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        # Embeddings
        self.embedding_shared = nn.Embedding(num_embeddings, embedding_size)

        # First LSTM and linear
        self.lstm_1 = nn.LSTM(input_size + embedding_size, hidden_size, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, 1)

        # Attention projections
        self.q_proj = nn.Linear(embedding_size, hidden_size)
        self.k_proj = nn.Linear(embedding_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Second LSTM and linear
        self.lstm_2 = nn.LSTM(hidden_size + embedding_size, hidden_size, batch_first=True)
        self.linear_2 = nn.Linear(hidden_size, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, e, x):
        """
        e: [batch x seq x station_id]
        x: [batch x seq x features]
        """
        batch_size, seq_len, _ = x.size()

        # ----- First LSTM -----
        emb = self.embedding_shared(e)  # [batch, seq, embedding]
        x1 = torch.cat((x, emb), dim=2)
        x1 = self.dropout(x1)
        out1, _ = self.lstm_1(x1)
        out1 = self.dropout(out1)
        target_1 = self.linear_1(out1)

        # ----- Multi-Head Attention -----
        Q = self.q_proj(emb)  # [batch, seq, hidden]
        K = self.k_proj(emb)
        V = self.v_proj(out1)

        # Split heads
        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, Vh)  # [batch, heads, seq, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.attn_ln(out1 + attn_output)  # residual + layernorm
        attn_output = attn_output + self.ffn(attn_output)  # feed-forward residual
        attn_output = self.dropout(attn_output)

        # ----- Second LSTM -----
        x2 = torch.cat((attn_output, emb), dim=2)  # combine attention output with embeddings
        out2, _ = self.lstm_2(x2)
        out2 = self.dropout(out2)
        target_2 = self.linear_2(out2)

        return target_1, target_2


## Second Shot:
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStageLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, num_stations, embedding_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        # ----- Stage 1: Rough LSTM -----
        self.station_emb_1 = nn.Embedding(num_stations, embedding_size)
        self.lstm1 = nn.LSTM(input_size + embedding_size, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

        # ----- Stage 2: Refinement with attention -----
        self.station_emb_2 = nn.Embedding(num_stations, embedding_size)

        # Attention projections for stage 2
        self.q_proj = nn.Linear(embedding_size, hidden_size)
        self.k_proj = nn.Linear(embedding_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # LayerNorm & FFN after attention
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.lstm2 = nn.LSTM(hidden_size + embedding_size, hidden_size, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, station_ids, x):
        """
        station_ids: [batch, seq_len]  -> integer IDs of stations
        x: [batch, seq_len, features]
        """
        batch_size, seq_len, _ = x.size()

        # ----- Stage 1: Rough LSTM -----
        emb1 = self.station_emb_1(station_ids)  # [batch, seq, emb]
        x1 = torch.cat((x, emb1), dim=2)
        x1 = self.dropout(x1)
        out1, _ = self.lstm1(x1)
        out1 = self.dropout(out1)
        target_1 = self.linear1(out1)

        # ----- Stage 2: Refinement with Attention -----
        emb2 = self.station_emb_2(station_ids)

        # Project embeddings to queries and keys
        Q = self.q_proj(emb2)  # [batch, seq, hidden]
        K = self.k_proj(emb2)
        V = self.v_proj(out1)  # values come from stage 1 LSTM

        # Split for multi-head attention
        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        # Temporal attention: attention across sequence dimension
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, heads, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, Vh)  # [batch, heads, seq, head_dim]

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Residual + LayerNorm + FFN
        attn_out = self.attn_ln(out1 + attn_out)
        attn_out = attn_out + self.ffn(attn_out)
        attn_out = self.dropout(attn_out)

        # Concatenate attention output with station embedding for LSTM2
        x2 = torch.cat((attn_out, emb2), dim=2)
        out2, _ = self.lstm2(x2)
        out2 = self.dropout(out2)
        target_2 = self.linear2(out2)

        return target_1, target_2


## Interleve LSTM and attention:
import torch
import torch.nn as nn
import torch.nn.functional as F

class StepwiseLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, num_stations, embedding_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        # Station embeddings
        self.station_emb = nn.Embedding(num_stations, embedding_size)

        # Stepwise LSTM
        self.lstm_cell = nn.LSTMCell(input_size + embedding_size, hidden_size)

        # Linear output per step
        self.linear_out = nn.Linear(hidden_size, 1)

        # Attention projections
        self.q_proj = nn.Linear(embedding_size, hidden_size)
        self.k_proj = nn.Linear(embedding_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, station_ids, x):
        """
        station_ids: [batch, seq_len]  -> integer IDs of stations
        x: [batch, seq_len, features]
        """
        batch_size, seq_len, _ = x.size()
        emb = self.station_emb(station_ids)  # [batch, seq, emb]

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]             # [batch, features]
            emb_t = emb[:, t, :]         # [batch, embedding]
            lstm_input = torch.cat((x_t, emb_t), dim=1)

            # Stepwise LSTM
            h, c = self.lstm_cell(lstm_input, (h, c))
            h = self.dropout(h)

            # ----- Spatial Attention across stations -----
            Q = self.q_proj(emb_t).unsqueeze(1)  # [batch, 1, hidden]
            K = self.k_proj(emb)                 # [batch, seq, hidden]
            V = self.v_proj(h).unsqueeze(1)      # [batch, 1, hidden] -> value is current step

            # Multi-head
            def split_heads(tensor, seq_dim=False):
                if seq_dim:
                    # [batch, seq, hidden] -> [batch, heads, seq, head_dim]
                    return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                else:
                    # [batch, 1, hidden] -> [batch, heads, 1, head_dim]
                    return tensor.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

            Qh, Kh, Vh = split_heads(Q), split_heads(K, seq_dim=True), split_heads(V)

            # Scaled dot-product attention
            scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, Vh)

            # Concatenate heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size).squeeze(1)
            attn_out = self.attn_ln(h + attn_out)
            attn_out = attn_out + self.ffn(attn_out)
            attn_out = self.dropout(attn_out)

            # Update LSTM hidden state with attention-refined output
            h = attn_out

            # Step output
            step_output = self.linear_out(attn_out)
            outputs.append(step_output.unsqueeze(1))  # [batch, 1, 1]

        # Concatenate outputs across sequence
        outputs = torch.cat(outputs, dim=1)  # [batch, seq_len, 1]
        return outputs


##
#Attention Models?

ATTENTION_MODEL_FACTORY = {
    'attention_model_1': ConcatenationEmbeddingModel
    #'attention_model_2': ImprovedConcatenationEmbeddingModel,
}

## Double Loss Attention Model
# Full Scale Graphlet
#
#
# 1) create prediction based on lstm(at, e_k) -> \hat{wt}
# 2) create refined prediction based on: lstm(at, e_k, all other \hat{wt})
# 2) create refined prediction based on: a_ij, lstm(at, e_k, a_kj*\hat{wt}) -> \hat{wt}
#
#
#
# 1) create prediction based on lstm(at, e_k) -> h_k -> \hat{wt}
# 2) Refine prediction based on lstm(at, e_k, \Sum{a_kj * h_j}) # (=> Message passing?)
#
#
#
# Top 3:
# 1) create prediction based on lstm(at, e_k) -> \hat{wt}
# 2) create a_ij
# 3) Select top 3: a_ij: lstm(at, e_k, \hat{wt}_1, \hat{wt}_2, \hat{wt}_3)


## Baseline Attention Model
# 