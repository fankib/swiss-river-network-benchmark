import torch
import torch.nn as nn

from swissrivernetwork.benchmark.lstm_embedding import VanillaLSTM, EmbeddingGateLSTMCell, EmbeddingCellLSTM, InterpolationLstmCell

##################
# Model Variants #
##################

class TrafficConcatenationEmbeddingModel(nn.Module):

    def __init__(self, input_size, output_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = VanillaLSTM(input_size+embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2) # x in [batch x sequence x features]
        out, (h_n, c_n) = self.lstm(x)        
        target = self.linear(h_n)
        return target

class TrafficEmbeddingGateMemoryModel(nn.Module):

    def __init__(self, input_size, output_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = EmbeddingGateLSTMCell(input_size, hidden_size, embedding_size, True)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, output_size)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, (h_n, c_n) = self.lstm(x, emb)        
        target = self.linear(h_n)
        return target

class TrafficEmbeddingGateHiddenModel(nn.Module):
    
    def __init__(self, input_size, output_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = EmbeddingGateLSTMCell(input_size, hidden_size, embedding_size, False)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, output_size)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, (h_n, c_n) = self.lstm(x, emb)        
        target = self.linear(h_n)
        return target

class TrafficInterpolationEmbeddingModel(nn.Module):
    
    def __init__(self, input_size, output_size, num_embeddings, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        cell = InterpolationLstmCell(input_size, hidden_size, embedding_size)
        self.lstm = EmbeddingCellLSTM(hidden_size, cell)
        self.linear = nn.Linear(hidden_size, output_size)    
    
    def forward(self, e, x, emb=None):
        if emb is None:
            emb = self.embedding(e) # e in [batch x sequence x station_id]        
        out, (h_n, c_n) = self.lstm(x, emb)        
        target = self.linear(h_n)
        return target


### Model Registry ###

EMBEDDING_MODEL_FACTORY = {
    'concatenation_embedding': TrafficConcatenationEmbeddingModel,
    'embedding_gate_memory': TrafficEmbeddingGateMemoryModel,
    'embedding_gate_hidden': TrafficEmbeddingGateHiddenModel,
    'interpolation_embedding': TrafficInterpolationEmbeddingModel
}
