import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int,dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create matrix of shape (seq_len, d_model) to store positional encoding
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # apply the sin to even and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add a batch dimension to positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)    
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,eps:float =10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplier
        self.bias = nn.Parameter(torch.zeros(1)) # Add
        
    def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff : int,dropout : float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        # check if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model) # output layer
    
    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        d_k = q.shape[-1]
        # q shape: (batch_size, num_heads, seq_len, d_k)-> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (q @ k.transpose(-2,-1)) / (d_k ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ v, attention_scores
        
    
        
    def forward(self, query, key, value, mask=None):
        
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # query shape: (batch_size, seq_len, d_model) --> (batch_size, num_heads, seq_len, d_k) -> (batch_size, num_heads, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(1,2)
        k = k.view(k.shape[0],k.shape[1],self.num_heads,self.d_k).transpose(1,2)
        v = v.view(v.shape[0],v.shape[1],self.num_heads,self.d_k).transpose(1,2)
        
        x,self.attention_scores = MultiheadAttention.attention(q,k,v,mask, self.dropout)
        
        # x shape: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.num_heads * self.d_k)
        
        # x shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.w_o(x)

class ResidualBlock(nn.Module):
    def __init__(self,  dropout: float) -> None:
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention,feed_forward_block:FeedForwardBlock,dropout: float) -> None:
        super().__init__()
        self.multihead_attention = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualBlock(dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.multihead_attention(x,x,x,mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = num_layers
        self.layer_norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention,cross_attention_block:MultiheadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualBlock(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # batch_size, seq_len, d_model -> batch_size, seq_len, vocab_size
        return torch.log_softmax(self.project(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder,decoder:Decoder,src_emd : InputEmbedding,tgt_emd : InputEmbedding,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,project_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emd = src_emd
        self.tgt_emd = tgt_emd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.project_layer = project_layer
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_emd(src)), src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_emd(tgt)), encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.project_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len : int,tgt_seq_len : int, d_model :int = 512,N : int =6,num_heads:int =8,dropout: float = 0.1,d_ff:int = 2048) -> Transformer:
    src_emd = InputEmbedding(src_vocab_size, d_model)
    tgt_emd = InputEmbedding(tgt_vocab_size, d_model)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    project_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    encoder = Encoder(nn.ModuleList([EncoderBlock(MultiheadAttention(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)]))
    
    decoder = Decoder(nn.ModuleList([DecoderBlock(MultiheadAttention(d_model, num_heads, dropout), MultiheadAttention(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)]))
    
    transformer = Transformer(encoder, decoder, src_emd, tgt_emd, src_pos, tgt_pos, project_layer)
    
    # Initialise the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    
    return transformer