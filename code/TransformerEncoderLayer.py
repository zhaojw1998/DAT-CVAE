import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from torch.nn import Transformer


class MultiheadSelfAttentionwithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0, max_len=128):
        super(MultiheadSelfAttentionwithRelativePositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.L = 2 * max_len - 1
        self.num_heads = num_heads
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)

        self.quaver_Er = nn.Parameter(torch.randn(num_heads, self.L, self.head_dim))
        self.beat_Er = nn.Parameter(torch.randn(num_heads, self.L//4, self.head_dim))
        self.bar_Er = nn.Parameter(torch.randn(num_heads, self.L//16, self.head_dim))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        #x: (batch, src_len, dmodel)
        #key_padding_mask: (batch, num_head, src_len, src_len), bool tensor
        #attn_mask:  (batch, num_head, src_len, src_len): float tensor
        bs, src_len, d_model = query.shape

        q = self.query(query).reshape(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, src_len, head_dim)
        k = self.key(key).reshape(bs, src_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  #(batch, num_head, head_dim, src_len)
        v = self.value(value).reshape(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, src_len, head_dim)

        
        Er_t_quaver = self.quaver_Er[:, self.max_len-src_len: self.max_len+src_len, :].transpose(-2, -1)   #(num_head, head_dim, 2*src_len-1)
        Er_t_beat = self.beat_Er[:, (self.max_len-src_len)//4: (self.max_len+src_len)//4, :].transpose(-2, -1)   #(num_head, head_dim, 2*src_len//4-1)
        Er_t_bar = self.bar_Er[:, (self.max_len-src_len)//16: (self.max_len+src_len)//16, :].transpose(-2, -1)   #(num_head, head_dim, 2*src_len//16-1)

        QEr_quaver = torch.matmul(q, Er_t_quaver) #(num_head, num_head, src_len, 2*src_len-1)
        QEr_beat = torch.matmul(q, Er_t_beat) #(num_head, num_head, src_len, 2*src_len//4-1)
        QEr_bar = torch.matmul(q, Er_t_bar) #(num_head, num_head, src_len, 2*src_len//16-1)
        #print(QEr[0, 0])
        Srel = self.skew(QEr_quaver) #(num_head, num_head, src_len, src_len)
        Srel_beat = self.skew_beat(QEr_beat) #(num_head, num_head, src_len, src_len)
        Srel_bar = self.skew_bar(QEr_bar) #(num_head, num_head, src_len, src_len)


        if key_padding_mask is not None:
            #print(key_padding_mask.shape)
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
            else:
                attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        attn = (torch.matmul(q, k) + Srel+Srel_beat+Srel_bar) / math.sqrt(self.head_dim) #(batch, num_head, src_len, src_len)
        #attn = (torch.matmul(q, k) + Srel) / math.sqrt(self.head_dim) #(batch, num_head, src_len, src_len)
        
        if attn_mask is not None:
            #print(attn.shape, attn_mask.shape)
            attn += attn_mask
            #for i in range(attn.shape[0]):
            #    print(attn_mask[i, 0])
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, tgt_len, head_dim)
        out = out.transpose(1, 2).reshape(bs, src_len, d_model) #(batch, tgt_len, d_model)

        return self.dropout(out), attn
        
    def skew(self, QEr):
        #QEr: (batch, num_heads, src_len, src_L), src_L= 2*src_len-1
        bs, num_heads, src_len, src_L = QEr.shape
        QEr = F.pad(QEr, (0, 1))    #(batch, num_heads, src_len, src_L+1)
        QEr = QEr.reshape(bs, num_heads, -1)   #(batch, num_heads, src_len*(src_L+1))
        QEr = F.pad(QEr, (0, src_L-src_len))    #(batch, num_heads, (src_len+1)*src_L)
        QEr = QEr.reshape(bs, num_heads, src_len+1, src_L)
        QEr = QEr[:, :, :src_len, -src_len:]    #(batch, num_heads, src_len, src_len)
        return QEr
    
    def skew_beat(self, QEr):
        #QEr: (batch, num_heads, src_len, 2*src_len//4-1)
        _, _, src_len, _ = QEr.shape
        QEr = torch.repeat_interleave(QEr, repeats=4, dim=-1)
        for i in range(src_len//4):
            QEr[:, :, i*4: (i+1)*4, :] = torch.roll(QEr[:, :, i*4: (i+1)*4, :], shifts=i*4, dims=-1)
        QEr = QEr[:, :, :src_len, -src_len:]
        return QEr

    def skew_bar(self, QEr):
        #QEr: (batch, num_heads, src_len, 2*src_len//16-1)
        _, _, src_len, _ = QEr.shape
        QEr = torch.repeat_interleave(QEr, repeats=16, dim=-1)
        for i in range(src_len//16):
            QEr[:, :, i*16: (i+1)*16, :] = torch.roll(QEr[:, :, i*16: (i+1)*16, :], shifts=i*16, dims=-1)
        QEr = QEr[:, :, :src_len, -src_len:]
        return QEr
    


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, norm_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttentionwithRelativePositionalEmbedding(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        #src: (batch, len, dmodel)
        #key_padding_mask: (batch, num_head, src_len, src_len), bool tensor
        #attn_mask:  (batch, num_head, src_len, src_len): float tensor
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



if __name__ == '__main__':
    model = TransformerEncoderLayer(64, 4, 128, .1)
    tgt = torch.ones(4, 128, 64)
    out = model(tgt)
    print(out.shape)
