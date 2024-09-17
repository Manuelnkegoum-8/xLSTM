from .layers import *
import torch 
import torch.nn as nn
import torch.nn.functional as F 


class mLSTM_cell(nn.Module):
    def __init__(self,dim=128,heads=4):
        super(mLSTM_cell,self).__init__()
        assert (
            dim%heads == 0
        ), "Embedding size needs to be divisible by heads"
        self.dim = dim
        self.tau = dim**0.5

        self.heads = heads
        self.head_dim = dim//heads
        self.Wq = nn.Linear(dim,dim,bias=True)
        self.Wk = nn.Linear(dim,dim,bias=True)
        self.Wv = nn.Linear(dim,dim,bias=True)
        self.Wi = nn.Linear(dim,heads,bias=True)
        self.Wf = nn.Linear(dim,heads,bias=True)
        self.group_norm = nn.GroupNorm(heads,self.dim)

    def create_input_gate(self,seq,itilde,bs):
        lower_mask_i = torch.tril(torch.ones((seq,seq),device=itilde.device))
        I_act = itilde.unsqueeze(-2)* lower_mask_i
        return I_act

    def create_forget_gate(self,seq,ftilde,bs):
        lower_mask = torch.tril(torch.ones((seq,seq),device=ftilde.device)).bool()
        tmp = F.logsigmoid(ftilde).unsqueeze(-1)
        uu = torch.cat(
        [
            torch.zeros((bs, self.heads, 1, 1),device=ftilde.device,dtype=ftilde.dtype),
            torch.cumsum(tmp, dim=-2),
        ],
        dim=-2,
        )  # (B, H, T+1, 1)
        # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
        # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
        # First entry of each row is zero.
        rep_log_fgates_cumsum = uu.repeat(1, 1, 1, seq + 1)  # (B, NH, S+1, S+1)
        # Now in each row cut off / subtract the forgetgate values of the later timesteps
        # where col j > row i
        log_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
        F_act = torch.where(lower_mask, log_matrix[:, :, 1:, 1:], -float("inf")) 
        return F_act


    def forward(self,q,k,v):
        # x bs,T,dim
        bs,seq,_ = q.size()
        itilde = self.Wi(q).permute(0,2,1) #bs*h*T
        ftilde = self.Wf(k).permute(0,2,1) # bs*h*T
       
        
        F_act = self.create_forget_gate(seq,ftilde,bs)
        # Create a lower triangular mask (including diagonal)
        I_act = self.create_input_gate(seq,itilde,bs)
        Dtilde = F_act + I_act
        max_d = torch.max(Dtilde,dim=-1, keepdim=True)[0]
        D = torch.exp(Dtilde - max_d)

        queries = self.Wq(q).view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxh*Txdim
        keys = self.Wk(k).view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim
        values = self.Wv(v).view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim

        Ctilde = torch.matmul(queries, keys.transpose(-2,-1)) * D/self.tau # # bsxhxTxT
        maxit = Ctilde.sum(dim=-1)
        maxit = torch.max(torch.abs(maxit),torch.exp(-max_d).squeeze(-1))
        Htilde = (Ctilde@values)/(maxit.unsqueeze(-1) + 1e-8)
        Htilde = Htilde.permute(0,2,1,3).contiguous().view(bs*seq,self.dim)
        Htilde = self.group_norm(Htilde).view(bs,seq,self.dim) #mult head group ,orm
        return Htilde



class mLSTM_block(nn.Module):
    def __init__(self,dim=128,heads=4):
        super(mLSTM_block,self).__init__()
        self.cell = mLSTM_cell(dim*2,heads)
        self.block_q = BlockDiagonalLinear(dim*2,dim*2,4,False)
        self.block_k = BlockDiagonalLinear(dim*2,dim*2,4,False)
        self.block_v = BlockDiagonalLinear(dim*2,dim*2,4,False)
        self.first_norm = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim,2*dim)
        self.mlp2 = nn.Linear(dim,2*dim)
        self.causal_conv = CausalConv1d(dim*2,dim*2,4)
        self.swish = SwishActivation(learnable=False)
        
        self.final_mlp = nn.Linear(dim*2,dim)
        self.learnable_skip = nn.Parameter(torch.ones(dim*2, requires_grad=True))
    
    def forward(self,x):
        inputs = x.clone()
        inputs = self.first_norm(inputs)
        inputs_a = self.mlp1(inputs)
        inputs_b = self.swish(self.mlp2(inputs))
        qk = self.causal_conv(inputs_a.permute(0,2,1)).permute(0,2,1)
        qk = self.swish(qk)
        q = self.block_q(qk)
        k = self.block_k(qk)
        v = self.block_v(inputs_a)
        out1 = self.cell(q,k,v)
        out1 = out1 + self.learnable_skip*qk
        out1 = out1 * inputs_b
        out1 = self.final_mlp(out1)
        out = x+out1
        return out