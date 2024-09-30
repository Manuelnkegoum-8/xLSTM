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
        

        self.heads = heads
        self.head_dim = dim//heads
        self.tau = self.head_dim**0.5
        self.Wi = nn.Linear(3*dim,heads)
        self.Wf = nn.Linear(3*dim,heads)
        self.group_norm = nn.GroupNorm(heads,self.dim)
        self.reset_parameters()

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
        in_gate = torch.cat((q,k,v),dim=-1)
        itilde = self.Wi(in_gate).permute(0,2,1) #bs*h*T
        ftilde = self.Wf(in_gate).permute(0,2,1) # bs*h*T
       
        
        F_act = self.create_forget_gate(seq,ftilde,bs)

        # Create a lower triangular mask (including diagonal)
        Dtilde = F_act + itilde.unsqueeze(-1).transpose(-2,-1)
        with torch.no_grad():
            max_d = torch.max(Dtilde,dim=-1, keepdim=True)[0]
        D = torch.exp(Dtilde - max_d)
        queries = q.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxh*Txdim
        keys = k.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim
        values = v.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim

        Ctilde = torch.matmul(queries, keys.transpose(-2,-1)) * D/self.tau # # bsxhxTxT
        maxit = Ctilde.sum(dim=-1,keepdim=True)
        maxit = torch.maximum(torch.abs(maxit),torch.exp(-max_d))
        Htilde = (Ctilde@values)/(maxit + 1e-6)
        Htilde = Htilde.permute(0,2,1,3).contiguous().view(bs*seq,self.dim)
        Htilde = self.group_norm(Htilde).view(bs,seq,self.dim) #mult head group norm
        return Htilde

    ## only for inference
    def recurrent_forward(self,state,q,k,v):
        bs,seq,_ = q.size() 
        in_gate = torch.cat((q,k,v),dim=-1)
        itilde = self.Wi(in_gate).permute(0,2,1).unsqueeze(-1) #bs*h*1*1
        ftilde = self.Wf(in_gate).permute(0,2,1).unsqueeze(-1) # bs*h*1*1

        if m is None:
            c = torch.zeros(size=(bs,self.heads,self.head_dim,self.head_dim), device=q.device)
            n = torch.zeros(size=(B, self.heads,self.head_dim,1), device=q.device)
            m = torch.zeros(size=(B, self.heads, 1, 1), device=q.device)
        else:
            c,n,m = state
        

        log_f = F.logsigmoid(ftilde)
        mt = torch.max(log_f+m,itilde) #bs*h*1*1
        it = torch.exp(itilde-mt)
        ft = torch.exp(log_f+m-mt)
        queries = q.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxh*Txdim
        keys = k.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim
        values = v.view(bs, seq, self.heads, self.head_dim).permute(0,2,1,3) # bsxhxTxdim
        queries = queries.squeeze(2).unsqueeze(-1) #make it bs*h*dim*1
        keys = keys.squeeze(2).unsqueeze(-1)/self.tau
        values = values.squeeze(2).unsqueeze(-1)

        ct = ft*c + it*torch.matmul(values, keys.transpose(-2,-1)) # bs*h*dim*dim
        nt = ft*n + it*keys #bs*h*dim*1
        normalizer = torch.maximum(torch.abs(nt.transpose(-2,-1)@qt) ,torch.exp(-mt)) #bs*h*1*1
        htilde = (ct@qt)/(normalizer+1e-6) #bs*h*dim*1
        htilde = htilde.permute(0,3,1,2).contiguous().view(bs*seq,self.dim)
        htilde = self.group_norm(htilde).view(bs,seq,self.dim)
        return htilde,(ct,nt,mt)

    def reset_parameters(self):
        # forget gate initialization
        torch.nn.init.zeros_(self.Wf.weight)
        equidistant_bias_init(self.Wf.bias,3.0,6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.Wi.weight)
        torch.nn.init.normal_(self.Wi.bias, mean=0.0, std=0.1)


class mLSTM_block(nn.Module):
    def __init__(self,dim=128,heads=4):
        super(mLSTM_block,self).__init__()
        self.cell = mLSTM_cell(dim*2,heads)
        self.block_q = BlockDiagonalLinear(dim*2,dim*2,heads,True)
        self.block_k = BlockDiagonalLinear(dim*2,dim*2,heads,True)
        self.block_v = BlockDiagonalLinear(dim*2,dim*2,heads,True)
        self.first_norm = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim,2*dim)
        self.mlp2 = nn.Linear(dim,2*dim)
        self.causal_conv = CausalConv1d(dim*2,dim*2,heads)
        self.swish = SwishActivation(learnable=False)
        self.final_mlp = nn.Linear(dim*2,dim)
        self.learnable_skip = nn.Parameter(torch.ones(dim*2, requires_grad=True))
        self.dropout = nn.Dropout(0.2)
    
    
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
        out = self.dropout(self.final_mlp(out1))
        return out
    
    def step(self,x,state):
        inputs = x.clone()
        inputs = self.first_norm(inputs)
        inputs_a = self.mlp1(inputs)
        inputs_b = self.swish(self.mlp2(inputs))
        qk = self.causal_conv(inputs_a.permute(0,2,1)).permute(0,2,1)
        qk = self.swish(qk)
        q = self.block_q(qk)
        k = self.block_k(qk)
        v = self.block_v(inputs_a)
        out1, new_state = self.cell.recurrent_forward(state,q,k,v)
        out1 = out1 + self.learnable_skip*qk
        out1 = out1 * inputs_b
        out = self.dropout(self.final_mlp(out1))
        return out,new_state