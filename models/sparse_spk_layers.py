import torch
from torch import nn
import numpy as np
import MinkowskiEngine as ME

from .spk_layers import SurrogateHeaviside

class SparseSpikingConv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel, out_shape, stride, return_dense=False, bias=False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.out_shape = out_shape
        self.kernel = kernel
        
        self.spike_fn = SurrogateHeaviside.apply
        self.return_dense = return_dense
        self.eps = 1e-8

        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel, 
                                            stride=stride, bias=False, dimension=2)
        
        self.reset_parameters()
        
    def forward(self, input, mem, bs, scale=1.):
#         self.clamp()
        
        conv_sparse = self.conv(input)
        conv_dense = conv_sparse.dense(
            shape=torch.Size([bs, self.out_channels, *self.out_shape])
        )[0]
        
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1,*self.out_shape))
        
        norm = (self.conv.kernel**2).sum((0,1))
        
        if mem is None:
            mem = torch.zeros((bs, self.out_channels, *self.out_shape))
            mem = mem.type_as(input.C)
        
        new_mem = mem*self.beta + conv_dense*(1.-self.beta)
        
        mthr = torch.einsum("abcd,b->abcd", new_mem, 1./(norm+self.eps))-b
        spk = self.spike_fn(mthr, scale)
        
        final_mem = new_mem - torch.einsum("abcd,b,b->abcd", spk, self.b, norm)
        
        if self.return_dense:
            return spk, final_mem
        else:
            p_spkF = spk.permute(1,0,2,3).contiguous().view(self.out_channels,-1).t()
            spkF = p_spkF[p_spkF.sum(dim=1) != 0]

            spkC_temp = torch.nonzero(spk)[:,(0,2,3)]
            spkF_temp = torch.zeros((spkC_temp.shape[0],))
            spkF_temp = spkF_temp.type_as(input.C)
            torch_sparse_tensor = torch.sparse_coo_tensor(
                spkC_temp.t().to(torch.int8), 
                spkF_temp.to(torch.int8), 
            ).coalesce()
            spkC = torch_sparse_tensor._indices().t().contiguous().to(torch.int)

            final_spk = ME.SparseTensor(spkF, spkC)

            return final_spk, final_mem
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.beta, mean=0.8, std=0.01)
        torch.nn.init.normal_(self.b, mean=0.1, std=0.01)
        torch.nn.init.xavier_uniform_(self.conv.kernel.data, torch.nn.init.calculate_gain('sigmoid'))
    
    def clamp(self, min_beta=0., max_beta=1., min_b=0.):
        self.beta.data.clamp_(min_beta,max_beta)
        self.b.data.clamp_(min=min_b)
    