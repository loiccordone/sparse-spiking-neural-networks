import torch
from torch import nn

from .spk_layers import SurrogateHeaviside, DenseSpikingReadoutLayer
from .sparse_spk_layers import SparseSpikingConv2D

class SparseSNN(torch.nn.Module):
    
    def __init__(self, scale=3.):
        super(SparseSNN, self).__init__()
        
        self.scale = scale
        self.spike_fn = SurrogateHeaviside.apply
        
        self.c1 = SparseSpikingConv2D(
            in_channels=1, out_channels=4, kernel=(5,5),
            out_shape=(64,64), stride=(2,2), 
        )
        self.c2 = SparseSpikingConv2D(
            in_channels=4, out_channels=8, kernel=(5,5),
            out_shape=(32,32), stride=(2,2), 
        )
        self.c3 = SparseSpikingConv2D(
            in_channels=8, out_channels=8, kernel=(3,3),
            out_shape=(16,16), stride=(2,2), 
        )
        self.c4 = SparseSpikingConv2D(
            in_channels=8, out_channels=16, kernel=(3,3),
            out_shape=(8,8), stride=(2,2), return_dense=True,
        )
        
        self.dropout = nn.Dropout(0.5)

        self.linear = DenseSpikingReadoutLayer(
            input_shape=16*8*8, output_shape=11,
            spike_fn=SurrogateHeaviside.apply, lateral_connections=False, 
            w_init_mean=0., w_init_std=0.15, beta_init=0.7, b_init=0.)
        
    def forward(self, x):
        nb_steps = len(x)
        bs = int(torch.max(x[0].C[:,0]))+1
        mem = [None, None, None, None]
        outs = []
        
        for t in range(nb_steps):
            out,mem[0] = self.c1(x[t], mem=mem[0], scale=self.scale, bs=bs)
            out,mem[1] = self.c2(out, mem=mem[1], scale=self.scale, bs=bs)
            out,mem[2] = self.c3(out, mem=mem[2], scale=self.scale, bs=bs)
            out,mem[3] = self.c4(out, mem=mem[3], scale=self.scale, bs=bs)
            out = out.flatten(start_dim=1)
            out = self.dropout(out)
            out,_ = self.linear(out)
            outs.append(out)
            
        return torch.stack(outs, dim=1).mean(dim=1)
    
    def clamp(self):
        self.c1.clamp()
        self.c2.clamp()
        self.c3.clamp()
        self.c4.clamp()
        self.linear.clamp()