import torch

from .spk_layers import SurrogateHeaviside, DenseSpikingConv2DLayer, DenseSpikingReadoutLayer

class DenseSNN(torch.nn.Module):
    
    def __init__(self, w_means=[.0,.0,.0,.0,.0], w_stds=[.15,.15,.15,.15,.15], betas=[.7,.7,.7,.7,.7], bs=[1.,1.,1.,1.,0.]):
        super(DenseSNN, self).__init__()
        
        self.spike_fn = SurrogateHeaviside.apply
        
        self.c1 = DenseSpikingConv2DLayer(in_channels=1, out_channels=4, kernel_size=(5,5), 
                                         dilation=(1,1), stride=(2,2), padding=(0,0),
                                         spike_fn=self.spike_fn, recurrent=False, lateral_connections=False, 
                                         w_init_mean=w_means[0], w_init_std=w_stds[0], beta_init=betas[0], b_init=bs[0])
        self.c2 = DenseSpikingConv2DLayer(in_channels=4, out_channels=8, kernel_size=(5,5), 
                                         dilation=(1,1), stride=(2,2), padding=(0,0),
                                         spike_fn=self.spike_fn, recurrent=False, lateral_connections=False, 
                                         w_init_mean=w_means[1], w_init_std=w_stds[1], beta_init=betas[1], b_init=bs[1])
        self.c3 = DenseSpikingConv2DLayer(in_channels=8, out_channels=8, kernel_size=(3,3), 
                                         dilation=(1,1), stride=(2,2), padding=(0,0),
                                         spike_fn=self.spike_fn, recurrent=False, lateral_connections=False,  
                                         w_init_mean=w_means[2], w_init_std=w_stds[2], beta_init=betas[2], b_init=bs[2])
        self.c4 = DenseSpikingConv2DLayer(in_channels=8, out_channels=16, kernel_size=(3,3), 
                                         dilation=(1,1), stride=(2,2), padding=(0,0),
                                         spike_fn=self.spike_fn, recurrent=False, lateral_connections=False, 
                                         w_init_mean=w_means[3], w_init_std=w_stds[3], beta_init=betas[3], b_init=bs[3])

        self.linear = DenseSpikingReadoutLayer(input_shape=16*6*6, output_shape=11,
                                            spike_fn=self.spike_fn, lateral_connections=False, 
                                     w_init_mean=w_means[-1], w_init_std=w_stds[-1], beta_init=betas[-1], b_init=bs[-1])

    def forward(self, x, scale=3.0,nb_steps=None):
        if nb_steps is None:
            nb_steps = x.shape[2]
        batch_size = x.shape[0]
        mem = [None, None, None, None]
        outs = torch.zeros(batch_size,nb_steps, 11, device=x.device)
        
        for t in range(nb_steps):
            out,mem[0] = self.c1(x[:,:,t,:,:], scale, mem[0])
            out,mem[1] = self.c2(out, scale, mem[1])
            out,mem[2] = self.c3(out, scale, mem[2])
            out,mem[3] = self.c4(out, scale, mem[3])
            out = out.flatten(start_dim=1)
            out,_ = self.linear(out)
            outs[:,t,:] = out
        
        return outs.mean(dim=1)
    
    def clamp(self):
        self.c1.clamp()
        self.c2.clamp()
        self.c3.clamp()
        self.c4.clamp()
        self.linear.clamp()