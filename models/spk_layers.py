import torch
import numpy as np

# modified from S2Net by Romain Zimmer (https://github.com/romainzimmer/s2net/blob/master/models.py)
            
import torch
import numpy as np

class DenseSpikingConv2DLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std,  b_init=1., beta_init=0.7, recurrent=False,
                 sigma=10.0, lateral_connections=False, padding=(0,0),
                 eps=1e-8, stride=(1,1),flatten_output=False):
        
        super(DenseSpikingConv2DLayer, self).__init__()
        
        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.padding = np.array(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.spike_fn = spike_fn
        self.sigma = sigma
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps
        
        self.flatten_output = flatten_output
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.b_init = b_init
        self.beta_init = beta_init
        
        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x, scale, mem=None):
        
        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(self.padding),
                                      dilation=tuple(self.dilation),
                                      stride=tuple(self.stride))
        
        batch_size, output_shape = conv_x.shape[0], (conv_x.shape[-2], conv_x.shape[-1])
        
        if mem is None:
            mem = torch.zeros(x.size(0), self.out_channels, *output_shape, dtype=x.dtype, device=x.device)
        
        if self.lateral_connections:
            d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)  
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1,*output_shape))   
            
        norm = (self.w**2).sum((1,2,3))
            
        input_ = conv_x
            
        # membrane potential update
        mem = mem*self.beta + input_*(1.-self.beta)
        mthr = torch.einsum("abcd,b->abcd",mem, 1./(norm+self.eps))-b 
                
        spk = self.spike_fn(mthr, scale)
        
        mem = mem - torch.einsum("abcd,b,b->abcd", spk, self.b, norm)
        
        return spk, mem
   
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std)
        torch.nn.init.normal_(self.beta, mean=self.beta_init, std=0.01)
        torch.nn.init.normal_(self.b, mean=self.b_init, std=0.01)
    
    def clamp(self):
        self.b.data.clamp_(min=0.)
        self.beta.data.clamp_(0.,1.)

        
class DenseSpikingReadoutLayer(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 in_channels=0, sigma=10.0, recurrent=False, lateral_connections=True, eps=1e-8, b_init=1., beta_init=0.7, time_reduction="mean"):
        
        super(DenseSpikingReadoutLayer, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = in_channels
        self.spike_fn = spike_fn
        self.sigma = sigma
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections
        self.time_reduction = time_reduction

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.beta_init = beta_init
        self.b_init = b_init
        
        if in_channels == 0:
            self.w = torch.nn.Parameter(torch.Tensor(input_shape, output_shape))
        else:
            self.w = torch.nn.Parameter(torch.Tensor(in_channels, input_shape, output_shape))
            
        if recurrent:
            self.v = torch.nn.Parameter(torch.Tensor((output_shape, output_shape)))
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()
        
        self.training = True
        
    def forward(self, x, mem=None):
        
        batch_size = x.shape[0]
        
        if self.in_channels == 0:
            h = torch.einsum("ab,bc->ac", x, self.w)
            norm = (self.w**2).sum(0)
        else:
            h = torch.einsum("abc,bcd->ad", x, self.w)
            norm = (self.w**2).sum((0,1))
            
        input_ = h
       
        # membrane potential update
        if self.time_reduction == "max":
            if mem==None:
                mem = input_*(1-self.beta)
            else:
                mem = mem*self.beta + input_*(1.-self.beta)
                
            out = mem/(norm+1e-8) - self.b
            
        elif self.time_reduction == "mean":
            out = h/(norm+1e-8) - self.b
    
        return out.to(x.dtype), mem
   
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.input_shape)))
        
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            
        torch.nn.init.normal_(self.b, mean=self.b_init, std=0.01)
    
    def clamp(self):
        self.b.data.clamp_(min=0.)
        if self.time_reduction == "max":
            self.beta.data.clamp_(0.,1.)


class DenseSpikingReadoutLayer(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 in_channels=0, sigma=10.0, recurrent=False, lateral_connections=True, eps=1e-8, b_init=1., beta_init=0.7, time_reduction="mean"):
        
        super(DenseSpikingReadoutLayer, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = in_channels
        self.spike_fn = spike_fn
        self.sigma = sigma
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections
        self.time_reduction = time_reduction

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.beta_init = beta_init
        self.b_init = b_init
        
        if in_channels == 0:
            self.w = torch.nn.Parameter(torch.Tensor(input_shape, output_shape))
        else:
            self.w = torch.nn.Parameter(torch.Tensor(in_channels, input_shape, output_shape))
            
        if recurrent:
            self.v = torch.nn.Parameter(torch.Tensor((output_shape, output_shape)))
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()
        
        self.training = True
        
    def forward(self, x, mem=None):
        
        batch_size = x.shape[0]
        
        if self.in_channels == 0:
            h = torch.einsum("ab,bc->ac", x, self.w)
            norm = (self.w**2).sum(0)
        else:
            h = torch.einsum("abc,bcd->ad", x, self.w)
            norm = (self.w**2).sum((0,1))
            
        input_ = h
       
        # membrane potential update
        if self.time_reduction == "max":
            if mem==None:
                mem = input_*(1-self.beta)
            else:
                mem = mem*self.beta + input_*(1.-self.beta)
                
            out = mem/(norm+1e-8) - self.b
            
        elif self.time_reduction == "mean":
            out = h/(norm+1e-8) - self.b
    
        return out.to(x.dtype), mem
   
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.input_shape)))
        
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            
        torch.nn.init.normal_(self.b, mean=self.b_init, std=0.01)
    
    def clamp(self):
        self.b.data.clamp_(min=0.)
        if self.time_reduction == "max":
            self.beta.data.clamp_(0.,1.)


class ReadoutLayer(torch.nn.Module):
    
    "Fully connected readout"
    
    def __init__(self,  input_shape, output_shape, w_init_mean, w_init_std, eps=1e-8, time_reduction="mean"):
        
        
        assert time_reduction in ["mean", "max"], 'time_reduction should be "mean" or "max"'
        
        super(ReadoutLayer, self).__init__()
        

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        
        self.eps = eps
        self.time_reduction = time_reduction
        
        
        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7*np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()
        
        self.mem_rec_hist = None
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
       
        h = torch.einsum("abc,cd->abd", x, self.w)
        
        norm = (self.w**2).sum(0)
        
        if self.time_reduction == "max":
            nb_steps = x.shape[1]
            # membrane potential 
            mem = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)

            # memrane potential recording
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)

            for t in range(nb_steps):

                # membrane potential update
                mem = mem*self.beta + (1-self.beta)*h[:,t,:]
                mem_rec[:,t,:] = mem
                
            output = torch.max(mem_rec, 1)[0]/(norm+1e-8) - self.b
            
        elif self.time_reduction == "mean":
            
            mem_rec = h
            output = torch.mean(mem_rec, 1)/(norm+1e-8) - self.b
        
        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        
        loss = None
        
        return output, loss
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.input_shape)))
        
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self, min_beta=0., max_beta=1.):
        
        if self.time_reduction == "max":
            self.beta.data.clamp_(min_beta,max_beta)

class SurrogateHeaviside(torch.autograd.Function):

    @staticmethod 
    def forward(ctx, input, scale=3.0):
        ctx.scale = scale
        ctx.save_for_backward(input)
        
        output = torch.zeros_like(input, dtype=input.dtype)
        output[input > 0] = 1.0
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
#         grad = grad_input*torch.sigmoid(ctx.scale*input)*torch.sigmoid(-ctx.scale*input)
        grad = grad_input/(ctx.scale*torch.abs(input)+1.0)**2
        return grad, None