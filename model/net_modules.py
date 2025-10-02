import torch
import torch.nn as nn
import numpy as np


class MLPNet(nn.Module):

    def __init__(self, activation='lrelu', input_size = (699+1024) * 3, hid_layer = [1024, 512, 256], output_size = 128, #opt['dims'],
                 batch_size=4, use_gpu=0, layer='UpperClothes', weight_norm=True, dropout=0.3):
        
        super().__init__()
        input_size = input_size #opt['in_dim']
        hid_layer = hid_layer  #opt['dims']
        output_size = output_size #opt['out_dim']

        if hid_layer is not None:
            dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        else:
            dims = [input_size] + [output_size]
    

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'lrelu':    
            self.actv = nn.LeakyReLU()
            self.out_actv = nn.ReLU()
        
        if activation == 'relu':    
            self.actv = nn.ReLU()
            self.out_actv = nn.ReLU()

        if activation == 'mish':    
            self.actv = nn.Mish()
            self.out_actv = None# = nn.ReLU()

        if activation == 'silu':    
            self.actv = nn.SiLU()
            self.out_actv = None# = nn.ReLU()
        
        #if activation == 'softplus':    
        #    self.actv = nn.Softplus(beta=opt['beta'])
        #    self.out_actv = nn.Softplus(beta=opt['beta'])

        if activation == 'none':  
            self.out_actv = None
        

    def forward(self, x):


       # x = p.reshape(len(p), -1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            # if  lin.weight.isnan().any():
            #     ipdb.set_trace()
            # if  lin.bias.isnan().any():
            #     ipdb.set_trace()
            # if  x.isnan().any():
            #     ipdb.set_trace()
            
            x = lin(x)
            # if  tmp.isnan().any():
            #     ipdb.set_trace()
            # x = tmp
            if l < self.num_layers - 2:
                x =  self.actv(x)

        # if self.squeeze_out:
        #     x = torch.sigmoid(x)
        if self.out_actv is not None:
            x =  self.out_actv(x)
        # if not torch.all(x>0):
        #     ipdb.set_trace()
        return x