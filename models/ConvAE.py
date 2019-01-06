import numpy as np
# Torch
import torch
from torch import nn
from torchvision import utils

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ConvAE(nn.Module):

    modules = {"conv1": lambda D_in, D_out: nn.Conv2d(D_in, D_out, kernel_size=3, stride=1, padding=1, bias=False),
               "relu1": nn.ReLU,
               "pool1": lambda: nn.MaxPool2d(2, 2, 0, return_indices=True),
               "linear1": lambda N_in, N_out: nn.Linear(N_in, N_out, bias=True),
               "flatten1": Reshape,
               "conv-strided1": lambda D_in, D_out: nn.Conv2d(D_in, D_out, kernel_size=3, 
                                                              stride=2, padding=1, bias=False),
               
               "de-conv1": lambda D_in, D_out: nn.ConvTranspose2d(D_in, D_out, kernel_size=3, stride=1, padding=1, bias=False),
               "de-relu1": nn.ReLU,
               "de-pool1": lambda: nn.MaxUnpool2d(2, 2),
               "de-linear1": lambda N_in, N_out: nn.Linear(N_in, N_out, bias=True),
               "de-flatten1": Reshape,
               "de-conv-strided1": lambda D_in, D_out: nn.ConvTranspose2d(D_in, D_out, kernel_size=3, 
                                                                          stride=2, padding=1, bias=False),
              }
    
    def __init__(self, input_dim, 
                 enc_config = [("conv1", 16), ("pool1", None), ("flatten1", None), ("linear1", 4096)],
                 verbose=False, disable_decoder=False, n_cuda_devices=8, store_activations=False, 
                 states_file=None, ae_skip_layers=0):
        """
        @params:
            ae_skip_layers: Specifies number of layers to skip from the bottom of encoder and top of decoder. 
        """
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.verbose = verbose
        self.n_cuda_devices = n_cuda_devices
        self.store_activations = store_activations
        self.disable_decoder = disable_decoder
        self.ae_skip_layers = ae_skip_layers
        
        self.enc_activations = []
        self.dec_activations = []
        
        # Encoder
        self.enc_layers, self.enc_layer_names, self.enc_layer_dims = self.prepare_encoder(input_dim, enc_config)
        self.encoder = nn.ModuleList(self.enc_layers[1:])
        self.code_size = self.enc_layer_dims[-1]
        
        if verbose: print("Enc dims: ", self.enc_layer_dims)
        
        # Decoder
        if not disable_decoder:
            self.dec_layers, self.dec_layer_names, self.dec_layer_dims = self.prepar_decoder(
                self.enc_layer_names, self.enc_layer_dims)
            self.decoder = nn.ModuleList(self.dec_layers)
            self.tanh = nn.Tanh()
            if verbose: print("Dec dims: ", self.dec_layer_dims)
                
        if states_file is not None:
            print("Loading states from: {}".format(states_file))
            # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032/1
            # Loading weights for CPU model while trained on GPU
            self.load_state_dict(torch.load(states_file, map_location=lambda storage, loc: storage))
        
    def forward(self, images):
        
        code, pool_idxs = self.encode(images, ret_pool_idxs=True, skip_layers=False)
        if self.disable_decoder:
            return code
        else:
            out = self.tanh(self.decode(code, pool_idxs=pool_idxs, skip_layers=False))
            return out, code
    
    def prepare_encoder(self, input_dim, enc_config):
        
        layers = [None]
        layer_names = ["input"]
        layer_dims = [input_dim]
        #  conv_device_id = -1
        
        for conf in enc_config:
            
            if self.verbose: print("{} -> {}".format(input_dim, conf))
            
            if len(input_dim) == 1:
                # flat
                layer_name, N_out = conf
                out_dim = (N_out, None)
            else:
                # volume
                layer_name, D_out = conf
                D, H, W = input_dim
                out_dim = (D_out, H, W)
            
            layer, layer_name, out_dim = self.create_layer(layer_name, out_dim, input_dim)
            
            #if layer_name.startswith("conv"):
            #    conv_device_id = (conv_device_id + 1) % self.n_cuda_devices
            #    layers.append(layer.cuda(conv_device_id))
            layers.append(layer)
            layer_names.append(layer_name)
            layer_dims.append(out_dim)
            input_dim = out_dim
            
        return layers, layer_names, layer_dims
    
    def prepar_decoder(self, layer_names, layer_dims):
        
        dec_layer_names = ["de-"+l for l in layer_names[::-1]]
        dec_layer_dims = layer_dims[::-1]
        layers = []
        layer_names = []
        layer_dims = []
        
        for i, lname in enumerate(dec_layer_names[:-1]):
            
            if self.verbose: print("{} -> {} {}".format(dec_layer_dims[i], lname, dec_layer_dims[i+1]))
            
            layer, layer_name, out_dim = self.create_layer(lname, dec_layer_dims[i+1], dec_layer_dims[i])
            layers.append(layer)
            layer_names.append(layer_name)
            layer_dims.append(out_dim)
        
        return layers, layer_names, layer_dims
        
    def encode(self, x, ret_pool_idxs=False, skip_layers=True):
        
        pool_idxs = []
        self.enc_activations = []
        if skip_layers:
            layer_end = -self.ae_skip_layers
        else:
            layer_end = None
        
        if self.store_activations:
            self.enc_activations.append(x)
        
        for i, l in enumerate(self.encoder[:layer_end]):
            if "Pool" in str(l):
                x, idxs = l(x)
                pool_idxs.append(idxs)
            else:
                x = l(x)
            if self.store_activations:
                self.enc_activations.append(x)
        
        if ret_pool_idxs:
            return x, pool_idxs
        else:
            return x
    
    def decode(self, x, pool_idxs=None, skip_layers=True):
        
        self.dec_activations = []
        if skip_layers:
            layer_end = self.ae_skip_layers
        else:
            layer_end = None
            
        if self.store_activations:
            self.dec_activations.append(x)
            
        for i, l in enumerate(self.decoder):
            
            if layer_end and i < layer_end:
                continue
            
            if "Unpool" in str(l):
                if pool_idxs is None:
                    # TODO: what's the best way?
                    # Use random pool idxs. 
                    _, pool_idxs = self.encode(torch.rand(1, *self.input_dim), ret_pool_idxs=True)
                x = l(x, pool_idxs.pop())
            elif "ConvTranspose" in str(l):
                x = l(x, output_size=(x.shape[0], *self.enc_layer_dims[-i-2])) # TODO: fix hack for output_size
            else:
                x = l(x)
            
            if self.store_activations:
                self.dec_activations.append(x)
                
        return x
    
    def get_encoder_activations(self):
        return self.enc_activations
    
    def get_decoder_activations(self):
        return self.dec_activations
    
#     def get_results_img(self, x, x_prime, nrows=8, padding=5):
        
#         return utils.make_grid(
#             torch.stack((x.long(), x_prime.long()), dim=1).view(-1, *self.input_dim),
#             nrow=nrows, padding=padding).permute(2, 1, 0).cpu().numpy().astype(np.uint8)

    def get_results_img(self, x, x_prime, nrow=8, padding=5):
        
        return utils.make_grid(
            torch.stack((x, x_prime), dim=1).view(-1,*self.input_dim),
            nrow=nrow, padding=padding).permute(1,2,0)
    
    def create_layer(self, layer_name, out_dim, input_dim):

        layer_fn = ConvAE.modules[layer_name]

        if layer_name.startswith("conv"):

            D, H, W = input_dim
            D_out = out_dim[0]
            layer = layer_fn(D, D_out)
            F, S, P = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            H = W = (W - F + 2*P) // S + 1
            D = D_out
            return layer, layer_name, (D, H, W)

        elif layer_name.startswith("de-conv"):

            D, H, W = input_dim
            D_out = out_dim[0]
            layer = layer_fn(D, D_out)
            F, S, P = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            H = W = (W * S) + max((F-S),0) -2 * P
            D = D_out
            return layer, layer_name, (D, H, W)

        elif layer_name.startswith("pool"):

            D, H, W = input_dim
            layer = layer_fn()
            F, S, P = layer.kernel_size, layer.stride, layer.padding
            H = W = (W - F) // S + 1
            return layer, layer_name, (D, H, W)

        elif layer_name.startswith("de-pool"):

            D, H, W = input_dim
            layer = layer_fn()
            F, S, P = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            H = W = (W - F) // S + 1
            return layer, layer_name, (D, H, W)

        elif  layer_name.startswith("linear"):

            N_in = input_dim[0]
            N_out = out_dim[0]
            layer = layer_fn(N_in, N_out)
            return layer, layer_name, (N_out, )

        elif  layer_name.startswith("de-linear"):

            N_in = input_dim[0]
            N_out = out_dim[0]
            layer = layer_fn(N_in, N_out)
            return layer, layer_name, (N_out, )

        elif layer_name.startswith("flatten"):

            D, H, W = input_dim
            layer = layer_fn(-1, D * H * W)
            return layer, layer_name, (D * H * W, )

        elif layer_name.startswith("de-flatten"):

            D_out, H_out, W_out = out_dim
            layer = layer_fn(-1, D_out, H_out, W_out)
            return layer, layer_name, out_dim

        elif layer_name.startswith("relu") or layer_name.startswith("de-relu"):

            return layer_fn(), layer_name, input_dim

        else:
            raise NotImplementedError("Layer {} not supported!".format(layer_name))
            
def normalize_0_1_numpy(x_numpy):
    return (x_numpy - np.min(x_numpy))/np.ptp(x_numpy)

def normalize_0_1(x):
    return (x - x.min())/(x.max() - x.min())

def std_score(img_a, img_b, std, std_max=4.):
    return (std/std_max)/(img_a - img_b).abs().mean() #std_max - (img_a - img_b).abs().mean()/std

def accuracy_1_min_mab(img_a, img_b):
    return 1. -(normalize_0_1(img_a) - normalize_0_1(img_b)).abs().mean()

def pearson_corr_coeff(x, y):
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    PearsonCorrCoeff = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return PearsonCorrCoeff

def create_network(block, times, pooling_freq=3, strided_conv_freq=3, strided_conv_channels=16):
    
    m = []
    block_len = len(block)
    
    for i in range(1,times+1):
        m.extend(block)
        if i % pooling_freq == 0:
            m.extend([("pool1", None)])
        if i % strided_conv_freq == 0:
            m.extend([("conv-strided1", strided_conv_channels)])
    return m
