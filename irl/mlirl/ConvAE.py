# Torch
import torch
from torch import nn

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
               "de-conv1": lambda D_in, D_out: nn.ConvTranspose2d(D_in, D_out, kernel_size=3, stride=1, padding=1, bias=False),
               "de-relu1": nn.ReLU,
               "de-pool1": lambda: nn.MaxUnpool2d(2, 2),
               "de-linear1": lambda N_in, N_out: nn.Linear(N_in, N_out, bias=True),
               "de-flatten1": Reshape,
              }
    
    def __init__(self, input_dim, enc_config = [
                                        ("conv1", 16), ("pool1", None), 
                                        ("flatten1", None), ("linear1", 4096)],
                 verbose=False):
        
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.verbose = verbose
        
        # Encoder
        self.enc_layers, self.enc_layer_names, self.enc_layer_dims = self.prepare_encoder(input_dim, enc_config)
        self.encoder = nn.ModuleList(self.enc_layers[1:])
        self.code_size = self.enc_layer_dims[-1]
        
        # Decoder
        self.dec_layers, self.dec_layer_names, self.dec_layer_dims = self.prepar_decoder(
            self.enc_layer_names, self.enc_layer_dims)
        self.decoder = nn.ModuleList(self.dec_layers)
        self.tanh = nn.Tanh()
        
    def forward(self, images):
        code, pool_idxs = self.encode(images)
        out = self.tanh(self.decode(code, pool_idxs))
        return out, code
    
    def prepare_encoder(self, input_dim, enc_config):
        
        layers = [None]
        layer_names = ["input"]
        layer_dims = [input_dim]
        
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
        
    def encode(self, x):
        
        pool_idxs = []
        for i, l in enumerate(self.encoder):
            if "Pool" in str(l):
                x, idxs = l(x)
                pool_idxs.append(idxs)
            else:
                x = l(x)
        return x, pool_idxs
    
    def decode(self, x, pool_idxs):
        
        for i, l in enumerate(self.decoder):
            if "Unpool" in str(l):
                x = l(x, pool_idxs.pop())
            else:
                x = l(x)
        return x
    
    def get_results_img(self, x, x_prime, nrows=8, padding=5):
        
        return utils.make_grid(
            torch.stack((x.long(), x_prime.long()), dim=1).view(-1, *self.input_dim),
            nrow=nrows, padding=padding).permute(2, 1, 0).cpu().numpy().astype(np.uint8)
    
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
            
def accuracy(n_img_a, n_img_b):
    img_a = unnormalize_image(n_img_a).float()
    img_b = unnormalize_image(n_img_b).float()
    
    img_a /= 255.
    img_b /= 255.
    
    return 1.-(img_a - img_b).abs().mean()

def create_network(block, times, pooling_freq=3):
    
    m = []
    block_len = len(block)
    
    for i in range(1,times+1):
        if i % pooling_freq == 0:
            m.extend(block)
            m.extend([("pool1", None)])
        else:
            m.extend(block)
    return m