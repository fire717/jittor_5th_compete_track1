import jittor as jt
from jittor import nn, Module
from jittor import init

class LayerNorm(Module):
    r""" LayerNorm that supports two data formats: channels_last or channels_first. 
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def execute(self, x):
        if self.data_format == "channels_last":
            u = x.mean(dim=-1, keepdims=True)
            s = (x - u).pow(2).mean(dim=-1, keepdims=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            x = self.weight * x + self.bias
            return x
            
        elif self.data_format == "channels_first":
            u = x.mean(dim=1, keepdims=True)
            s = (x - u).pow(2).mean(dim=1, keepdims=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            shape = [1, -1] + [1] * (x.ndim - 2)  # [1, C, 1, 1, ...]
            weight = self.weight.reshape(shape)
            bias = self.bias.reshape(shape)
            x = weight * x + bias
            return x

class DropPath(Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jt.random(shape)
        random_tensor = jt.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Jittor implementation of truncated normal initialization
    tensor = jt.array(tensor)
    # Generate tensor with normal distribution
    jt.init.gauss_(tensor, mean, std)
    
    # Apply truncation by resampling values outside [a, b]
    while True:
        mask = (tensor < a) | (tensor > b)
        if not mask.any():
            break
        # Resample values outside the desired range
        new_samples = jt.init.gauss_(jt.empty(mask.sum().item()), mean, std)
        tensor[mask] = new_samples
    return tensor

class Block(Module):
    r""" ConvNeXt Block for Jittor """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs with linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale parameter
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = jt.array(layer_scale_init_value * jt.ones(dim))
            self.gamma.stop_grad()  # Mark as parameter
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(Module):
    r""" ConvNeXt for Jittor """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and downsampling convs
        stem = nn.Sequential(
            nn.Conv(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # feature stages
        dp_rates = [x for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # if hasattr(self, 'head'):
        #     self.head.weight = self.head.weight * head_init_scale
        #     self.head.bias = self.head.bias * head_init_scale

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def execute_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Global average pooling: (N, C, H, W) -> (N, C)
        # x = jt.mean(x, dim=[-2, -1])
        # x = jt.mean(x, dim=-1)
        # x = jt.mean(x, dim=-1)
        return x#self.norm(x)

    def execute(self, x):
        x = self.execute_features(x)
        # x = self.head(x)
        return x

# Model variants
def convnext_tiny(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_small(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_base(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

def convnext_large(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

def convnext_xlarge(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)