import torch
import jittor as jt


models = [
          # '/data/models/ViT-B-16.pt',
          # '/data/models/ViT-B-32.pt',
          # '/data/models/ViT-L-14.pt',
          # '/data/models/ViT-L-14-336px.pt'
          '/data/models/convnext_tiny_22k_1k_384.pth',
          '/data/models/convnext_small_22k_1k_384.pth',
          '/data/models/convnext_base_22k_1k_384.pth',
          '/data/models/convnext_large_22k_1k_384.pth'
          ]


for model in models:
    print(model)
    #clip = torch.load(model)['model'].state_dict()
    ckpt = jt.load(model)['model']
    for k in ckpt.keys():
        ckpt[k] = ckpt[k].float().cpu()
    jt.save(ckpt, model[:-3]+'pkl')