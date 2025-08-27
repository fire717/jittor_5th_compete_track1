import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from jittor.models import Resnet50
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
import cv2
import albumentations as A
from transformers import ConvNextConfig, ConvNextModel
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from models.convnext import *
import random  
import copy
import glob
from copy import deepcopy

random.seed(42)
np.random.seed(42)


jt.flags.use_cuda = 1
jt.misc.set_global_seed(42)


cfg = {
    'model_name': 'convnext_large',
    'pretrained': '/data/models/convnext_large_22k_1k_384.pkl',

    'class_num': 6,
    'full_train': True,

    'epochs': 20,
    'batchsize': 4,
}


def exists(val):
    return (val is not None)

def is_float_dtype(dtype):
    return any([(dtype == float_dtype) for float_dtype in (jt.float64, jt.float32, jt.float16)])

def clamp(value, min_value=None, max_value=None):
    assert (exists(min_value) or exists(max_value))
    if exists(min_value):
        value = max(value, min_value)
    if exists(max_value):
        value = min(value, max_value)
    return value

class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model=None,
        beta=0.99,
        update_after_step=0,
        update_every=1,
        inv_gamma=1.0,
        power=(2 / 3),
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
        ignore_names=set()
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = ema_model
        # is_stop_grad property is lost here with copy
        # but it will be corrected later with copy_params_from_model_to_ema()
        if (not exists(self.ema_model)):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()
        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
        self.ignore_names = ignore_names
        self.initted=jt.Var([False]).stop_grad()
        self.step=jt.Var([0]).stop_grad()

    def copy_params_from_model_to_ema(self):
        for (ma_params, current_params) in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            if (not is_float_dtype(current_params.dtype)):
                continue
            if (current_params.is_stop_grad()):
                ma_params.assign(current_params.copy()).stop_grad()
            else:
                ma_params.assign(current_params.copy()).start_grad()

    def get_current_decay(self):
        epoch = clamp(((self.step.item() - self.update_after_step) - 1), min_value=0.0)
        value = (1 - ((1 + (epoch / self.inv_gamma)) ** (- self.power)))
        if (epoch <= 0):
            return 0.0
        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1
        if ((step % self.update_every) != 0):
            return
        if (step <= self.update_after_step):
            self.copy_params_from_model_to_ema()
            return
        if (not self.initted.item()):
            self.copy_params_from_model_to_ema()
            self.initted.assign(jt.Var([True]))
        self.update_moving_average(self.ema_model, self.online_model)

    @jt.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()
        for ((name, current_params), (_, ma_params)) in zip(list(current_model.named_parameters()), list(ma_model.named_parameters())):
            if (name in self.ignore_names):
                continue
            if (not is_float_dtype(current_params.dtype)):
                continue
            if (name in self.param_or_buffer_names_no_ema):
                if (current_params.is_stop_grad()):
                    ma_params.assign(current_params.copy()).stop_grad()
                else:
                    ma_params.assign(current_params.copy()).start_grad()
                continue
            difference = (ma_params - current_params)
            difference.assign(difference * (1.0 - current_decay))
            if (current_params.is_stop_grad()):
                ma_params.assign(ma_params - difference).stop_grad()
            else:
                ma_params.assign(ma_params - difference).start_grad()

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits= m(x)
            
            if output is None:
                output = logits
            else:
                output += logits
                
        output /= len(self.models)
        return output

def blackEdge(img,p=0.5):
    if random.random()>p:
        return img 
    h,w = img.shape[:2]
    top = random.randint(10,80)
    img[:top,:,:] = 0
    img[h-top:,:,:] = 0

    if random.random()<0.5:
        left = random.randint(2,20)
        img[:,:left,:] = 0
        img[:,w-left:,:] = 0
    return img
        

def data_aug(img):
    # opencv img, BGR
    img = np.array(img)



    img = A.OneOf([A.ShiftScaleRotate(
                            shift_limit=0.,
                            scale_limit=0.1,
                            rotate_limit=10,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                             value=0, mask_value=0,
                            p=0.5),
                    # A.GridDistortion(num_steps=5, distort_limit=0.2,
                    #     interpolation=1, border_mode=4, p=0.4),
                    # A.RandomGridShuffle(grid=(3, 3),  p=0.3)
                    ],
                    p=0.5)(image=img)['image']

    # img = A.HorizontalFlip(p=0.5)(image=img)['image'] 
    # img = A.VerticalFlip(p=0.5)(image=img)['image'] 
    
    img = A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.05, 
                                           contrast_limit=0.05, p=0.5), 
                    A.HueSaturationValue(hue_shift_limit=10, 
                        sat_shift_limit=10, val_shift_limit=10,  p=0.5)], 
                    p=0.4)(image=img)['image']


    # img = A.GaussNoise(var_limit=(5.0, 10.0), mean=0, p=0.2)(image=img)['image']


    # img = A.RGBShift(r_shift_limit=5,
    #                     g_shift_limit=5,
    #                     b_shift_limit=5,
    #                     p=0.5)(image=img)['image']

    
    # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
    # img = A.OneOf([A.GaussianBlur(blur_limit=3, p=0.1),
    #                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
    #                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.4)], 
    #                 p=0.4)(image=img)['image']

    # img = A.CoarseDropout(max_holes=3, max_height=20, max_width=20, 
    #                     p=0.5)(image=img)['image']

    # img = blackEdge(img,p=0.5)

    
    img = Image.fromarray(img)
    return img


class MyCrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None, gamma=0):
        super().__init__()
        self.class_weight = class_weight #means alpha
        self.label_smooth = label_smooth
        self.gamma = gamma
        self.epsilon = 1e-7
        
    def execute(self, x, y, mask=None):
        #print(x.shape, y.shape)
        one_hot_label = jt.nn.one_hot(y, x.shape[1])

        if self.label_smooth:
            one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = nn.softmax(x, 1)
        #print(y_softmax)
        y_softmax = jt.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = jt.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        if self.class_weight:
            loss = loss*self.class_weight

        if self.gamma:
            loss = loss*((1-y_softmax)**self.gamma)
        #print(loss.shape)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            #print(loss)
            loss = loss*mask


        loss = jt.mean(jt.sum(loss, -1))
        return loss

# ============== Dataset ==============
cate2_dict = {0:0,  1:0,  2:1,  3:1,  4:1,  5:2}
class ImageFolder(Dataset):
    def __init__(self, root, mode, annotation_path=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.mode = mode
        self.transform = transform

        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
            data_dir = [(os.path.join(root,x[0]), int(x[1]), cate2_dict[int(x[1])]) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(os.path.join(root,x), None, None) for x in data_dir]
        self.data_dir = data_dir

        if mode!='test':
            data_count = [0 for _ in range(6)]
            for data in self.data_dir:
                data_count[data[1]] += 1
            print(f"{mode} count Before ",data_count) #[364, 608, 215, 16, 6, 291]
        
        if mode=='train':
            add_data = []
            for data in data_dir:
                if data[1] == 3:
                    for _ in range(7):
                        add_data.append(data)
                elif data[1] == 4:
                    for _ in range(18):
                        add_data.append(data)

            self.data_dir = self.data_dir+add_data
            

            data_count = [0 for _ in range(6)]
            for data in self.data_dir:
                data_count[data[1]] += 1
            print(f"{mode} count After ",data_count) #[364, 608, 215, 16, 6, 291]

            ###add busi
            add_data = []
            base_dir = "TrainSet/Dataset_BUSI_with_GT_clean"
            for cate in range(6):
                imgs = os.listdir(os.path.join(base_dir,str(cate)))
                for img in imgs:
                    d = [os.path.join(base_dir,str(cate),img), cate, cate2_dict[cate]]
                    add_data.append(d)
          
            print("add busi:",len(add_data))
            self.data_dir = self.data_dir+add_data


            ###add ultrasound_breast_classification
            # add_data = []
            # imgs = glob.glob("TrainSet/ultrasound_breast_classification/*/*/*")
            # print("add ultrasound_breast_classification :",len(imgs))
            # for img in imgs:
            #     if "benign" in img:
            #         d = [img, -1, 0]
            #     elif "malignant" in img:
            #         d = [img, -1, 1]
            #     add_data.append(d)
            # self.data_dir = self.data_dir+add_data

            ###add BUS_UC
            # add_data = []
            # imgs = glob.glob("TrainSet/BUS_UC/Benign/images/*")
            # for img in imgs:
            #     d = [img, -1, 0]
            #     add_data.append(d)
            # imgs = glob.glob("TrainSet/BUS_UC/Malignant/images/*")
            # for img in imgs:
            #     d = [img, -1, 1]
            #     add_data.append(d)
            # self.data_dir = self.data_dir+add_data

            ###BUS_UCLM
            add_data = []
            base_dir = "TrainSet/BUS-UCLM_clean"
            for cate in range(6):
                imgs = os.listdir(os.path.join(base_dir,str(cate)))
                for img in imgs:
                    d = [os.path.join(base_dir,str(cate),img), cate, cate2_dict[cate]]
                    add_data.append(d)
          
            print("add BUS_UCLM:",len(add_data))
            self.data_dir = self.data_dir+add_data
            
            # add_data = []
            # img_dir = "TrainSet/BUS-UCLM/images"
            # with open("TrainSet/BUS-UCLM/INFO.csv",'r') as f:
            #     lines = f.readlines()
            #     for line in lines:
            #         items = line.strip().split(";")
            #         img_path = os.path.join(img_dir,items[0])
            #         label = items[2]
            #         if label == 'Normal':
            #             add_data.append([img_path,5,2])
            #         elif label == 'Benign':
            #             add_data.append([img_path,-1,0])
            #         elif label == 'Malignant':
            #             add_data.append([img_path,-1,1])
            # self.data_dir = self.data_dir+add_data


            random.shuffle(self.data_dir)

        self.total_len = len(self.data_dir)
        print(self.mode, self.total_len)

    def __getitem__(self, idx):
        image_path, label, label2 = self.data_dir[idx][0], self.data_dir[idx][1], self.data_dir[idx][2]

        image = Image.open(image_path).convert('RGB')
        # gray_image = image.convert('L')  # L模式是单通道灰度图
        # image = Image.merge('RGB', (gray_image, gray_image, gray_image))

        if self.mode=='train':
            image = data_aug(image)

        if self.transform:
            image = self.transform(image)
        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        label2 = image_name if label2 is None else label2
        return jt.array(image), label, label2

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(jt.ones(1) * p)  # 使用jittor的ones
        self.eps = eps
        self.pool = jt.nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):  # Jittor使用execute而不是forward
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        # 使用Jittor的操作替换PyTorch操作
        # 注意：Jittor的clamp/min/max操作与PyTorch兼容
        x = jt.clamp(x, min_v=eps)  # 等价于x.clamp(min=eps)
        x = x.pow(p)
        # Jittor的池化函数在nn模块中
        x = self.pool(x)  # 固定输出尺寸为(1,1)
        return x.pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.item()) + \
                ', ' + 'eps=' + str(self.eps) + ')'

# ============== Model ==============
class Net(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()
        # self.base_net = Resnet50(num_classes=num_classes, pretrained=pretrain)
        # Initializing a ConvNext convnext-tiny-224 style configuration
        # configuration = ConvNextConfig()

        # # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        # model = ConvNextModel(configuration)

        # # Accessing the model configuration
        # configuration = model.config
        # print(configuration)

        self.pretrain_model = eval(cfg['model_name'])()#convnext_small()
        # print(self.pretrain_model)

        if cfg['pretrained']:
            ckpt = jt.load(cfg['pretrained'])#['model']
            # print(ckpt.keys())
            # bb
            # del ckpt["_fc.weight"]
            # del ckpt["_fc.bias"]
            self.pretrain_model.load_state_dict(ckpt) 

        # print(self.pretrain_model)
        # bb
        self.backbone =  self.pretrain_model
        # num_features = self.backbone.bn2.num_features
        num_features = 768
        if 'base' in cfg['model_name']:
            num_features = 1024
        elif 'large' in cfg['model_name']:
            num_features = 1536
        self.pool =  GeM(p=3)
        self.head1 = nn.Linear(num_features, cfg['class_num'])
        # self.backbone.classifier = nn.Linear(num_features, cfg['class_num'])
        self.head2 = nn.Sequential(
                        # nn.Dropout(0.2),
                        nn.Linear(cfg['class_num'], 3)
                        )

    def execute(self, x):
        out = self.backbone(x)
        #print(out.shape)
        # out = out.view(out.size(0), -1)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        out1 = self.head1(out)
        out2 = self.head2(out1)
        return out1,out2

myloss = MyCrossEntropyLoss()

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, 
            now_epoch:int, num_epochs:int,scheduler):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))
    step = 0
    for data in pbar:
        step += 1
        image, label, label2 = data
        pred,pred2 = model(image)
        # loss = nn.cross_entropy_loss(pred, label)

        mask = label>=0
        mask = mask.int() 

        loss1 = myloss(pred, label, mask)
        loss2 = myloss(pred2, label2)
        loss = loss1*0.5+loss2*0.5

        loss.sync()
        optimizer.step(loss)
        scheduler.step()
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.2f}')

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model:nn.Module, ema_model, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    #print("Evaluating...")
    cate_count = [0,0,0,0,0,0]
    cate_right = [0,0,0,0,0,0]
    for data in val_loader:
        image, label, label2 = data
        if ema_model is None:
            pred,pred2 = model(image)
        else:
            pred,pred2 = ema_model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        targets.append(label.numpy())
        for i in range(len(pred)):
            cate_count[label.numpy()[i]] += 1  
            if label.numpy()[i]==pred[i]:
                cate_right[label.numpy()[i]] += 1 
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(np.float32(preds == targets))

    cate_acc = []
    for i in range(6):
        a = cate_right[i]/(cate_count[i]+1e-7)
        a = int(a*1000)/1000.
        cate_acc.append(a)
    return acc,cate_acc

def run(model:nn.Module, ema_model, optimizer:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, 
    num_epochs:int, modelroot:str, scheduler):
    best_acc = 0
    best_e = 0
    for epoch in range(num_epochs):
        training(model, optimizer, train_loader, epoch, num_epochs, scheduler)
        if ema_model is not None:
            ema_model.update()
        acc,cate_acc = evaluate(model, ema_model, val_loader)
        if acc > best_acc:
            best_acc = acc
            best_e = epoch
            model.save(os.path.join(modelroot, 'best.pkl'))
            print("cate_acc:",cate_acc)
        print(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.4f} best_e={best_e}, now acc = {acc:.4f}\n')
    if ema_model is None:
        model.save(os.path.join(modelroot, 'last.pkl'))
    else:
        ema_model.ema_model.save(os.path.join(modelroot, 'last.pkl'))
# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    for data in test_loader:
        image, image_names, _ = data
        image_names = [os.path.basename(x) for x in image_names]
        pred,_ = model(image)
        pred.sync()

        # ###tta
        # flipped = jt.flip(image, dim=3)
        # #print(flipped.shape)
        # pred2,_ = model(flipped)
        # pred2.sync()

        # pred = pred+pred2

        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')

# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./TrainSet')
    parser.add_argument('--modelroot', type=str, default='./output')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='./output/last.pkl')
    parser.add_argument('--result_path', type=str, default='./output/result.txt')
    parser.add_argument('--ema', type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.modelroot,exist_ok=True)

    model = Net(pretrain=True, num_classes=6)
    ema_model = None
    if args.ema:
        print("Initializing EMA model..")

        ema_model = EMA(model)
        
    
    transform_train = Compose([
        Resize((512, 512)),
        # RandomCrop(448),
        # RandomHorizontalFlip(),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((512, 512)),
        # CenterCrop(448),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not args.testonly:
        optimizer = nn.Adam(model.parameters(), lr=0.0001)
        scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])


        train_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            mode='train',
            annotation_path=os.path.join(args.dataroot, 'labels/trainval.txt' if cfg['full_train'] else 'labels/train.txt'),
            transform=transform_train,
            batch_size=cfg['batchsize'],
            num_workers=8,
            shuffle=True
        )
        val_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            mode='val',
            annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
            transform=transform_val,
            batch_size=cfg['batchsize'],
            num_workers=8,
            shuffle=False
        )
        run(model, ema_model, optimizer, train_loader, val_loader, cfg['epochs'], args.modelroot, scheduler)
    else:
        test_loader = ImageFolder(
            root=args.dataroot,
            mode='test',
            transform=transform_val,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )
        model.load(args.loadfrom)
        test(model, test_loader, args.result_path)
