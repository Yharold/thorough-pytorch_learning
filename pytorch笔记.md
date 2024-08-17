# pytorch笔记

## 一、张量操作

1. 张量创建:举例如下：
```python
x = torch.rand(4,3)
x = torch.zeros(4,3,dtype=torch.long)
x = torch.tensor([5,4,3])
x = torch.randn_like(x,dtype=torch.float)
```
常见方法：
|函数|功能|
|------------|----------------|
|Tensor(size)|基础构造函数|
|tensor(data)|将data转为tensor|
|ones(size)|全1|
|zeors(size)|全0|
|eye(size)|对角1|
|arange(s,e,step)|从s到e，步长为step|
|linspace(s,e,step)|从s到e，均匀分为step份|
|rand/randn(size)|均匀分布和正态分布|
|normal(mean,std)|正态分布|
|randperm(m)|随机排列|

2. 张量操作

|函数|功能|
|------------|----------------|
|x[+/-/*/]y|直接相加/减/乘|
|x[:,1]|索引操作|
|x.view()/x.reshape()|变换操作，但view只是改变了观察角度，内存还是同一个，reshape真的改变了内存|
|x.item()|将单个张量转为标量|
|x.clone()|克隆|

## 二、自动求导

`autograd`包提供了自动求导机制。每个张量都会有一个`requires_grad`属性和`grad`，`requires_grad`为`True`时，张量就会将梯度累加到`grad`中。同样可以使用`requires_grad_(True/False)`方法将`requires_grad`设置为`True`或者`False`。

通过`backward()`方法，可以计算梯度。但需要调用这个方法的张量是一个标量，如果不是，则需要传入一个与这个张量相同形状的张量。例如

```python
x = torch.tensor([5],requires_grad=True)
x = x**2
x.backward() # 此时可以直接计算

y = torch.tensor([5,4],requires_grad=True)
y = y**2
y.backward(torch.ones_like(y))# 需要传入一个和y同样形状的张量，但传入的值是什么取决于具体问题
```

`detach()`方法可以将一个张量与其计算历史分离，这样求导时不会计算这一步的运算。

`with torch.no_grad():`可以将模型中的所有具有`requireds_grad`的参数设置为`False`。在推理时十分有用。

`grad_fn`保存了创建这个张量时的`Function`，即计算方法。`Tensor`和`Function`共同构成了一个无环图`acyclic graph`。这个无环图记录了完整的计算历史，利用这个无环图就能求取一个张量的梯度。所以`grad_fn`同样是求取梯度方面十分重要的一个属性。

## 三、并行计算

并行计算分为单卡训练和多卡训练。单卡就不用说了，直接将模型和数据使用`.cuda()`方法或者设置`device="cuda"`属性转移到GPU上就可以了。这里主要讲多卡。

多卡训练分为两种：

1. DataParallel:简称DP，适用于单机多卡
2. DistributedDataParallel:简称DDP，适用于多机多卡，也可以用到单机单卡

### DP

DP的使用十分简单，只需要执行一行代码即可
```python
model=Net()
model.cuda()
if torch.cuda.device_count() >1:
    model = nn.DataParallel(model)
```
这种方式不指定使用的GPU，也就是由程序自己分配。我们也可以自己指定使用哪些GPU来执行训练
```python
model = nn.DataParallel(model,device_ids=[0,1])
```
这是一种，即指定使用的GPU的编号。第二种就是在环境变量设置GPU的编号。这样未设置的编号的GPU对于程序是不可见的，相当于另类的手动分配。
```python
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
```

DP有一个很大的缺点就是很容易第一块GPU负载大，其他的GPU负载小，即负载不均衡。这是因为默认的输出都被输出到第一块GPU了。

### DDP

DDP虽然是针对于多机多卡，但单机多卡也可以使用。DDP会针对每个GPU启动一个进程，这些进程在一开始会保持一致(模型初始化参数也一致，每个进程有自己的优化器),同时在更新模型的时候,梯度传播也是一致的,这样就能保证每个GPU上的模型参数是完全一致的,这样就不会出现DP的负载不均衡问题了.

但DDP的使用比较麻烦,这里先介绍几个概念

- GROUP:进程组,默认情况下只有一个组.一个job就是一个组,也称作一个world.
- WORLD_SIZE:全局进程个数.如果是多机多卡就是机器数量,如果是单机多卡就是GPU数量
- RANK:进程序号,用于进程间通信,表示进程优先级.比如rank=0的主机是master节点.多机多卡时rank表示第几台机器,单机多卡就是第几块GPU.
- LOCAL_RANK:进程内GPU的编号.例如rank=3,local_rank=0表示第三个进程内的第一块GPU。

DDP的使用方法

1. 添加--local_rank参数：
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank",type=int)
args = parser.parse_args()

# 这个可以自动获取，不用添加这个代码
local_rank = int(os.environ["LOCAL_RANK"])
# 这个是必须的，没有的话默认只使用0号GPU
# 这个是在所有和GPU代码之前添加
torch.cuda.set_device(args.local_rank)
```
2. 初始化后端`backend`:

常用的后端有`gloo`,`nccl`,`mpi`.一般情况下cpu的分布式使用`gloo`,GPU的分布式使用`nccl`.

选好后端之后需要设置网络接口.
```python
import os
os.environ["GLOO_SOCKET_IFNAME"]="eth0" #这两个只写一个
os.environ["NCCL_SOCKET_IFNAME"]="eth0" #eth0是接口，可能改变，具体情况看自己主机，使用ifconfig可以查看到接口名称

#检查nccl是否可用
torch.distributed.is_nccl_available()
#选择nccl后端
torch.distributed.init_process_group(backend='nccl')
```
3. 划分数据：将每个batch划分为几个partition
```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,sampler=train_sampler)
```
4. 包装模型
```python
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
```
5. 启动DDP
DDP的启动需要专门的启动器,对于单机多卡只需要直接运行下面的命令
```linux
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

DDP在每次迭代中有自己的`optimizer`,并独立完成所有优化步骤，进程内与一般的训练无异。各进程计算完成后，只需要将**梯度**进行**汇总平均**，再由rank=0的进程将其广播到所有进程,之后,各进程用该梯度来独立的更新参数.

DP是将**梯度**汇总到主GPU,反向传播更新参数,再广播参数到其他GPU.

## 四、模型

### 定义模型

一般创建新的模型都会继承`nn.module`这个基类,然后只需要构建`__init__`和`forward`这两个方法即可.

基于`nn.module`,可以使用`Sequential`,`ModuleList`,`ModuleDict`三种方式定义模型

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.ReLU(),
            nn.Linear(out_channel,out_channel)
        )
        # add_module方法还需要传入一个字符串作为key
        self.layer1.add_module("name",nn.Linear(in_channel,out_channel))

        self.layer2 = nn.ModuleList(
            [nn.Linear(784, 256), nn.ReLU()]
        )

        self.layer3 = nn.ModuleDict({
                'linear': nn.Linear(784, 256),
                'act': nn.ReLU(),
        })
    def forward(self,x):
        # 这三种方法的取值都是用for循环
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        return x
```

### 修改模型

查看模型结构和参数有多重方法，比较简单的直接打印模型即可，也可以调用`model.parameters()`,`model.modules()`等.
```python
from torchvision import models
net = models.resnet50()
print(net) # 查看模型结构
```
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
..............
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

由上面可以看出，模型的参数其实也是一个字典结构，我们只需要修改key对应的value就能实现模型的修改。

```python
classifier = nn.Sequential(OrderedDict([
    ("fc1",nn.Linear(2048,128)),
    ("relu1",nn.ReLU()),
    ("fc2",nn.Linear(128,10)),
    ("output",nn.Softmax(dim=1))
]))
net.fc = classifier
```
这操作就是把原本的fc层换成了我们定义的classifier结构。这就完成了模型的修改

### 模型保存与读取

保存模型可以分为保存整个模型和只保存权重

```python
from torchvision import models
model = models.resnet152(pretrained=True)
save_dir = './resnet152.pth'

# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重
torch.save(model.state_dict, save_dir)
```
单卡和多卡存储的模型有一些区别,多卡模型名称的前面多了一个"module"

```python
layer:conv1.weight # 单卡
layer:module.conv1.weight # 多卡
```
下面是每种情况的分类

```python
import os,torch
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model = models.resnet152(pretrained=True)
model.cuda()
save_dir = "resnet152.pt"

# 单卡保存和加载
# 全模型
torch.save(model,save_dir) 
loaded_model = torch.load(save_dir).cuda()
# 仅权重
torch.save(model.state_dict(),save_dir)
loaded_model = models.resnet152()# 需要提前定义模型
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model.cuda()

# 单卡保存多卡加载
# 全模型,和常规的先加载模型，后设置多卡一样
torch.save(model,save_dir)
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
loaded_model = torch.load(save_dir)
loaded_model = nn.DataParallel(loaded_model).cuda()
# 仅权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #
loaded_model = models.resnet152() 
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()

# 多卡保存，单卡加载，重点在于去掉模型中的"module"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
model = nn.DataParallel(model).cuda()
torch.save(model,save_dir) # 保存
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 加载
loaded_model = torch.load(save_dir).module

torch.save(model.module.state_dict(),save_dir)# 多卡保存
loaded_model.load_state_dict(torch.load(save_dir)) # 此时模型参数名称中没有module，可以直接加载

# 当权重名称中有module时，也很简单
loaded_model.load_state_dict({k.replace("module.",""):v for k,v in torch.load(save_dir).items()})

# 多卡保存，多卡加载,正常保存和加载即可
torch.save(model.state_dict(),save_dir)
loaded_model = models.resnet152()
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()
```

除了权重，也可以保存诸如**epoch**,**loss**,**optimizer**,**lr**等数据

```python
torch.save({
    "model":model.state_dict(),
    "optimizer":optimizer.state_dict(),
    "lr_scheduler":lr_scheduler.state_dict(),
    "epoch":epoch,
    "args":args,
},save_dir)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
epoch = checkpoint["epoch"]
args = checkpoint["args"]
```

## 五、学习率与数据增强

### 动态调整学习率

pytorch中封装好了一些方法供我们使用：

- lr_scheduler.LambdaLR
- lr_scheduler.MultiplicativeLR
- lr_scheduler.StepLR
- lr_scheduler.MultiStepLR
- lr_scheduler.ExponentialLR
- lr_scheduler.CosineAnnealingLR
- lr_scheduler.ReduceLROnPlateau
- lr_scheduler.CyclicLR
- lr_scheduler.OneCycleLR
- lr_scheduler.CosineAnnealingWarmRestarts
- lr_scheduler.ConstantLR
- lr_scheduler.LinearLR
- lr_scheduler.PolynomialLR
- lr_scheduler.ChainedScheduler
- lr_scheduler.SequentialLR

这些类都继承自`_LRScheduler`,可以通过`help(torch.optim.lr_scheduler)`查看这些类的使用方法，也可以通过`help(torch.optim.lr_scheduler._LRScheduler)`查看`_LRScheduler`的使用方法。

```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率

    # scheduler的优化是在每一轮后面进行的
    scheduler1.step() 
    ...
    schedulern.step()
```

想要自定义可以动态调整的学习率可以通过自定义函数来实现

```python
def adjust_learning_rate(optimizer,epoch):
    lr = args.lr * (0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

optimizer= torch.optim.SGD(model.parameter(),lr = args.lr,momentum=0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
```

### 数据增强

深度学习中经常会遇见过拟合问题，一般会使用一些正则化方法或者减小权重数量来解决，但最简单的办法就是增加数据。数据增强意思就是提高训练数据的大小和质量，以便得到更好的训练结果。

我们这里使用`imgaug`库来实现数据增强.`imgaug`提供了一些对于图像增强的方法。

安装:`pip install imgaug`

操作方法如下：

```python

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
%matplotlib inline
img = imageio.imread("./lenna.jpg")# 读取图片
ia.imshow(img)# 展示图片

ia.seed(4)# 设置随机种子
rotate = iaa.Affine(rotate=(-4,45)) 
img_aug = rotate(image=img) # 旋转图片
ia.imshow(img_aug)

# 对一张图片进行多个操作,类似于torchvision.transforms.Compose()
# 设置
iaa.Sequential(children=None,
                random_order=False,
                name=None,
                deterministic=False,
                random_state=None,
                )
# 创建序列
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25,25)),
    iaa.AdditiveGaussianNoise(scale=(10,60)),
    iaa.Crop(percent=(0,0.2))

])
# 处理
image_aug = aug_seq(image=img)
ia.imshow(image_aug)

# 对多张图片进行处理，只需要将图片放进一个list即可
images = [img1,img2,img3,...]
images_aug = rotate(images=images)
ia.show(np.hstack(images_aug))

images_aug = aug_seq.augment_images(images=images)
```

imgaug有一个特性，即可以对一个batch中的一部分图片应用Augmenters，其他的应用另外的Augmenters.

```python
# 设置
iaa.Sometimes(p=0.5,
then_list=None, #以概率p对图片进行变换 
else_list=None, #以概率1-p对图片进行变换
name=None,
deterministic=False,
random_state=None)
```

## 六、argparse

argparse是python的一个标准库，不需要安装。这个库可以让我们直接在命令行中向程序传入参数。使用argparse一般分为三个步骤

1. 创建ArgumentParser()对象
2. 调用add_argument()方法添加参数
3. 使用parse_args()解析参数

```python
import argparse
# 创建parser
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('-o','--output',action='store_true',help="show output")
# action='store_true'会讲output参数记录为True
# type规定了参数的格式
# default规定了默认值
parser.add_argument('--lr',type=float,default=3e-5,help='select the learning rate,default=1e-3')
parser.add_argument('--batch_size',type=int,required=True,help='input batch size')

args = parser.parse_args()

if args.output:
    print("This is some output")
```
在命令行使用`python demo.py --lr 3e-4 --batch_size 32`就能将参数传入进去

## 七、可视化

### 模型结构可视化

使用torchinfo可以显示模型更详细的信息，使用方法很简单

```python
import torchvision.models as models
from torchinfo import summary
resnet18  = models.resnet18()
summary(resnet18,(1,3,224,224)) #（1,3,224,224）是输入数据形状
```
输出如下：
```
=========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
=========================================================================================
ResNet                                   --                        --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 64, 56, 56]           --
│    └─BasicBlock: 2-1                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
│    └─BasicBlock: 2-2                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-9                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-10                 [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 56, 56]           128
│    │    └─ReLU: 3-12                   [1, 64, 56, 56]           --
├─Sequential: 1-6                        [1, 128, 28, 28]          --
│    └─BasicBlock: 2-3                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-13                 [1, 128, 28, 28]          73,728
│    │    └─BatchNorm2d: 3-14            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-15                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-16                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-17            [1, 128, 28, 28]          256
│    │    └─Sequential: 3-18             [1, 128, 28, 28]          8,448
│    │    └─ReLU: 3-19                   [1, 128, 28, 28]          --
│    └─BasicBlock: 2-4                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-20                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-21            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-22                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-23                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-24            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-25                   [1, 128, 28, 28]          --
├─Sequential: 1-7                        [1, 256, 14, 14]          --
│    └─BasicBlock: 2-5                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-26                 [1, 256, 14, 14]          294,912
│    │    └─BatchNorm2d: 3-27            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-28                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-29                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-30            [1, 256, 14, 14]          512
│    │    └─Sequential: 3-31             [1, 256, 14, 14]          33,280
│    │    └─ReLU: 3-32                   [1, 256, 14, 14]          --
│    └─BasicBlock: 2-6                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-33                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-34            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-35                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-36                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-37            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-38                   [1, 256, 14, 14]          --
├─Sequential: 1-8                        [1, 512, 7, 7]            --
│    └─BasicBlock: 2-7                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-39                 [1, 512, 7, 7]            1,179,648
│    │    └─BatchNorm2d: 3-40            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-42                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-43            [1, 512, 7, 7]            1,024
│    │    └─Sequential: 3-44             [1, 512, 7, 7]            132,096
│    │    └─ReLU: 3-45                   [1, 512, 7, 7]            --
│    └─BasicBlock: 2-8                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-46                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-47            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-49                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-50            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 1000]                 513,000
=========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
Total mult-adds (G): 1.81
=========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 39.75
Params size (MB): 46.76
Estimated Total Size (MB): 87.11
=========================================================================================
```

### CNN可视化

CNN的可视化分为卷积核可视化与特征可视化，卷积核可视化就是将卷积核画出来即可，下面是代码

```python
import torch
from torchvision.models import vgg11
model = vgg11(pretrained=True)
# 提取第三层卷积
conv1 = dict(model.features.named_children())['3']
# 得到卷积核
kernel_et = conv1.weight.detach()
num = len(conv1.weight.detach())
# 绘制图像
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20,17))
    if (len(i_kernel))>1:
        # 绘制第i个卷积核
        for idx,filer in enumerate(i_kernel):
            plt.subplt(9,9,idx+1)
            plt.axis("off")
            plt.imshow(filer[:,:].detach(),cmap='bwr')
```
特征可视化可以使用pytorch提供的接口，hook。我们将hook类注入到模型的某一层中，当模型向前计算的时候会调用hook类的__call__函数。

```python
def plot_feature(model,idx,inputs):
    hh = Hook()
    model.features[idx].register_forward_hook(hh)
    model.eval()
    _ = model(inputs)
    print(hh.modul_name)
    print(hh.features_in_hook[0][0].shape)
    print(hh.features_out_hook[0].shape)
    out1 = hh.features_out_hook[0]
    total_ft  = out1.shape[1]
    first_item = out1[0].cpu().clone()    
    plt.figure(figsize=(20, 17))
    
    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx+1) 
        
        plt.axis('off')
        #plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[ :, :].detach())
```

class activation map(CAM)是用来判断哪些变量对模型来说是重要的。相比卷积核和特征图，CAM更直观，能够快速确定重要区域。CAM可以通过pytorch-grad-cam来实现

```python
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

model = vgg11(pretrained=True)
img_path = './dog.png'
img = Image.open(img_path).resize((224,224))
rgb_img = np.float32(img)/255
# 转为张量
img_tensor = torch.from_numpy(rgb_img).permute(2,0,1).unsqueeze(0)
# 选择目标层
target_layers = [model.features[-1]]
# 选择激活图
cam = GradCAM(model=model,target_layers=target_layers)
# 设置preds，preds是模型的预测输出
targets = [ClassifierOutputTarget(preds)]
# 运行cam，得到激活图
grayscale_cam = cam(input_tensor=img_tensor,target=targets)
grayscal_cam = grayscale_cam[0,:]
# 显示图像
cam_img = show_cam_on_image(rgb_img,grayscale_cam,use_rgb=True)
print(type(cam_img))

```

除了上面的可视化工具外，FlashTorch也能快速帮我们实现CNN的可视化

```python
#可视化梯度
import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transfoms,load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('great_grey_owl.jpg')
owl = apply_transforms(image)
target_class = 24
backprop.visualize(owl,target_class,guided=True,use_gpu=True)

## 可视化卷积核
from flashtorch.activmax import GradientAscent
model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)
conv5_1=model.features[24]
conv5_1_filters = [45,271,263,489]
g_ascent.visualize(conv5_1,conv5_1_filters,title="VGG16:conv5_1")

```

### TensorBoard可视化训练过程

TensorBoard可以记录指定的数据，包括feature map，权重，loss等。记录的内容会保存在我们指定的一个文件夹内，可以通过网页的形式可视化。


1. 指定文件夹
```python
#使用tensorboard 
from tensorbardX import SummaryWriter
#使用torch的tensorboard
from torch.utils.tensorboard import SummaryWriter
#指定文件夹
writer = SummaryWriter('./runs')
```
2. 启动TensorBoard:命令行输入下面命令即可
```
tensorboard --logdir='./runs' --port=xxxx
```

下面是tensorboard的使用

1. 查看模型结构:使用`add_graph()`
```python
model = Net()
writer = SummaryWriter('./runs')
writer.add_graph((model,input_to_model=torch.randn(1,3,224,224)))
writer.close()
```
2. 查看图片:使用`add_image()/add_images()`
```python
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
images,labels = next(iter(train_loader))
# 查看一张图片
writer = SummaryWriter('./runs')
writer.add_image('images[0]',images[0])
writer.close()
# 将图片拼接查看多张图片
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid',img_grid)
writer.close()

# 将多张图片直接写入
writer.add_images("images",images,global_step=0)
writer.close()
```
3. 连续变量可视化
```python
writer = SummaryWriter('./runs')
for i in range(500):
    x = i
    y = x**2
    # i是指将x/y记录在第i步
    writer.add_scalar("x",x,i)
    writer.add_scalar("y",y,i)
writer.close()
# 想在同一张图显示多个曲线，需要建立子路径，并修改标签一致
writer1 = SummaryWriter('./pytorch_tb/x')
writer2 = SummaryWriter('./pytorch_tb/y')
for i in range(500):
    x = i
    y = x*2
    writer1.add_scaler("same",x,i)
    writer2.add_scaler("same",y,i)
writer1.close()
writer2.close()
```
4. 参数分布可视化:使用`add_histogram()`
```python
import torch
import numpy as np

def norm(mean,std):
    t = std*torch.randn((100,20))+mean
    return t

writer = SummaryWriter('./pytorch_tb/')
for step,mean in enumerate(range(-10,10,1)):
    w = norm(mean,1)
    writer.add_histogram("w",w,step)
    writer.flush()
writer.close()
```
5. 远程使用TensorBoard

训练的数据一般都是在远程的电脑上完成的，但这些电脑大多是Linux系统，只有命令行，无法打开浏览器查看结果。所以，我们可以使用SSH工具，在本地打开远程的TensorBoard查看训练情况。使用方法如下：
- MobaXterm
	1. 在MobaXterm点击Tunneling
	2. 选择New SSH tunnel，我们会出现以下界面。

    ![ssh_tunnel](./figures/ssh_tunnel_UI.png)
	3. 对新建的SSH通道做以下设置，第一栏我们选择`Local port forwarding`，`< Remote Server>`我们填写**localhost**，`< Remote port>`填写6006，tensorboard默认会在6006端口进行显示，我们也可以根据 **tensorboard --logdir=/path/to/logs/ --port=xxxx**的命令中的port进行修改，`< SSH server>` 填写我们连接服务器的ip地址，`<SSH login>`填写我们连接的服务器的用户名，`<SSH port>`填写端口号（通常为22），`< forwarded port>`填写的是本地的一个端口号，以便我们后面可以对其进行访问。
	4. 设定好之后，点击Save，然后Start。在启动tensorboard，这样我们就可以在本地的浏览器输入`http://localhost:6006/`对其进行访问了
- Xshell 
	1. Xshell的连接方法与MobaXterm的连接方式本质上是一样的，具体操作如下：
	2. 连接上服务器后，打开当前会话属性，会出现下图，我们选择隧道，点击添加
	![xhell_ui](./figures/xshell_ui.png)
	3. 按照下方图进行选择，其中目标主机代表的是服务器，源主机代表的是本地，端口的选择根据实际情况而定。
	![xhell_set](./figures/xshell_set.png)
	4. 启动tensorboard，在本地127.0.0.1:6006 或者 localhost:6006进行访问。
- SSH
	1. 该方法是将服务器的6006端口重定向到自己机器上来，我们可以在本地的终端里输入以下代码：其中16006代表映射到本地的端口，6006代表的是服务器上的端口。
	```shell
      ssh -L 16006:127.0.0.1:6006 username@remote_server_ip
	```
	2. 在服务上使用默认的6006端口正常启动tensorboard
	```shell
	tensorboard --logdir=xxx --port=6006
	```
	3. 在本地的浏览器输入地址
	```shell
	127.0.0.1:16006 或者 localhost:16006
	```

### wandb可视化训练过程

TensorBoard对数据的保存仅限于本地，也很难分析超参数对不同试验的影响。wandb可以解决这些问题。wandb是Weights and Biases的缩写，它可以自动记录模型训练过程中的超参数和输出指标，然后可视化和比较结果。

安装很简单：`pip install wandb`

登录wandb需要一个keys，可以去官网注册，然后使用`wandb login`登录，然后粘贴自己的keys就可以了。

下面是一个代码演示：
```python
import wandb

warnings.filterwarnings('ignore')
# 初始化，project是项目名，name是实验名
wandb.init(project='thorough-pythorch',name='wandb_demo')
# 设置超参数
config = wandb.config
config.batch_size = 64
config.epochs = 5
...

def train(model,device,train_loader,optimizer):
    model.train()
    for batch_id,(data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

def test(model,device,test_loader,classes):
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output,target).item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).sum().item())
            example_images.append(wandb.Image(data[0],caption="Pred:{}Truth:{}".format(classes[pred[0].item()],classes[target[0]])))

        wandb.log({
            "examples":example_images,
            "test accuracy":100.*correct /len(test_loader.dataset),
            "test loss":test_loss
        })
wandb.watch_called = False

def main():
    ...
    model = resnet18()
    wandb.watch(model,log="all")

main()
```
总结一下就是：
1. 初始化wandb:`wandb.init(project="",name="")`
2. 设置配置文件:`wandb.config.batch_siz/lr/..=...`
3. 记录数据/模型/权重:`wandb.log()`，`wandb.watch(model,log='all')`,`wandb.save('model.pth')`

## 八、ONNX使用方法

当我们想要将模型部署到手机，开发板之上的时候，我们需要一个专门的推理框架。pytorch大多在训练或者优化的时候使用，而推理的时候更过使用onnx。

### 安装

我们需要安装两个包`onnx`,`onnxruntime`。
```python
pip install onnx
pip install onnxruntime
```
我们还要注意onnx和onnxruntime之间的适配关系，可以通过这个地址进行查看：https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md。

如果我们想使用GPU推理，我们需要安装onnxruntim-gpu，同样适配关系可以在这个链接查看：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

### 导出
我们训练好的模型需要转为onnx的格式才能使用onnxruntime
进行推理，导出onnx有很多方法，例如下面代码：
```python
import torch.onnx
onnx_file_name = "model.onnx"
model = torch_model
model = model.load_state_dict(torch.load("model.pth"))
model.eval()# 这个必须设置
batch_size = 1
dummy_input = torch.randn(batch_size,1,224,224,requires_grad=True)
# 运行一下模型
output = model(dummy_input)
# 导出模型
torch.onnx.export(model, #模型
    dummy_input,    #一组实例化的输入
    onnx_file_name, # 保存路径
    export_params=True, #True参数也会导出，False不导出参数，即是一个未训练过的模型
    opset_version=10, # onnx版本
    do_constant_folding=True, # 是否执行折叠优化   
    input_names = ["input"], #输入模型的张量名
    output_names = ["output"], #输出模型的张量名
    # dynamic_axes将batch_size的维度指定为动态
    dynamic_axes={"input":{0:"batch_size"}, 
                  "output":{0:"batch_size"}})

```

### 检验

导出模型后，我们会得到一个onnx文件，我们需要检验模型文件是否可以使用。
```python
import onnx
try:
    onnx.checker.check_model(self.onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid:%s"%e)
else:
    print("This model is valid")
```

### 可视化

如果我们想查看onnx模型的节点，可以使用Netron。下载链接是https://github.com/lutzroeder/netron

### 推理

当使用onnx模型进行推理时，基本的用法如下：
```python
import onnxruntime
onnx_file_name = "xxx.onnx"
# 创建onnx会话
ort_session = onnxruntime.InferenceSession(onnx_file_name)
# 输入
ort_inputs = {"input":input_img}
# 这种形式更好一些,避免了手动输入key
ort_inputs = {ort_seeion.get_inputs()[0].name:input_img}
# 运行会话,结果是一个列表，所以需要索引
ort_output = ort_session.run(None,ort_inputs)[0]
# 这种运行方式更好，明确指出了输出是谁
output = {ort_session.get_output()[0],name}
ort_output = ort_session.run([output],ort_inputs)[0]
```
上面的代码里需要注意的是，onnx的输入是一个array，而不是tensor，所以我们一定要确定我们的数据格式。转化tensor为array的方法有很多，比如下面代码：
```python
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
```

