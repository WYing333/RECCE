"""
Author: Andreas Rössler
"""
from torchvision import transforms

xception_default_data_transforms = {
    'train': transforms.Compose([   ##可以看出Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  ##ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
                                ##其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。
        transforms.Normalize([0.5]*3, [0.5]*3) ##在transforms.Compose([transforms.ToTensor()])中加入transforms.Normalize()，
                                               ##如下所示：transforms.Compose([transforms.ToTensor(),transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])，
                                               ##则其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}