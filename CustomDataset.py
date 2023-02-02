import os
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
#from utils import image_processing
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, annotations_file, image_dir, repeat=1, transform=None, target_transform=None): #repeat can delete
        '''
        :param annotations_file: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        :param transform: manipulation of input
        '''
        self.img_labels = self.read_file(annotations_file)
        self.len = len(self.img_labels)
        self.image_dir = image_dir
        self.repeat = repeat
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        index = idx % self.len
        # print("i={},index={}".format(i, index))
        image_name, label = self.img_labels[index]
        img_path = image_name
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(label)
        # label=np.array(label)
        # print(label)
        return image, label
 
    def __len__(self):
        # if self.repeat == None:
        #     data_len = 10000000
        # else:
        #     data_len = len(self.image_label_list) * self.repeat
        # return data_len

        return len(self.img_labels)

    def read_file(self, filename):
        img_labels = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                img_labels.append((name, labels))
        return img_labels

 
 

if __name__=='__main__':
    
    train_filename="/home/ywang/train_balanced_label.txt"
    val_filename="/home/ywang/validation_label.txt"
    #test_filename="/home/ywang/test_label.txt"
    image_dir='/home/ywang/XceptionExtract'
 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
 
    epoch_num=2   #总样本循环次数
    batch_size=4  #训练时的一组数据的大小
    #train_data_nums=10
    #max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数
 
    train_data = CustomDataset(annotations_file=train_filename, image_dir=image_dir,repeat=1, transform=transform)
    val_data = CustomDataset(annotations_file=val_filename, image_dir=image_dir,repeat=1, transform=transform)
    #test_data = CustomDataset(filename=test_filename, image_dir=image_dir,repeat=1, transform=transform)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)


    # # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {len(train_features)}")
    # print(f"Labels batch shape: {len(train_labels)}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # #plt.imshow(img, cmap="gray")
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.show()
    # print(f"Label: {label}")


    # Another way to Display image and label.
    # Class labels
    classes = ('real', 'fake')

    # Report split sizes
    print('Training set has {} instances'.format(len(train_data)))
    print('Validation set has {} instances'.format(len(val_data)))

    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)

    #print(labels)
    # print(labels[0])
    # print(type(labels[0]))
    # print(labels[0][0])
    # print(type(labels[0][0]))
    # print(labels[0][0].item())
    # print(type(labels[0][0].item()))
    print(labels)
    print('  '.join(classes[labels[0][j].item()] for j in range(batch_size)))
 



    # # [1]使用epoch方法迭代，CustomDataset的参数repeat=1
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         image=batch_image[0,:]
    #         image=image.numpy()#image=np.array(image)
    #         image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #         image_processing.cv_show_image("image",image)
    #         print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
    #         # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    
    # '''
    # 下面两种方式，TorchDataset设置repeat=None可以实现无限循环，退出循环由max_iterate设定
    # '''
    # train_data = TorchDataset(filename=train_filename, image_dir=image_dir,repeat=None)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # # [2]第2种迭代方法
    # for step, (batch_image, batch_label) in enumerate(train_loader):
    #     image=batch_image[0,:]
    #     image=image.numpy()#image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image",image)
    #     print("step:{},batch_image.shape:{},batch_label:{}".format(step,batch_image.shape,batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    #     if step>=max_iterate:
    #         break
    # [3]第3种迭代方法
    # for step in range(max_iterate):
    #     batch_image, batch_label=train_loader.__iter__().__next__()
    #     image=batch_image[0,:]
    #     image=image.numpy()#image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image",image)
    #     print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)