import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable

class DCGAN_Discriminator(nn.Module):

    def __init__(self, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, 3,
                               stride=2, padding=2, bias=False)
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3,
                               stride=2, padding=0, bias=False)
        self.BN2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3,
                               stride=2, padding=0, bias=False)
        self.BN3 = nn.BatchNorm2d(128)

        self.fc = nn.Conv2d(128, 1, 3, 1, 0, bias=False)


    def forward(self, x):
        """
            补充代码
        """
        x=self.conv1(x)
        x=self.BN1(x)
        x=torch.nn.functional.leaky_relu(x,negative_slope=0.01,inplace=False)
        x=self.conv2(x)
        x=self.BN2(x)
        x=torch.nn.functional.leaky_relu(x,negative_slope=0.01,inplace=False)
        x=self.conv3(x)
        x=self.BN3(x)
        x=torch.nn.functional.leaky_relu(x,negative_slope=0.01,inplace=False)
        x=self.fc(x)
        x=torch.sigmoid(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self, n_channel=1, noise_dim=100):
        super(DCGAN_Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(noise_dim, 128, 3,
                                        stride=1, padding=0, bias=False)

        self.BN1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3,
                                        stride=2, padding=0, bias=False)

        self.BN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3,
                                        stride=2, padding=0, bias=False)
        self.BN3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, n_channel, 3,
                                        stride=2, padding=2, bias=False)

    def forward(self, x):
        """
            补充代码
        """
        x=self.conv1(x)
        x=self.BN1(x)
        x=torch.nn.functional.relu(x, inplace=False)
        x=self.conv2(x)
        x=self.BN2(x)
        x=torch.nn.functional.relu(x, inplace=False)
        x=self.conv3(x)
        x=self.BN3(x)
        x=torch.nn.functional.relu(x, inplace=False)
        x=self.conv4(x)
        x=torch.tanh(x)
        x = x.squeeze(-1)
        return x

def load_dataset(batch_size=10, download=True):
    transform = transforms.Compose([transforms.CenterCrop(27),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                          download=download,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                         download=download,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader

def gen_noise(n_instance, n_dim=2):
    return torch.randn(n_instance, n_dim, 1, 1)
    #return torch.Tensor(np.random.uniform(low=-1.0, high=1.0,
                              #            size=(n_instance, n_dim)))


def train_DCGAN(Dis_model, Gen_model, D_criterion, G_criterion, D_optimizer,
                G_optimizer, trainloader, n_epoch, batch_size, noise_dim,
                n_update_dis=2, n_update_gen=1, use_gpu=False, print_every=10,
                update_max=None):
    for epoch in range(n_epoch):

        D_running_loss = 0.0
        G_running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs from true distribution
            true_inputs, _ = data
            if use_gpu:
                true_inputs = true_inputs.cuda()
            true_inputs = Variable(true_inputs)

            # get the inputs from the generator
            noises = gen_noise(batch_size, n_dim=noise_dim)
            if use_gpu:
                noises = noises.cuda()
            fake_inputs = Gen_model(Variable(noises))
            # 保存训练过程中生成的图片，在自己电脑上实验的同学可以保存训练过程中生成的图片查看
            fake_image_name = str(epoch)+"_"+str(i)+'.png'
            image_path = os.path.join("..\\data\\gen", fake_image_name)
            if i%100 == 0:
                torchvision.utils.save_image(fake_inputs, nrow=10, pad_value=255, fp=image_path)
            inputs = torch.cat([true_inputs, fake_inputs])
            # get the labels
            labels = np.zeros(2 * batch_size)
            labels[:batch_size] = 1
            labels = torch.from_numpy(labels.astype(np.float32))
            if use_gpu:
                labels = labels.cuda()
            labels = Variable(labels)
            if i % n_update_gen == 0:
                # Discriminator
                D_optimizer.zero_grad()
                outputs = Dis_model(inputs)
                D_loss = D_criterion(outputs[:, 0], labels)
                D_loss.backward(retain_graph=True)

                # Generator
                G_optimizer.zero_grad()
                G_loss = G_criterion(outputs[batch_size:, 0],
                                        labels[:batch_size])                     
                G_loss.backward()
                D_optimizer.step()
                G_optimizer.step()
            
            
            # print statistics
            D_running_loss += D_loss.item()
            G_running_loss += G_loss.item()
            if i % print_every == (print_every - 1):
                print('[%d, %5d] D loss: %.3f ; G loss: %.3f' %
                      (epoch+1, i+1, D_running_loss / print_every,
                       G_running_loss / print_every))
                D_running_loss = 0.0
                G_running_loss = 0.0

            if update_max and i > update_max:
                break

    print('Finished Training')


def run_DCGAN(n_epoch=50, batch_size=100, use_gpu=False, dis_lr=4e-4,
              gen_lr=4e-4, n_update_dis=1, n_update_gen=1, noise_dim=100,
              n_channel=1, update_max=None):
    # loading data
    trainloader, testloader = load_dataset(batch_size=batch_size, download=True)

    # initialize models
    Dis_model = DCGAN_Discriminator(n_channel=n_channel)
    Gen_model = DCGAN_Generator(n_channel=n_channel, noise_dim=noise_dim)

    if use_gpu:
        Dis_model = Dis_model.cuda()
        Gen_model = Gen_model.cuda()

    # assign loss function and optimizer (Adam) to D and G
    D_criterion = torch.nn.BCELoss()
    D_optimizer = optim.Adam(Dis_model.parameters(), lr=dis_lr,
                             betas=(0.5, 0.999))

    G_criterion = torch.nn.BCELoss()
    G_optimizer = optim.Adam(Gen_model.parameters(), lr=gen_lr,
                             betas=(0.5, 0.999))
    #torch.autograd.set_detect_anomaly(True)    
    train_DCGAN(Dis_model, Gen_model, D_criterion, G_criterion, D_optimizer,
                G_optimizer, trainloader, n_epoch, batch_size, noise_dim,
                n_update_dis, n_update_gen, update_max=update_max)


if __name__ == '__main__':
    run_DCGAN()