import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf):
        super(Generator, self).__init__()
        self.num_gpu = ngpu
        self.G_Net = nn.Sequential(
            # input z into a conv
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # (ngf * 8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # (ngf * 4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # (ngf * 2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # (nc, 64, 64)
        )

    def forward(self, input):
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.G_Net, input, range(self.num_gpu))
        else:
            output = self.G_Net(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.num_gpu = ngpu
        self.D_Net = nn.Sequential(
            # (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # (ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # (ndf * 2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # (ndf * 4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # (ndf * 8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False)
            # ,nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.D_Net, input, range(self.num_gpu))
        else:
            output = self.D_Net(input)
        return output.view(-1, 1).squeeze(1)

# model = Generator(1, 100, 3, 64)
# x = torch.rand(1, 100, 1, 1)
# print(model(x).shape)

# model = Discriminator(1, 3, 64)
# x = torch.rand(100, 3, 64, 64)
# print(model(x).shape)