import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type = str, default = './img_align_celeba', help = '存放数据的地址')
    parser.add_argument('--workers', type = int, default = 2, help = 'number of dataloader workers')
    parser.add_argument('--batchSize', type = int, default = 64, help = 'batch size')
    parser.add_argument('--imageSize', type = int, default = 64, help = 'the input of the image')
    parser.add_argument('--nz', type = int, default = 100, help = '向量z的长度')
    parser.add_argument('--ngf', type = int, default = 64, help = 'Size of feature maps in generator')
    parser.add_argument('--ndf', type = int, default = 64, help = 'Size of feature maps in discriminator')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs')
    parser.add_argument('--lr', type = float, default = 2e-3, help = 'learning default = 0.002')
    parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for Adam, default = 0.5')
    parser.add_argument('--cuda', action = 'store_true', help = 'enables cuda')
    parser.add_argument('--ngpu', type = int, default = 1, help = 'number of GPUs to use')
    parser.add_argument('--netG', default = '', help = "path to netG (to continue training)")
    parser.add_argument('--netD', default = '', help = "path to netD (to continue training)")
    parser.add_argument('--outf', default = './m_img', help = 'folder to output images')
    parser.add_argument('--ckpt', default = './ckpt', help = 'folder to checkpoints')
    parser.add_argument('--manualSeed', type = int, help = 'manual seed')

    args = parser.parse_args()
    return args