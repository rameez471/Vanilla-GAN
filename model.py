import torch
from torch import nn

class Generator(nn.Module):

    def __init__(self,input_dim=10,im_ch=1,hidden_dim=64):
        super(Generator,self).__init__()
        self.input_dim = input_dim,
        self.gen = nn.Sequential(
            self.make_gen_blocks(input_dim, hidden_dim*4),
            self.make_gen_blocks(hidden_dim*4, hidden_dim*2, kernel_size=4, strides=1),
            self.make_gen_blocks(hidden_dim * 2, hidden_dim),
            self.make_gen_blocks(hidden_dim, im_ch, kernel_size=4, strides=1, final_layer=True)
        )

    def make_gen_blocks(self,input_channels, output_channels, kernel_size=3, strides=2, final_layer=False):
    
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, strides),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, strides),
                nn.Tanh()
            ) 

    def forward(self, noise):
        x = noise.viewn(len(noise), self.input_dim,1,1)
        return self.gen(x)


class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_blocks(im_chan, hidden_dim),
            self.make_disc_blocks(hidden_dim, hidden_dim*2),
            self.make_disc_blocks(hidden_dim*2, 1, final_layer=True)
        )
        

    def make_disc_blocks(self,input_channels, output_channels, kernel_size=4, strides=2, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, strides),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, strides)
            ) 

    def forward(self,image):
        
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred),-1)


