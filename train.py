import torch
from torch import nn
from utils import *
from model import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Adam
from tqdm.auto import tqdm

mnist_shape = (1,28,28)
n_classes = 10

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)

generator_input_dim, discriminator_input_chn = get_input_dimensions(z_dim, mnist_shape,n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = Adam(gen.parameters(), lr=lr)

disc = Discriminator(im_chan=discriminator_input_chn).to(device)
disc_opt = Adam(disc.parameters(), lr=lr)

gen = gen.apply(weight_init)
disc = disc.apply(weight_init)

#Training Loop
cur_step = 0
generator_losses = []
discriminator_losses = []

noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

for epoch in range(n_epochs):

    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)

        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device),n_classes)
        image_one_hot_labels = one_hot_labels[:,:,None,None]
        image_one_hot_labels = image_one_hot_labels.repeat(1,1,mnist_shape[1],mnist_shape[2])
        # Discriminator
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size,z_dim,device=device)

        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        
        fake = gen(noise_and_labels)

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_real_loss + disc_fake_loss)/2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        discriminator_losses += [disc_loss.item()]

        #Generator
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        generator_losses += [gen_loss.item()]

        if cur_step % display_step == 0 and cur_step>0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(generator_losses[display_step:]) / display_step
            print(f"Step {cur_step} Epoch {epoch}: Generator loss: {gen_mean}, Discriminator loss: {disc_mean}")
            show_tensor_image(fake)
            show_tensor_image(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1,step_bins).mean(1),
                label='Generator Loss'
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()
        cur_step += 1