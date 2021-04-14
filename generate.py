import torch 
from model import Generator
from utils import show_tensor_image, get_noise, get_input_dimensions, get_one_hot_labels, combine_vectors
import matplotlib.pyplot as plt

z_dim = 64
n_classes = 10
mnist_shape = (1,28,28)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen_input_dim, disc_input_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

gen = Generator(input_dim=gen_input_dim).to(device)
gen = gen.eval()
gen.load_state_dict(torch.load('gen.pt',map_location=torch.device(device)))

class_ = int(input('Enter class: '))

noise = get_noise(1,z_dim)
one_hot_label = get_one_hot_labels(torch.Tensor([class_]).long(),n_classes)
noise_and_label = combine_vectors(noise, one_hot_label)

image = gen(noise_and_label)

show_tensor_image(image)


