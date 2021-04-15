# Vanilla-GAN

A Generative Adversarial Network (GAN) is a class of machine learning systems   
invented by Ian Goodfellow in 2014. Two neural networks contest with each   
other in a game (in the sense of game theory, often but not always in the   
form of a zero-sum game).Given a training set, this technique learns to   
generate new data with the same statistics as the training set.  

This code uses the MNIST dataset to train itself and afterwards   
generates the similar images. The MNIST database (Modified National I  
nstitute of Standards and Technology database) is a large  
 database of handwritten digits that is commonly used   
for training various image processing systems. The MNIST   
database contains 60,000 training images and 10,000 testing images.  

A Generative Adversarial Network has two main Neural Networks.  

## Generator
The Generator takes a random vector(Noise), and generates   
a 28x28 image. The generator triesto resambles the generated   
image with the input MNIST dataset images.  

## Discriminator

The discriminator distinguish between the real images and the generated  
 fake images. Its input is a 28x28 image and output is its probability    
of being real.  

![alt text](https://www.researchgate.net/publication/333831200/figure/fig5/AS:782113389412353@1563481771648/GAN-framework-in-which-the-generator-and-discriminator-are-learned-during-the-training.png)

## References
* Ian J. Goodfellow et. al. 2014 Generative Adversarial Networks  [Link](https://arxiv.org/abs/1406.2661)
* Mehdi Mirza et. al. 2014 Conditional Geneative Adversarial Nets [Link](https://arxiv.org/abs/1411.1784)
