# pytorch-mnist-Conditional-VAE
Conditional Variational AutoEncoder on the MNIST dataset using PyTorch with one-hot encoded labels for controlled digit generation.

## Features
- **Conditional Generation**: Generate specific digits (0-9) by providing target labels
- **Raw MNIST Loading**: Direct loading from MNIST binary files
- **One-hot Encoding**: Labels converted to one-hot vectors for conditional training
- **Concatenated Input**: Images (784 dims) + Labels (10 dims) = 794-dimensional input
- **Controlled Sampling**: Generate specific digits on demand

## Dependencies
- PyTorch
- torchvision
- numpy
- matplotlib (for visualization)

## Architecture
- **Encoder**: Takes concatenated image+label (794 dims) → latent space (2 dims)
- **Decoder**: Takes latent vector + label → reconstructed image (784 dims)
- **Conditional Training**: Uses both image and label information for better generation control

## Results
### Conditional Generation
Generate specific digits by providing target labels:

![Digit 0](samples/digit_0.png)
*Generated samples for digit 0*

![Digit 1](samples/digit_1.png)
*Generated samples for digit 1*

![Digit 2](samples/digit_2.png)
*Generated samples for digit 2*

![Digit 3](samples/digit_3.png)
*Generated samples for digit 3*

![Digit 4](samples/digit_4.png)
*Generated samples for digit 4*

## Usage
```python
# Generate specific digit
samples = generate_conditional_samples(digit=5, num_samples=16)

# Load raw MNIST data
train_images = load_mnist_images('./mnist_data/raw/train-images-idx3-ubyte')
train_labels = load_mnist_labels('./mnist_data/raw/train-labels-idx1-ubyte')

# Convert to one-hot encoding
train_labels_onehot = F.one_hot(train_labels_tensor, num_classes=10).float()

# Concatenate for conditional training
train_conditional_data = torch.cat([train_images_flat, train_labels_onehot], dim=1)
```

## Key Improvements
1. **Conditional VAE**: Modified architecture to handle label conditioning
2. **Raw Data Loading**: Direct loading from MNIST binary files
3. **One-hot Encoding**: Proper label encoding for conditional generation
4. **Controlled Generation**: Generate specific digits by providing target labels
5. **Better Architecture**: Separate handling of image and label components

## Reference
1. Auto-Encoding Variational Bayes. Diederik P Kingma, Max Welling (paper): 
https://arxiv.org/abs/1312.6114
2. pytorch-MNIST-VAE(github):
https://github.com/lyeoni/pytorch-mnist-VAE/tree/master
3. Basic VAE Example (github): 
https://github.com/pytorch/examples/tree/master/vae
4. hwalsuklee/tensorflow-mnist-VAE (github): 
https://github.com/hwalsuklee/tensorflow-mnist-VAE
