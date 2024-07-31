import torch
import torch.nn as nn

# Assume vae is your pretrained VAE model
class SingleChannelToThreeChannel(nn.Module):
    def __init__(self):
        super(SingleChannelToThreeChannel, self).__init__()
        # Define a conv layer to convert 1 channel to 3 channels
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding="same")
    
    def forward(self, x):
        return self.conv(x)
    
class ThreeChannelToSingleChannel(nn.Module):
    def __init__(self):
        super(ThreeChannelToSingleChannel, self).__init__()
        # Define a conv layer to convert 3 channels to 1 channel
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding="same")
    
    def forward(self, x):
        return self.conv(x)
    
# Define a new model that integrates the new input layer with the VAE
class VAEWithSingleChannelInput(nn.Module):
    def __init__(self, vae):
        super(VAEWithSingleChannelInput, self).__init__()
        self.input_layer = SingleChannelToThreeChannel()
        self.vae = vae
    
    def forward(self, x):
        x = self.input_layer(x)
        return self.vae(x)
    
# Define a new model that integrates the new input and output layers with the VAE
class VAEWithSingleChannelInputOutput(nn.Module):
    def __init__(self, vae):
        super(VAEWithSingleChannelInputOutput, self).__init__()
        self.input_layer = SingleChannelToThreeChannel()
        self.vae = vae
        self.output_layer = ThreeChannelToSingleChannel()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.vae(x)
        x = self.output_layer(x)
        return x