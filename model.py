import torch.nn as nn
import torchvision.transforms.functional as TTF


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def deconv_block(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))


class Unet(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256]):
        super(Unet, self).__init__()
        self.channels = channels
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_downs = nn.ModuleList()
        self.conv_ups = nn.ModuleList()
        self.bottleneck = conv_block(channels[-1], channels[-1])
        self.deconvs = nn.ModuleList()

        for i in range(len(channels)):
            if i + 1 >= len(channels):
                break
            self.conv_downs.append(conv_block(channels[i], channels[i + 1]))

        channels[0] = 1
        channels = channels[::-1]
        for i in range(len(channels)):
            if i + 1 >= len(channels):
                break

            self.deconvs.append(deconv_block(channels[i], channels[i]))
            self.conv_ups.append(conv_block(channels[i], channels[i + 1]))

    def forward(self, x):
        residuals = []
        channels = self.channels

        # Encoder
        out = x
        for i in range(len(channels)):
            if i + 1 >= len(channels):
                break
            out = self.conv_downs[i](out)
            residuals.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        channels[0] = 1
        channels = channels[::-1]
        residuals.reverse()
        for i in range(len(channels)):
            if i + 1 >= len(channels):
                break

            out = self.deconvs[i](out)

            if out != residuals[i].shape:
                out = TTF.resize(out, residuals[i].shape[2:])

            out = out + residuals[i]
            out = self.conv_ups[i](out)

        return out


