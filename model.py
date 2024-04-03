import torch
import torch.nn as nn
import torch.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim=100):
        super(Generator, self).__init__()
        self.channels = [512, 256, 128, 64, 3]
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.conv_trans_1 = self._build_conv_trans_block(input_dim, self.channels[0], 6, padding=0, stride=self.stride)
        self.conv_trans_2 = self._build_conv_trans_block(self.channels[0], self.channels[1], self.kernel_size, self.padding, self.stride)
        self.conv_trans_3 = self._build_conv_trans_block(self.channels[1], self.channels[2], self.kernel_size, self.padding, self.stride)
        self.conv_trans_4 = self._build_conv_trans_block(self.channels[2], self.channels[3], self.kernel_size, self.padding, self.stride)
        self.conv_trans_5 = self._build_conv_trans_block(self.channels[3], self.channels[4], self.kernel_size, self.padding, self.stride,
                                                       final_layer=True)

    def _build_conv_trans_block(self, in_channels, out_channels, kernel_size, padding, stride, final_layer=False):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if not final_layer:
            layers += [nn.BatchNorm2d(out_channels), nn.ReLU(True)]
        else:
            layers += [nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.conv_trans_1(input_tensor)
        x = self.conv_trans_2(x)
        x = self.conv_trans_3(x)
        x = self.conv_trans_4(x)
        x = self.conv_trans_5(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = [3, 64, 128, 256, 512]
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.conv_1 = self._build_conv_block(self.channels[0], self.channels[1], self.kernel_size, self.stride, self.padding, first_layer=True)
        self.conv_2 = self._build_conv_block(self.channels[1], self.channels[2], self.kernel_size, self.stride, self.padding)
        self.conv_3 = self._build_conv_block(self.channels[2], self.channels[3], self.kernel_size, self.stride, self.padding)
        self.conv_4 = self._build_conv_block(self.channels[3], self.channels[4], self.kernel_size, self.stride, self.padding)
        self.conv_5 = self._build_conv_block(self.channels[4], 1, self.kernel_size + 1, self.stride, 0, final_layer=True)

    def _build_conv_block(self, in_channels, out_channels, kernel_size, stride, padding, first_layer=False, final_layer=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if first_layer:
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif final_layer:
            layers += [nn.Sigmoid()]
        else:
            layers += [nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x


def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


if __name__ == "__main__":
    generator = Generator()
    noise_vector = torch.randn(10, 100, 1, 1)
    fake_images = generator(noise_vector)
    print(fake_images.size())

    discriminator = Discriminator()
    real_images = torch.randn(10, 3, 96, 96)
    decision_output = discriminator(real_images)
    print(decision_output.size())
