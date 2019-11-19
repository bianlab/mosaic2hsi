import torch
import torch.nn as nn

dtype = 'float32'


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 9)
        self.input = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3,
                                kernel_size=3, stride=1, padding=1, bias=False)
        '''self.mulSpec_layer1 = nn.Conv2d(in_channels=3, out_channels=31,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.mulSpec_layer2 = nn.Conv2d(in_channels=31, out_channels=31,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.mulSpec_layer3 = nn.Conv2d(in_channels=31, out_channels=31,
                                        kernel_size=1, stride=1, padding=0)'''
        self.mulSpec_layer1 = nn.Conv2d(in_channels=1, out_channels=31,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.mulSpec_layer2 = nn.Conv2d(in_channels=1, out_channels=31,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.mulSpec_layer3 = nn.Conv2d(in_channels=1, out_channels=31,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.input2 = nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.residual_layer2 = self.make_layer(Conv_ReLU_Block, 15)
        self.output2 = nn.Conv2d(in_channels=64, out_channels=31, kernel_size=3,
                                 padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))'''

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        '''out = self.relu(self.mulSpec_layer1(out))
        out = self.relu(self.mulSpec_layer2(out))
        out = self.mulSpec_layer3(out)'''
        out1 = self.relu(self.mulSpec_layer1(out[0:, 0, 0:, 0:].view(-1, 1, out.shape[2], out.shape[3])))
        out2 = self.relu(self.mulSpec_layer2(out[0:, 1, 0:, 0:].view(-1, 1, out.shape[2], out.shape[3])))
        out3 = self.relu(self.mulSpec_layer3(out[0:, 2, 0:, 0:].view(-1, 1, out.shape[2], out.shape[3])))
        out = torch.add(torch.add(out1, out2), out3)
        residual2 = out
        out = self.relu(self.input2(out))
        out = self.residual_layer2(out)
        out = self.output2(out)
        out = torch.add(out, residual2)
        return out


