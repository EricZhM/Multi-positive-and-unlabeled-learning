import torch.nn.init as init
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dataset_name, out_number):
        super().__init__()
        if dataset_name == 'USPS':
            inp = 16
        elif dataset_name == 'MNIST':
            inp = 28
        elif dataset_name == 'KMNIST':
            inp = 28
        elif dataset_name == 'FashionMNIST':
            inp = 28
        else:
            raise NotImplementedError
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(inp*inp, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer2 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer3 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer4 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(True), nn.BatchNorm1d(300))
        self.layer5 = nn.Sequential(nn.Linear(300, out_number))

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class MLP5(nn.Module):
    def __init__(self, inp, out_number):
        super(MLP5, self).__init__()

        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Linear(inp * inp, 300),
            nn.ReLU(True),
            nn.BatchNorm1d(300)
        )
        init.kaiming_normal_(self.layer1[0].weight, mode='fan_in', nonlinearity='relu')
        if self.layer1[0].bias is not None:
            init.constant_(self.layer1[0].bias, 0)

        self.layer2 = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.BatchNorm1d(300)
        )
        init.kaiming_normal_(self.layer2[0].weight, mode='fan_in', nonlinearity='relu')
        if self.layer2[0].bias is not None:
            init.constant_(self.layer2[0].bias, 0)

        self.layer3 = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.BatchNorm1d(300)
        )
        init.kaiming_normal_(self.layer3[0].weight, mode='fan_in', nonlinearity='relu')
        if self.layer3[0].bias is not None:
            init.constant_(self.layer3[0].bias, 0)

        self.layer4 = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.BatchNorm1d(300)
        )
        init.kaiming_normal_(self.layer4[0].weight, mode='fan_in', nonlinearity='relu')
        if self.layer4[0].bias is not None:
            init.constant_(self.layer4[0].bias, 0)

        self.layer5 = nn.Sequential(
            nn.Linear(300, out_number)
        )
        init.xavier_normal_(self.layer5[0].weight)
        if self.layer5[0].bias is not None:
            init.constant_(self.layer5[0].bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)