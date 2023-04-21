from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset




# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
# class ResNet18_client_side(nn.Module):
#     def __init__(self):
#         super(ResNet18_client_side, self).__init__()
#         self.layer1 = nn.Sequential (
#             nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
#             nn.BatchNorm2d(64),
#             nn.ReLU (inplace = True),
#             nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
#         )
#         self.layer2 = nn.Sequential  (
#             nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
#             nn.BatchNorm2d(64),
#             nn.ReLU (inplace = True),
#             nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
#             nn.BatchNorm2d(64),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # 其中，m.weight 表示子模块 m 的权重张量，.data 表示获取该张量的底层数据，并且.normal_() 表示在该数据上进行 Inplace 操作，即直接在原数据上修改而不返回新的数据。
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
#     def forward(self, x):
#         resudial1 = F.relu(self.layer1(x))
#         out1 = self.layer2(resudial1)
#         out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
#         resudial2 = F.relu(out1)
#         return resudial2
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResNet18_client_side(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet18_client_side, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
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

class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pool_size=4):  # Add a new argument pool_size
        super(ResNet18_server_side, self).__init__()
        self.in_planes = 64
        self.pool_size = pool_size  # Add this line to store pool_size
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        # print("Output shape before pooling:", out.shape)
        out = F.avg_pool2d(out, kernel_size=self.pool_size)  # Use self.pool_size instead of 8
        # print("Output shape after pooling:", out.shape)
        out = out.view(out.size(0), -1)
        y_hat = self.linear(out)
        return y_hat

# class Baseblock(nn.Module):
#     expansion = 1
#
#     def __init__(self, input_planes, planes, stride=1, dim_change=None):
#         super(Baseblock, self).__init__()
#         self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.dim_change = dim_change
#
#     def forward(self, x):
#         res = x
#         output = F.relu(self.bn1(self.conv1(x)))
#         output = self.bn2(self.conv2(output))
#
#         if self.dim_change is not None:
#             res = self.dim_change(res)
#
#         output += res
#         output = F.relu(output)
#
#         return output

# class ResNet18_server_side(nn.Module):
#     def __init__(self, block, num_layers, classes):
#         super(ResNet18_server_side, self).__init__()
#         self.input_planes = 64
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#         )
#
#         self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
#         self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
#         self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
#         self.averagePool = nn.AvgPool2d(kernel_size=7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _layer(self, block, planes, num_layers, stride=2):
#         dim_change = None
#         if stride != 1 or planes != self.input_planes * block.expansion:
#             dim_change = nn.Sequential(
#                 nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * block.expansion))
#         netLayers = []
#         netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
#         self.input_planes = planes * block.expansion
#         for i in range(1, num_layers):
#             netLayers.append(block(self.input_planes, planes))
#             self.input_planes = planes * block.expansion
#
#         return nn.Sequential(*netLayers)
#
#     def forward(self, x):
#         out2 = self.layer3(x)
#         out2 = out2 + x  # adding the resudial inputs -- downsampling not required in this layer
#         x3 = F.relu(out2)
#
#         x4 = self.layer4(x3)
#         x5 = self.layer5(x4)
#         x6 = self.layer6(x5)
#
#         x7 = F.avg_pool2d(x6, 1)
#         x8 = x7.view(x7.size(0), -1)
#         y_hat = self.fc(x8)
#
#         return y_hat


