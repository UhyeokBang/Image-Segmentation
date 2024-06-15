import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out

class Bottleneck(nn.Module):
    expansion = 4  # Expansion factor for channels in the bottleneck

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.fuse_method = fuse_method

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif i < j:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_channels[j], self.num_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.num_channels[i])
                    ))
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_channels_conv = self.num_channels[i]
                            convs.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], num_out_channels_conv, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_out_channels_conv)
                            ))
                        else:
                            num_out_channels_conv = self.num_channels[j]
                            convs.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], num_out_channels_conv, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_out_channels_conv),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=x[i].shape[2:], mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(F.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self):
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'BLOCK': Bottleneck, 'FUSE_METHOD': 'SUM'}
        self.stage2_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [18 * Bottleneck.expansion, 36 * Bottleneck.expansion], 'BLOCK': Bottleneck, 'FUSE_METHOD': 'SUM'}
        self.stage3_cfg = {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [18 * Bottleneck.expansion, 36 * Bottleneck.expansion, 72 * Bottleneck.expansion], 'BLOCK': Bottleneck, 'FUSE_METHOD': 'SUM'}
        self.stage4_cfg = {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [18 * Bottleneck.expansion, 36 * Bottleneck.expansion, 72 * Bottleneck.expansion, 144 * Bottleneck.expansion], 'BLOCK': Bottleneck, 'FUSE_METHOD': 'SUM'}


        self.transition1 = self._make_transition_layer([64], self.stage2_cfg['NUM_CHANNELS'])
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg)
        self.transition2 = self._make_transition_layer(pre_stage_channels, self.stage3_cfg['NUM_CHANNELS'])
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg)
        self.transition3 = self._make_transition_layer(pre_stage_channels, self.stage4_cfg['NUM_CHANNELS'])
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg)

        self.final_layer = nn.Conv2d(sum(pre_stage_channels), 1, kernel_size=1, stride=1, padding=0)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    convs.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*convs))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = layer_config['BLOCK']
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not i == 0:
                num_channels = [num_channels * block.expansion for num_channels in num_channels]
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_channels, fuse_method))
        return nn.Sequential(*modules), num_channels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)        

        x = torch.cat(y_list,1)
        x = self.final_layer(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        return x
