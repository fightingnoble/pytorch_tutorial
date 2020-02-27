import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from .quantize import QBN2d, QConv2d, QLinear
from .quantize2 import crxb_Conv2d, crxb_Linear

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

score_bit = 8
fin_bit = 8

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes, quantized=False, physical=False,
                 input_qbit=8, weight_qbit=8, activation_qbit=8, **crxb_cfg):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        i_bit = input_qbit
        w_bit = weight_qbit
        a_bit = score_bit
        if physical:
            self.classifier = nn.Sequential(crxb_Linear(n_channel, num_classes, input_qbit=i_bit, weight_qbit=w_bit,
                                                        activation_qbit=a_bit, **crxb_cfg))
        elif quantized:
            self.classifier = nn.Sequential(QLinear(n_channel, num_classes, True,
                                                    input_qbit=i_bit, weight_qbit=w_bit, activation_qbit=a_bit))
        else:
            self.classifier = nn.Sequential(nn.Linear(n_channel, num_classes))
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, quantized=False, physical=False,
                input_qbit=8, weight_qbit=8, activation_qbit=8, **crxb_cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            i_bit = input_qbit if i else fin_bit
            w_bit = weight_qbit
            a_bit = activation_qbit
            if physical:
                conv2d = crxb_Conv2d(in_channels, out_channels, kernel_size=3, padding=padding,
                                     input_qbit=i_bit, weight_qbit=w_bit, activation_qbit=a_bit, **crxb_cfg)
            elif quantized:
                conv2d = QConv2d(in_channels, out_channels, kernel_size=3, padding=padding,
                                 input_qbit=i_bit, weight_qbit=w_bit, activation_qbit=a_bit)
            else:
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                bn2d = nn.BatchNorm2d(out_channels, affine=False) if not physical | quantized else \
                    QBN2d(out_channels, activation_qbit, affine=False)
                layers += [conv2d, bn2d, nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def cifar10(n_channel, pretrained=None, **kwargs):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
           (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True, **kwargs)
    model = CIFAR(layers, n_channel=8 * n_channel, num_classes=10, **kwargs)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
           (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8 * n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
