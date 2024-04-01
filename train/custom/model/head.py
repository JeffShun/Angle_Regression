import torch.nn as nn

class RegresionHead(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_classes: int,
    ):
        super(RegresionHead, self).__init__()
        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_out = nn.Conv2d(num_features_in, num_classes, kernel_size=1)

    def forward(self, inputs):
        regressions = self.conv_out(self.avg_pool(inputs))
        return regressions.squeeze(-1).squeeze(-1)






