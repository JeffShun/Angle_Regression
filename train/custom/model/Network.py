import torch
import torch.nn as nn

class Regression_Network(nn.Module):
    def __init__(
        self, 
        backbone, 
        head, 
        apply_sync_batchnorm=False
    ):
        super(Regression_Network, self).__init__()
        self.backbone = backbone
        self.head = head

        self.initialize_weights()
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.export
    def forward(self, img):
        features = self.backbone(img)
        regressions = self.head(features)
        return regressions

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
