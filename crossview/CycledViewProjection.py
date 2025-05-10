import torch
import torch.nn as nn
import torch.nn.functional as F


class CycledViewProjection(nn.Module):
    def __init__(self, in_dim):
        super(CycledViewProjection, self).__init__()
        self.transform_module = TransformModule()
        self.retransform_module = TransformModule()

    def forward(self, x):
        B, C, H, W = x.shape
        transform_feature = self.transform_module(x)
        transform_features = transform_feature.view(B, C, H, W)
        retransform_features = self.retransform_module(transform_features)
        return transform_feature, retransform_features


class TransformModule(nn.Module):
    def __init__(self):
        super(TransformModule, self).__init__()
        self.fc_transform = None
        self.dim = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Dynamically set up fc_transform if not already done
        if self.fc_transform is None or self.dim != H:
            self.dim = H  # assumes square feature map
            input_dim = H * W
            self.fc_transform = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ).to(x.device)

        x = x.view(B, C, -1)              # Flatten H × W
        x = self.fc_transform(x)          # Apply MLP
        x = x.view(B, C, self.dim, self.dim)  # Restore spatial layout
        return x


# Optional test/debug
if __name__ == '__main__':
    features = torch.randn(8, 128, 32, 32)  # Simulate output of encoder
    CVP = CycledViewProjection(128)
    out, reout = CVP(features)
    print("Transform output:", out.shape)
    print("Retransform output:", reout.shape)
