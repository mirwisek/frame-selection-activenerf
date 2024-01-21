import torch
import torch.nn.functional as F
from torchvision.models import resnet18

# Image encoder
class ImageEncoder(torch.nn.Module):
    def __init__(self, device):
      super().__init__()
      self.device = device
      self.resnet = resnet18(pretrained=True).to(device)
      self.resnet.eval()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjust the dimensions to match [channels, batch_size, height, width]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)
        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)
        latents = [feats1, feats2, feats3, feats4]
        output_size = (100, 100)
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], output_size, mode="bilinear", align_corners=True
            )
        latents = torch.cat(latents, dim=1)
        return latents