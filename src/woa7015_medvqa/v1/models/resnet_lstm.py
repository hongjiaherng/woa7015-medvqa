import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.q_embed = nn.Embedding(5000, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(512 + hidden_dim, vocab_size)

    def forward(self, img, q):
        v = self.resnet(img)
        e = self.q_embed(q)
        _, (h, _) = self.lstm(e)
        h = h[-1]
        fused = torch.cat([v, h], dim=1)
        return self.classifier(fused)
