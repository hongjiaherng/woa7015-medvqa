import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTMClassifier(nn.Module):
    """
    CNN (ResNet18) + LSTM question encoder + fusion + classifier.
    Answer prediction is classification over Top-K answers (+ <unk>).
    """

    def __init__(
        self,
        num_answers: int,
        question_vocab_size: int,
        img_feat_dim: int = 256,
        txt_emb_dim: int = 256,
        lstm_hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_cnn: bool = False,
    ):
        super().__init__()

        # ---- CNN backbone ----
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # -> (B,512,1,1)
        self.img_fc = nn.Linear(512, img_feat_dim)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # ---- Question encoder ----
        self.emb = nn.Embedding(question_vocab_size, txt_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(txt_emb_dim, lstm_hidden_dim, batch_first=True)

        # ---- Fusion + classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + lstm_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_answers),
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        # images: (B,C,H,W)
        # questions: (B,T)
        img_feat = self.cnn(images).squeeze(-1).squeeze(-1)  # (B,512)
        img_feat = self.img_fc(img_feat)  # (B,img_feat_dim)

        q_emb = self.emb(questions)  # (B,T,E)
        _, (h, _) = self.lstm(q_emb)
        q_feat = h[-1]  # (B,lstm_hidden_dim)

        fused = torch.cat([img_feat, q_feat], dim=1)
        logits = self.classifier(fused)
        return logits
