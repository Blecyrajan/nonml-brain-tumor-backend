import torch
import torch.nn as nn
import timm

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[4]
        )

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0
        )

        self.classifier = nn.Linear(512 + 192, num_classes)

    def forward(self, x):
        cnn_maps = self.cnn(x)[0]
        cnn_feat = torch.mean(cnn_maps, dim=(2, 3))
        vit_feat = self.vit(x)

        fused = torch.cat((cnn_feat, vit_feat), dim=1)
        logits = self.classifier(fused)

        return logits, cnn_maps


model = BrainTumorModel(num_classes=4)

state_dict = torch.load(
    "model/brain_tumor_rsc_model (1).pth",
    map_location=torch.device("cpu")
)

model.load_state_dict(state_dict)
model.eval()
