from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain import get_model


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class AddCoords(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(b, 1, h, 1)
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(b, 1, 1, w)
        return torch.cat([x, xx, yy], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        maximum = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maximum, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([x.mean(1, keepdim=True), maximum], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAMBlock(nn.Module):
    def __init__(self, channels, spatial_kernel):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class ResBlockFusion(nn.Module):
    def __init__(self, input_channels, output_channels=64, coordinates=True, spatial_kernel=7):
        super().__init__()
        self.use_coords = coordinates
        self.add_coords = AddCoords() if coordinates else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv2d(input_channels + (2 if coordinates else 0), output_channels, 1, bias=False),
            LayerNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.ln1 = LayerNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.ln2 = LayerNorm2d(output_channels)
        self.cbam = CBAMBlock(output_channels, spatial_kernel)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        projected = self.proj(self.add_coords(x))
        out = self.act(self.ln1(self.conv1(projected)))
        out = self.ln2(self.conv2(out))
        out = self.cbam(out)
        return self.act(out + projected)


class StageFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stage_embed = nn.Embedding(3, 32)
        self.mlp = nn.Sequential(
            nn.Linear(33, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )
        self.width = channels

    def forward(self, feature, stage, beta):
        condition = torch.cat([self.stage_embed(stage), beta.view(-1, 1)], dim=1)
        scale, bias = self.mlp(condition).chunk(2, dim=1)
        scale = scale.view(feature.size(0), self.width, 1, 1)
        bias = bias.view(feature.size(0), self.width, 1, 1)
        return feature * (1.0 + scale) + bias


class FrozenCUBRecognizer(nn.Module):
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        config_path = str(Path(config_path).resolve())
        cfg = Config.fromfile(config_path)
        init_default_scope(cfg.get("default_scope", "mmpretrain"))
        model = get_model(config_path, pretrained=str(checkpoint_path), device="cpu")
        model.backbone.out_indices = (0, 1, 2, 3)
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])
        self.out_channels = {"p3": None, "p4": None, "p5": None}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image, return_cls=False):
        image = (image.clamp(0, 1) - self.mean) / (self.std + 1e-12)
        features = self.backbone(image)
        p3, p4, p5 = features[1], features[2], features[3]
        output = {"p3": p3, "p4": p4, "p5": p5}
        if return_cls:
            logits = self.head(self.neck(features))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            output["logits"] = logits
        return output


class WeightGenerationNetPro(nn.Module):
    def __init__(self, recognizer_config, recognizer_checkpoint):
        super().__init__()
        self.backbone = FrozenCUBRecognizer(recognizer_config, recognizer_checkpoint)
        self.stage_embed = nn.Embedding(3, 32)
        self.stage_to_map = nn.Sequential(
            nn.Linear(33, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
        )
        self.fuse_p5 = ResBlockFusion(2048 * 3, spatial_kernel=3)
        self.fuse_p4 = ResBlockFusion(1024 * 3, spatial_kernel=3)
        self.fuse_p3 = ResBlockFusion(512 * 3, spatial_kernel=5)
        self.fuse_p0 = ResBlockFusion(3 * 3, coordinates=False, spatial_kernel=7)
        self.reduce_p4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            LayerNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.reduce_p3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            LayerNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.final_fusion = ResBlockFusion(128, spatial_kernel=7)
        self.mu_head = nn.Conv2d(64, 1, 3, padding=1)
        self.logvar_head = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, task, anchor, stage, beta, sample, return_logits):
        task_features = self.backbone(task, return_cls=return_logits)
        anchor_features = self.backbone(anchor)
        image = self.fuse_p0(torch.cat([task, anchor, task - anchor], dim=1))
        p5 = self.fuse_p5(torch.cat([
            task_features["p5"], anchor_features["p5"],
            task_features["p5"] - anchor_features["p5"],
        ], dim=1))
        p4 = self.fuse_p4(torch.cat([
            task_features["p4"], anchor_features["p4"],
            task_features["p4"] - anchor_features["p4"],
        ], dim=1))
        p3 = self.fuse_p3(torch.cat([
            task_features["p3"], anchor_features["p3"],
            task_features["p3"] - anchor_features["p3"],
        ], dim=1))
        p5 = F.interpolate(p5, p4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.reduce_p4(torch.cat([p4, p5], dim=1))
        p4 = F.interpolate(p4, p3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.reduce_p3(torch.cat([p3, p4], dim=1))
        semantic = F.interpolate(p3, image.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.final_fusion(torch.cat([semantic, image], dim=1))
        condition = torch.cat([self.stage_embed(stage), beta.view(-1, 1)], dim=1)
        condition = self.stage_to_map(condition).view(task.size(0), 64, 1, 1)
        feature = feature + condition
        mean = self.mu_head(feature)
        logvar = self.logvar_head(feature)
        temperature = (1.0 / (1.0 + beta.view(-1, 1, 1, 1))).clamp(0.35, 1.0)
        gate_logits = mean
        if sample and self.training:
            gate_logits = gate_logits + torch.exp(0.5 * logvar.clamp(-8.0, 4.0)) * torch.randn_like(logvar)
        keep = torch.sigmoid(gate_logits / temperature)
        mediated = (keep * task + (1.0 - keep) * anchor).clamp(0.0, 1.0)
        return mediated, keep, mean, logvar, task_features.get("logits")


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels=24):
        super().__init__()
        expanded = channels * 2
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.sca = SimplifiedChannelAttention(channels)
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, expanded, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        y = residual + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class ContextConditionedNAFStage(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.num_stages = 3
        self.intro = nn.Conv2d(input_channels, 24, 3, padding=1)
        self.body = nn.Sequential(NAFBlock(), NAFBlock())
        self.film = StageFiLM(24)
        self.outro = nn.Conv2d(24, 3, 3, padding=1)
        nn.init.zeros_(self.outro.weight)
        nn.init.zeros_(self.outro.bias)


class NAFNetTaskStage(ContextConditionedNAFStage):
    def __init__(self):
        super().__init__(4)

    def forward(self, x, stage, beta):
        beta_map = beta.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])
        feature = self.intro(torch.cat([x, beta_map], dim=1))
        feature = self.film(feature, stage, beta)
        return (x + self.outro(self.body(feature))).clamp(0.0, 1.0)


class NAFNetProxStage(ContextConditionedNAFStage):
    def __init__(self):
        super().__init__(10)

    def forward(self, x, anchor, stage, beta):
        beta_map = beta.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])
        packed = torch.cat([x, anchor, x - anchor, beta_map], dim=1)
        feature = self.intro(packed)
        feature = self.film(feature, stage, beta)
        return (x + self.outro(self.body(feature))).clamp(0.0, 1.0)


class CaMeU(nn.Module):
    def __init__(self, recognizer_config, recognizer_checkpoint):
        super().__init__()
        self.beta_params = nn.Parameter(torch.full((3,), 0.1))
        self.register_buffer("T_param", torch.tensor(0.5))
        self.lpn = WeightGenerationNetPro(recognizer_config, recognizer_checkpoint)
        self.task_net = NAFNetTaskStage()
        self.prox_net = NAFNetProxStage()

    @property
    def recognizer(self):
        return self.lpn.backbone

    def train(self, mode=True):
        super().train(mode)
        self.recognizer.eval()
        return self

    def forward(self, restored, stages=3, return_training_state=False):
        anchor = restored
        states = []
        training_path = self.training or return_training_state
        for index in range(stages):
            stage = torch.full((restored.size(0),), index, dtype=torch.long, device=restored.device)
            beta_scalar = F.softplus(self.beta_params[index])
            beta = beta_scalar.expand(restored.size(0))
            task = self.task_net(anchor, stage, beta)
            mediated, keep, mean, logvar, task_logits = self.lpn(
                task, anchor, stage, beta, self.training, training_path
            )
            mediated_logits = None
            if training_path:
                mediated_logits = self.recognizer(mediated, return_cls=True)["logits"]
            projected = self.prox_net(mediated, anchor, stage, beta)
            states.append({
                "beta": beta_scalar,
                "anchor": anchor,
                "task": task,
                "mediated": mediated,
                "projected": projected,
                "keep": keep,
                "mean": mean,
                "logvar": logvar,
                "task_logits": task_logits,
                "mediated_logits": mediated_logits,
            })
            anchor = projected
        final_logits = None
        if training_path:
            final_logits = self.recognizer(anchor, return_cls=True)["logits"]
        if return_training_state:
            return anchor, states, final_logits
        return anchor


class CaMeULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        import lpips

        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        self.lpips.requires_grad_(False)

    def image_loss(self, prediction, target):
        return F.l1_loss(prediction, target) + self.lpips(prediction, target, normalize=True).mean()

    def forward(self, output, states, final_logits, target, labels):
        final_image = self.image_loss(output, target)
        stage_image = sum(self.image_loss(state["projected"], target) for state in states)
        task_ce = torch.stack([F.cross_entropy(state["task_logits"], labels) for state in states]).mean()
        mediated_ce = torch.stack([
            F.cross_entropy(state["mediated_logits"], labels) for state in states
        ]).mean()
        recognition = task_ce + 0.2 * mediated_ce + 0.2 * F.cross_entropy(final_logits, labels)
        local = output.new_zeros(())
        fidelity = output.new_zeros(())
        kl = output.new_zeros(())
        for state in states:
            beta = state["beta"]
            residual = state["keep"] * (state["task"] - state["anchor"])
            local = local + 0.5 * beta * residual.square().mean()
            fidelity = fidelity + 0.5 * beta * (
                state["projected"] - state["mediated"]
            ).square().mean()
            kl = kl + 0.5 * (
                state["logvar"].exp() + state["mean"].square() - 1.0 - state["logvar"]
            ).mean()
        total = final_image + stage_image + recognition + 0.1 * local + 0.1 * fidelity + 1e-4 * kl
        return total, {
            "total": total.detach(),
            "final_image": final_image.detach(),
            "stage_image": stage_image.detach(),
            "recognition": recognition.detach(),
            "local": local.detach(),
            "fidelity": fidelity.detach(),
            "kl": kl.detach(),
        }
