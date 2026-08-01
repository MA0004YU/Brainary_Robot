# Simulation 权重下载

物理沙盒的感知(GroundingDINO 检测 + SAM 分割)需要两个权重,放在 `simulation/weights/cg_weights/`。

```bash
cd simulation/weights/cg_weights

# 1) GroundingDINO(~660MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 2) SAM ViT-H(~2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

下完应有:
```
simulation/weights/cg_weights/
├── groundingdino_swint_ogc.pth
└── sam_vit_h_4b8939.pth
```

> FoundationPose 的权重(位姿细化/打分)另见 `simulation/weights/fp_weights/`(`config/global_config.yaml`
> 的 `weights_config` 指向它);若做位姿估计还需补这部分。

这些权重体积大,**不进 git**(已在 .gitignore 排除 `simulation/weights/`)。
