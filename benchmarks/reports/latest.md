# Benchmark Report

Generated: 2026-04-06T08:02:08.413Z

| Benchmark | Level | Nodes | Errors | Warnings | Diagnostics | Sample | Strategy | Resolution |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| MNIST Net | 1 | 14 | 0 | 0 | 1 | fallback | fallback | 128 × 128 |
| Super Resolution Net | 1 | 9 | 0 | 0 | 0 | fallback | fallback | 500 × 500 |
| AlexNet | 1 | 23 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| SqueezeNet 1.0 | 1 | 67 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| VGG 11 | 2 | 31 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| VGG 16 | 2 | 41 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| ResNet 18 | 2 | 70 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| DenseNet 121 | 2 | 432 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| GoogLeNet | 3 | 198 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| Inception V3 | 3 | 317 | 0 | 0 | 2 | repo | explicit-path | 500 × 500 |
| MobileNet V2 | 3 | 154 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| MobileNet V3 Large | 3 | 188 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| EfficientNet B0 | 4 | 250 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| ConvNeXt Tiny | 4 | 193 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| ShuffleNet V2 x1.0 | 4 | 259 | 0 | 0 | 0 | repo | explicit-path | 500 × 500 |
| RegNet Y 400MF | 4 | 271 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |
| Swin Transformer Tiny | 5 | 993 | 0 | 0 | 90 | repo | explicit-path | 512 × 512 |
| Vision Transformer B16 | 5 | 142 | 0 | 0 | 1 | repo | explicit-path | 512 × 512 |
| MaxViT T | 5 | 1554 | 0 | 0 | 22 | repo | explicit-path | 500 × 500 |
| Wide ResNet 50-2 | 5 | 176 | 0 | 0 | 0 | repo | explicit-path | 512 × 512 |

## Trace Modes

| Benchmark | Trace Mode | Exactness | Unsupported Reason |
| --- | --- | --- | --- |
| MNIST Net | runtime-exact | runtime_exact | — |
| Super Resolution Net | runtime-exact | runtime_exact | — |
| AlexNet | runtime-exact | runtime_exact | — |
| SqueezeNet 1.0 | runtime-exact | runtime_exact | — |
| VGG 11 | runtime-exact | runtime_exact | — |
| VGG 16 | runtime-exact | runtime_exact | — |
| ResNet 18 | runtime-exact | runtime_exact | — |
| DenseNet 121 | runtime-exact | runtime_exact | — |
| GoogLeNet | runtime-exact | runtime_exact | — |
| Inception V3 | runtime-exact | runtime_exact | — |
| MobileNet V2 | runtime-exact | runtime_exact | — |
| MobileNet V3 Large | runtime-exact | runtime_exact | — |
| EfficientNet B0 | runtime-exact | runtime_exact | — |
| ConvNeXt Tiny | runtime-exact | runtime_exact | — |
| ShuffleNet V2 x1.0 | runtime-exact | runtime_exact | — |
| RegNet Y 400MF | runtime-exact | runtime_exact | — |
| Swin Transformer Tiny | runtime-exact | runtime_exact | — |
| Vision Transformer B16 | runtime-exact | runtime_exact | — |
| MaxViT T | runtime-exact | runtime_exact | — |
| Wide ResNet 50-2 | runtime-exact | runtime_exact | — |

## Sample Provenance

### MNIST Net
- source: fallback
- strategy: fallback
- path: benchmarks/samples/mnist-digit.png
- resolution: 128 × 128
- evidence: Repository sample was not available, so a curated image-classification fallback asset is shown.

### Super Resolution Net
- source: fallback
- strategy: fallback
- path: benchmarks/samples/vision-classification.jpg
- resolution: 500 × 500
- evidence: Repository sample was not available, so a curated image-super-resolution fallback asset is shown.

### AlexNet
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog1.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog1.jpg.

### SqueezeNet 1.0
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog2.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog2.jpg.

### VGG 11
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### VGG 16
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### ResNet 18
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### DenseNet 121
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog2.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog2.jpg.

### GoogLeNet
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog1.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog1.jpg.

### Inception V3
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog1.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog1.jpg.

### MobileNet V2
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### MobileNet V3 Large
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### EfficientNet B0
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog2.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog2.jpg.

### ConvNeXt Tiny
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### ShuffleNet V2 x1.0
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog2.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog2.jpg.

### RegNet Y 400MF
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### Swin Transformer Tiny
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### Vision Transformer B16
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.

### MaxViT T
- source: repo
- strategy: explicit-path
- path: gallery/assets/dog1.jpg
- resolution: 500 × 500
- evidence: Resolved from the benchmark repository at gallery/assets/dog1.jpg.

### Wide ResNet 50-2
- source: repo
- strategy: explicit-path
- path: gallery/assets/astronaut.jpg
- resolution: 512 × 512
- evidence: Resolved from the benchmark repository at gallery/assets/astronaut.jpg.


## Error Diagnostics


## Edit Flow

- benchmark: level2-resnet18
- dragged block: ReLU
- code generated: true
- save to worktree: true
- block palette search 'relu' matches: 4
- activation collapse hides ReLU: true
- codex source-aware edit commit: 8cc11de
- codex source-aware edit branch: dl-viz-codex-1775463423432
- codex changed files: torchvision/models/resnet.py

### Codex Source-aware Edit Diff

```diff
commit 8cc11decf40766b42906e0114f4d6b727057bce4
Author: leehogwang <leehogwang05@gmail.com>
Date:   Mon Apr 6 17:17:29 2026 +0900

    dl-viz: codex edited resnet.py
---
 torchvision/models/resnet.py | 1 +
 1 file changed, 1 insertion(+)

diff --git a/torchvision/models/resnet.py b/torchvision/models/resnet.py
index 47067ec..a03f503 100644
--- a/torchvision/models/resnet.py
+++ b/torchvision/models/resnet.py
@@ -163,6 +163,7 @@ class Bottleneck(nn.Module):
         return out
 
 
+# codex benchmark marker
 class ResNet(nn.Module):
     def __init__(
```