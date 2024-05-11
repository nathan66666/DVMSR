# DVMSR
This repository contains the source code for our paper:

[DVMSR: Distillated Vision Mamba for Efficient Super-Resolution](https://arxiv.org/abs/2405.03008)

CVPRW 2024

Xiaoyan Lei, Wenlong Zhang, Weifeng Cao

## Abstract
Efficient Image Super-Resolution (SR) aims to accelerate SR network inference by minimizing computational complexity and network parameters while preserving performance. Existing state-of-the-art Efficient Image Super-Resolution methods are based on convolutional neural networks. Few attempts have been made with Mamba to harness its long-range modeling capability and efficient computational complexity, which have shown impressive performance on high-level vision tasks. In this paper, we propose DVMSR, a novel lightweight Image SR network that incorporates Vision Mamba and a distillation strategy. The network of DVMSR consists of three modules: feature extraction convolution, multiple stacked Residual State Space Blocks (RSSBs), and a reconstruction module. Specifically, the deep feature extraction module is composed of several residual state space blocks (RSSB), each of which has several Vision Mamba Moudles(ViMM) together with a residual connection. To achieve efficiency improvement while maintaining comparable performance, we employ a distillation strategy to the vision Mamba network for superior performance. Specifically, we leverage the rich representation knowledge of teacher network as additional supervision for the output of lightweight student networks. Extensive experiments have demonstrated that our proposed DVMSR can outperform state-of-the-art efficient SR methods in terms of model parameters while maintaining the performance of both PSNR and SSIM.

## Quick test
How to test the model?
1. Create a new environment
   
```bash
conda create -n DVMSR python=3.10.13
conda activate DVMSR
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```


2. Clone the repo
```bash
git clone https://github.com/nathan66666/DVMSR.git
```

3. Install dependent packages
```bash
cd DVMSR
pip install -r requirements.txt
```
4. Testing Command
```bash
python test_demo.py
```

How to calculate the number of parameters, FLOPs, and activations

```bash
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team07_DVMSR import DVMSR
    from fvcore.nn import FlopCountAnalysis

    model = DVMSR()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # fvcore is used in NTIRE2024_ESR for FLOPs calculation
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops/10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```
