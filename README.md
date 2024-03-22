# DVMSR

# Quick test
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
