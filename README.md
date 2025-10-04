# TSLDSeg
Official implementation of paper: "TSLDSeg: A Texture-aware and Semantic-enhanced Latent Diffusion Model for Medical Image Segmentation"

This repository is based on [SDSeg](https://github.com/lin-tianyu/Stable-Diffusion-Seg),  
a latent diffusion model for medical image segmentation.  
Specifically, our work focuses on alleviating the information loss of perceptual compression in conditioning by:
- Enhancing fine-grained representations to preserve high-frequency details (edges, fine textures, orientations).
- Leveraging hypergraph modeling to capture semantic relationships and spatial/topological constraints.

## Requirements

A suitable [conda](https://conda.io/) environment named `TSLDSeg` can be created
and activated with:

```bash
conda env create -f environment.yaml
conda activate TSLDSeg
```

Then, install some dependencies by:
```bash
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

    
<details>

<summary>Solve GitHub connection issues when downloading <code class="inlinecode">taming-transformers</code> or <code class="inlinecode">clip</code></summary>


After creating and entering the `TSLDSeg` environment:
1. create an `src` folder and enter:
```bash
mkdir src
cd src
```
2. download the following codebases in `*.zip` files and upload to `src/`:
    - https://github.com/CompVis/taming-transformers, `taming-transformers-master.zip`
    - https://github.com/openai/CLIP, `CLIP-main.zip`
3. unzip and install taming-transformers:
```bash
unzip taming-transformers-master.zip
cd taming-transformers-master
pip install -e .
cd ..
```
4. unzip and install clip:
```bash
unzip CLIP-main.zip
cd CLIP-main
pip install -e .
cd ..
```
5. install TSLDSeg:
```bash
cd ..
pip install -e .
```

Then you're good to go!

</details>

## Model Weights

### Pretrained Models
TSLDSeg uses pre-trained weights from LDM to initialize before training.

For pre-trained weights of the autoencoder and conditioning model, run

```bash
bash scripts/download_first_stages_f8.sh
```

For pre-trained wights of the denoising UNet, run

```bash
bash scripts/download_models_lsun_churches.sh
```


## Scripts
### Training Scripts

Take CVC dataset as an example, run

```bash
nohup python -u main.py --base configs/latent-diffusion/cvc-ldm-kl-8.yaml -t --gpus 0, --name experiment_name > nohup/experiment_name.log 2>&1 &
```

You can check the training log by 

```bash
tail -f nohup/experiment_name.log
```

Also, tensorboard will be on automatically. You can start a tensorboard session with `--logdir=./logs/`. For example,
```bash
tensorboard --logdir=./logs/
```

> [!NOTE]
> If you want to use parallel training, the code `trainer_config["accelerator"] = "gpu"` in `main.py` should be changed to `trainer_config["accelerator"] = "ddp"`. However, parallel training is not recommended since it has no performance gain (in my experience).

> [!WARNING]
> A single TSLDSeg model ckeckpoint is around 5GB. By default, save only the last model and the model with the highest dice score. If you have tons of storage space, feel free to save more models by increasing the `save_top_k` parameter in `main.py`.


### Testing Scripts

After training an TSLDSeg model, you should **manually modify the run paths** in `scripts/slice2seg.py`, and begin an inference process like

```bash
python -u scripts/slice2seg.py --dataset cvc
```
## Acknowledgement

This work is built upon the following open-source projects. We sincerely thank the authors for their excellent contributions:

- [SDSeg](https://github.com/lin-tianyu/Stable-Diffusion-Seg)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

