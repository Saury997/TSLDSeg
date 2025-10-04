import argparse
import glob
import os
import time

import numpy
import torch
from einops import rearrange
from omegaconf import OmegaConf
from ptflops import get_model_complexity_info
from pytorch_lightning import seed_everything
from thop import profile
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config
from ldm.data import CVCTest, KSEGTest, ETISTest, ISICTest, DSBTest, BUSITest


def prepare_for_first_stage(x, gpu=True):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    if gpu:
        x = x.to(memory_format=torch.contiguous_format).float().cuda()
    else:
        x = x.float()
    return x


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # if targs is None:
    #     return None
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() > 0 and targs.sum() == 0:
        return 1
    elif pred.sum() > 0 and targs.sum() > 0:
        # intersection = (pred * targs).sum()
        # union = pred.sum() + targs.sum() - intersection
        # return (2. * intersection) / (union + 10e-6)
        return (2. * (pred * targs).sum()) / (pred.sum() + targs.sum() + 1e-10)
    else:
        return 0


def iou_score(pred, targs):
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)

    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    # return intersection, union
    return intersection / (union + 1e-10)


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    x_grid = "cond_stage_model.oeem.orient_block.gabor_conv.x_grid"
    if x_grid in sd:
        sd[x_grid] = sd[x_grid].clone()
    y_grid = "cond_stage_model.oeem.orient_block.gabor_conv.y_grid"
    if y_grid in sd:
        sd[y_grid] = sd[y_grid].clone()

    model = instantiate_from_config(config.model)
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    print("\033[31mmissing keys:\033[0m")
    print(m)
    print("\033[31munexpected keys:\033[0m")
    print(u)
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["pred_sum"] > 0 and kwargs["gt_sum"] > 0:
        return 2 * inter / (union + 1e-10)
    elif kwargs["pred_sum"] > 0 and kwargs["gt_sum"] == 0:
        return 1
    else:
        return 0

def cal_params_flops(model, size):
    input = torch.randn(1, 4, size//8, size//8).cuda()
    c = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input, c, numpy.array([-1])))
    print('flops', flops / 1e9)  ## 打印计算量
    print('params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))


# Example function to calculate and print GMACs and parameter count for a given model
def print_model_stats(model, input_size=(3, 224, 224)):
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model created, param count: {total_params}')

    # Calculate GMACs using ptflops
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)

    # Display GMACs and params
    print(f'Model: {macs} GMACs, {params} parameters')


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["raw", "direct", "ddim", "plms", "dpm_solver"],
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument("--dataset", type=str, default='busi',  # '-b' for binary, '-m' for multi
                        help="uses the model trained for given dataset", )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/kseg-ldm-kl-8.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/lsun_churches256/model.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--times", type=int, default=1,
                        help="times of testing for stability evaluation", )
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the predictions for the whole test set.", )
    opt = parser.parse_args()

    if opt.dataset == "cvc":
        run = "debug_runs/2024-12-29T06-35-18_test0"
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-cvc2kseg-sd"
        dataset = KSEGTest()
    elif opt.dataset == "kseg":
        run = "2025-01-07T05-41-52_trainingOnKSEGwithFinetune"
        print("Evaluate on kseg dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-kseg2cvc-sd"
        dataset = CVCTest()
    elif opt.dataset == "dsb18":
        run = "2025-02-15T10-27-27_DSB18"
        print("Evaluate on DSB18 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-samples-dsb18"
        dataset = DSBTest()
    elif opt.dataset == "etis":
        run = "2025-02-14T02-58-24_ETIS"
        print("Evaluate on DSB18 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-samples-etis"
        dataset = ETISTest()
    elif opt.dataset == "isic17":
        run = "2025-02-12T09-24-16_ISIC17"
        print("Evaluate on DSB18 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-samples-isic17"
        dataset = ISICTest(year=17)
    elif opt.dataset == "busi":
        run = "2025-04-07T14-08-23_BUSI"
        print("Evaluate on DSB18 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("/root/Stable-Diffusion-Seg", "logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"/root/Stable-Diffusion-Seg/logs/{run}/checkpoints/last.ckpt"
        opt.outdir = "/root/Stable-Diffusion-Seg/outputs/slice2seg-samples-busi"
        dataset = BUSITest()
    else:
        raise NotImplementedError(f"Not implement for dataset {opt.dataset}")

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    config = OmegaConf.load(f"{opt.config}")
    config["model"]["params"].pop("ckpt_path")
    config["model"]["params"]["cond_stage_config"]["params"].pop("ckpt_path")
    config["model"]["params"]["first_stage_config"]["params"].pop("ckpt_path")

    model, pl_sd = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)


    for idx in range(opt.times):
        if opt.times > 1:   # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")

        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        start = time.time()
        metrics_dict, _ = model.log_dice(data=data, save_dir=outpath)
        print(f"Inference Speed: {len(data) /(time.time() - start)}")

        dice_list = metrics_dict["test_avg_dice"]
        iou_list = metrics_dict["test_avg_iou"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][direct]: {sum(dice_list) / len(dice_list)}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][direct]: {sum(iou_list) / len(iou_list)}\033[0m")

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")

    # print_model_stats(model)
    cal_params_flops(model, 256)


if __name__ == "__main__":
    main()
