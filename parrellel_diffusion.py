import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from glob import glob
import argparse
# Custom imports
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Lambda
from pytorch_lightning import seed_everything
from glob import glob
from tqdm import tqdm
from diffusers import DDIMScheduler
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl
from torch.multiprocessing import spawn
from torchvision.io import read_image
from PIL import Image

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_and_prepare_image(image_path, device):
    # Adjust this function as necessary to accommodate your dataset
    image = read_image(image_path).float() / 255.0  # Normalize to [0, 1]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB
    # Resize and center crop as necessary
    return image.to(device)

def initialize_model(device):
    model_path = '/h3cstore_ns/ydchen/code/DatasetDM/weights/SD1.4'
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    return model

def load_image(image_path, device):
    image = read_image(image_path)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def create_gaussian_kernel(kernel_size=5, sigma=1):
    """
    创建一个高斯核。
    
    Args:
        kernel_size: 高斯核的大小，它应该是奇数。
        sigma: 高斯核的标准差。
    
    Returns:
        高斯核。
    """
    # 生成x, y坐标的网格
    x = torch.arange(kernel_size)
    x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    
    # 计算高斯核
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # 归一化高斯核
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # 增加批次和通道维度，以满足conv2d的要求
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return gaussian_kernel

def gaussian_blur(image, kernel_size=5, sigma=1):
    """
    对图像应用高斯模糊。
    
    Args:
        image: 输入图像，形状为[C, H, W]。
        kernel_size: 高斯核的大小。
        sigma: 高斯核的标准差。
    
    Returns:
        高斯模糊后的图像。
    """
    channels = image.shape[0]
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    # 重复核以匹配图像的通道数
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    # 应用高斯模糊，使用分组卷积对每个通道独立处理
    padding = kernel_size // 2
    blurred_image = F.conv2d(image.unsqueeze(0), gaussian_kernel, padding=padding, groups=channels)
    return blurred_image.squeeze(0)

def to_black_and_white_with_threshold_and_blur(image, threshold=0.5, kernel_size=5, sigma=1):
    """
    将图像转换为基于阈值的黑白图，并应用高斯模糊平滑图像。
    
    Args:
        image: 输入图像，形状为[C, H, W]，像素值范围为[0, 1]。
        threshold: 决定像素颜色的阈值。
        kernel_size: 高斯核的大小。
        sigma: 高斯核的标准差。
    
    Returns:
        转换并平滑后的图像。
    """
    # 首先对图像进行高斯模糊处理
    blurred_image = gaussian_blur(image, kernel_size, sigma)
    
    # 然后将模糊后的图像转换为黑白
    grayscale = blurred_image.max(dim=0, keepdim=True)[0]
    black_image = torch.zeros_like(blurred_image)
    white_image = torch.ones_like(blurred_image)
    filtered_image = torch.where(grayscale <= threshold, black_image, white_image)
    
    return filtered_image

def to_black_and_white(image, threshold=0.3):
    """
    将图像转换为基于阈值的黑白图。
    浅于阈值的像素变成白色，深于阈值的变成黑色。
    
    Args:
        image: 输入图像，形状为[C, H, W]，像素值范围为[0, 1]。
        threshold: 决定像素颜色的阈值。
        
    Returns:
        转换后的图像。
    """
    # 计算图像的灰度值，简化为取最大通道值
    grayscale = image.max(dim=0, keepdim=True)[0]
    # 创建黑色和白色图像
    black_image = torch.zeros_like(image)
    white_image = torch.ones_like(image)
    # 根据阈值将像素设置为黑色或白色
    filtered_image = torch.where(grayscale <= threshold, black_image, white_image)
    return filtered_image

def binary_image(image, threshold=28):
    # Convert the image to grayscale
    if isinstance(image, np.ndarray):
        image = image.squeeze()
        image = Image.fromarray(image)
    if isinstance(image, torch.Tensor):
        # 确保tensor的值在0-1范围内
        image = torch.clamp(image, 0, 1)

        # 将tensor的值范围从[0, 1]扩展到[0, 255]并转换为uint8
        image = (image * 255).to(torch.uint8)

        # 去掉批次的维度，得到3x512x512
        image = image.squeeze()

        # 转置tensor以适应PIL的格式，即512x512x3
        image = image.permute(1, 2, 0)

        # 转换为PIL Image
        image = Image.fromarray(image.cpu().numpy(), 'RGB')
    gray_image = image.convert('L')
    # Convert the grayscale image to a binary image
    threshold = 25  # This threshold can be adjusted if necessary
    binary_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')
    return binary_image

def ddp_inference(rank, world_size, model_path, data_list, out_dir_base,filter_value=0):
    # setup(rank, world_size)
    device = torch.device(f"cuda")
    model = initialize_model(device)
    # model = DDP(model, device_ids=[rank])

    transforms = Compose([
        Lambda(lambda image_path: load_image(image_path, device))
    ])

    # Make sure to distribute your data_list equally among GPUs
    distributed_data_list = data_list[rank::world_size]
    for source_image_path in tqdm(distributed_data_list):
        # sample_count = len(os.listdir(out_dir_base))
        out_dir = os.path.join(out_dir_base, source_image_path.split("/")[-2])
        source_out_dir = os.path.join(out_dir, "source")
        os.makedirs(source_out_dir, exist_ok=True)
        reconstructed_out_dir = os.path.join(out_dir, "reconstructed")
        os.makedirs(reconstructed_out_dir, exist_ok=True)
        without_masactrl_out_dir = os.path.join(out_dir, "without_masactrl")
        os.makedirs(without_masactrl_out_dir, exist_ok=True)
        with_masactrl_out_dir = os.path.join(out_dir, "with_masactrl")
        os.makedirs(with_masactrl_out_dir, exist_ok=True)
        concat_image_out_dir = os.path.join(out_dir, "concat_image")
        os.makedirs(concat_image_out_dir, exist_ok=True)
        # binary1_image_out_dir = os.path.join(out_dir, "binary_image_1")
        # os.makedirs(binary1_image_out_dir, exist_ok=True)
        # binary2_image_out_dir = os.path.join(out_dir, "binary_image_2")
        # os.makedirs(binary2_image_out_dir, exist_ok=True)
        # os.makedirs(out_dir, exist_ok=True)

        source_image = transforms(source_image_path)
        source_prompt = ""
        name = os.path.basename(source_image_path).split(".")[0]
        class_name = name.split("#")[3]
        target_prompt = f"a {class_name}"
        prompts = [source_prompt, target_prompt]

        with torch.no_grad():
            start_code, latents_list = model.invert(source_image, source_prompt, guidance_scale=7.5, num_inference_steps=50, return_intermediates=True)
            start_code = start_code.expand(len(prompts), -1, -1, -1)

            editor = AttentionBase()
            regiter_attention_editor_diffusers(model, editor)
            image_fixed = model([target_prompt], latents=start_code[-1:], num_inference_steps=50, guidance_scale=7.5)

            STEP = 4
            LAYPER = 10
            editor = MutualSelfAttentionControl(STEP, LAYPER)
            regiter_attention_editor_diffusers(model, editor)
            image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5)
            if filter_value == 0:
                save_image(source_image * 0.5 + 0.5, os.path.join(source_out_dir, f"{name}.png"))
                save_image(image_masactrl[0:1], os.path.join(reconstructed_out_dir, f"{name}.png"))
                save_image(image_fixed, os.path.join(without_masactrl_out_dir, f"{name}.png"))
                save_image(image_masactrl[-1:], os.path.join(with_masactrl_out_dir, f"{name}.png"))
                concat_image = torch.cat([source_image, image_masactrl[0:1], image_fixed, image_masactrl[-1:]], dim=3)
                save_image(concat_image * 0.5 + 0.5, os.path.join(concat_image_out_dir, f"{name}.png"))
            elif filter_value == 1:
                save_image(to_black_and_white(source_image * 0.5 + 0.5), os.path.join(source_out_dir, f"{name}.png"))
                save_image(to_black_and_white(image_masactrl[0:1]), os.path.join(reconstructed_out_dir, f"{name}.png"))
                save_image(to_black_and_white(image_fixed), os.path.join(without_masactrl_out_dir, f"{name}.png"))
                save_image(to_black_and_white(image_masactrl[-1:]), os.path.join(with_masactrl_out_dir, f"{name}.png"))
            elif filter_value == 2:
                save_image(to_black_and_white_with_threshold_and_blur(source_image * 0.5 + 0.5), os.path.join(source_out_dir, f"{name}.png"))
                save_image(to_black_and_white_with_threshold_and_blur(image_masactrl[0:1]), os.path.join(reconstructed_out_dir, f"{name}.png"))
                save_image(to_black_and_white_with_threshold_and_blur(image_fixed), os.path.join(without_masactrl_out_dir, f"{name}.png"))
                save_image(to_black_and_white_with_threshold_and_blur(image_masactrl[-1:]), os.path.join(with_masactrl_out_dir, f"{name}.png"))
            elif filter_value == 3:
                # save_image(binary_image(source_image * 0.5 + 0.5), os.path.join(source_out_dir, f"{name}.png"))
                # save_image(binary_image(image_masactrl[0:1]), os.path.join(reconstructed_out_dir, f"{name}.png"))
                # save_image(binary_image(image_fixed), os.path.join(without_masactrl_out_dir, f"{name}.png"))
                # save_image(binary_image(image_masactrl[-1:]), os.path.join(with_masactrl_out_dir, f"{name}.png"))
                # concat_image = torch.cat([source_image, image_masactrl[0:1], image_fixed, image_masactrl[-1:]], dim=3)
                # save_image(concat_image * 0.5 + 0.5, os.path.join(concat_image_out_dir, f"{name}.png"))
                binary_image(source_image * 0.5 + 0.5).save(os.path.join(source_out_dir, f"{name}.png"))
                binary_image(image_masactrl[0:1]).save(os.path.join(reconstructed_out_dir, f"{name}.png"))
                binary_image(image_fixed).save(os.path.join(without_masactrl_out_dir, f"{name}.png"))
                binary_image(image_masactrl[-1:]).save(os.path.join(with_masactrl_out_dir, f"{name}.png"))
                concat_image = torch.cat([source_image * 0.5 + 0.5, image_masactrl[0:1], image_fixed, image_masactrl[-1:]], dim=3)
                binary_image(concat_image).save(os.path.join(concat_image_out_dir, f"{name}.png"))
            else:
                raise ValueError("filter_value should be 0, 1 or 2")

            # save_image(source_image * 0.5 + 0.5, os.path.join(out_dir, f"{name}_source.png"))
            # save_image(image_masactrl[0:1], os.path.join(out_dir, f"{name}_reconstructed_source.png"))
            # save_image(image_fixed, os.path.join(out_dir, f"{name}_without_masactrl.png"))
            # save_image(image_masactrl[-1:], os.path.join(out_dir, f"{name}_with_masactrl.png"))

            print(f"Synthesized images for {name} are saved in {out_dir}")

    # cleanup()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, default='/h3cstore_ns/DatasetDM/weights/SD1.4')
    parser.add_argument("--data_list", type=str, default='/h3cstore_ns/ydchen/mask_edit/hr/*/*')
    parser.add_argument("--out_dir_base", type=str, default="/h3cstore_ns/ydchen/mask_editting_test")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--filter_value", type=int, default=3)
    return parser.parse_args()

def main():
    # world_size = 8
    # model_path = '/h3cstore_ns/DatasetDM/weights/SD1.4'
    # data_list = glob('/h3cstore_ns/DatasetDM/hr/lr/*.png')
    # out_dir_base = "/h3cstore_ns/ydchen/mask_editting"
    args = get_args()
    world_size = args.world_size
    model_path = args.model_path
    data_list = glob(args.data_list)
    print(f'total len of the generate img is {len(data_list)}')
    out_dir_base = args.out_dir_base
    my_rank = args.local_rank
    

    os.makedirs(out_dir_base, exist_ok=True)
    ddp_inference(my_rank, world_size, model_path, data_list, out_dir_base,filter_value=args.filter_value)

    # spawn(ddp_inference, args=(world_size, model_path, data_list, out_dir_base), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
    # sample command: python3 -m torch.distributed.launch --nproc_per_node=8 /data/ydchen/VLP/MasaCtrl/parrellel_diffusion.py

