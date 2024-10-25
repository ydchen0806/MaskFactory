import cv2
import os
from PIL import Image
from diffusers.utils import load_image
from stable_diffusion_multi_controlnet import *
from diffusers.utils import load_image

def apply_canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def get_canny(ori_mask_path):
    ori_mask=Image.open(ori_mask_path)
    im=cv2.imread(ori_mask_path)
    canny_im=apply_canny(im,30,230)
    return canny_im,ori_mask


class Multicontrol_model():
    def __init__(self):
        self.pipe = StableDiffusionMultiControlNetPipeline.from_pretrained(
            "/h3cstore_ns/ydchen/code/DatasetDM/DIS_neurips24/data_generation/data_generation/mc/mgmx", safety_checker=None, torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        self.controlnet_canny = ControlNetModel.from_pretrained("/h3cstore_ns/ydchen/code/DatasetDM/DIS_neurips24/data_generation/data_generation/mc/sd-controlnet-canny", torch_dtype=torch.float16).to(
            "cuda"
        )
        self.controlnet_seg = ControlNetModel.from_pretrained(
            "/h3cstore_ns/ydchen/code/DatasetDM/DIS_neurips24/data_generation/data_generation/mc/sd-controlnet-seg", torch_dtype=torch.float16
        ).to("cuda")
    
    def generate(self,mask_path,prompt_path,width=512,height=512,steps=80):
        canny_im,ori_mask=get_canny(mask_path)
        pil_canny=Image.fromarray(canny_im)
        # width, height = 5600,4000
        if isinstance(prompt_path, str):  # 如果 prompt_path 是字符串类型
            # 尝试作为文件路径打开
            try:
                with open(prompt_path, 'r') as f:
                    pt = f.read()
            except FileNotFoundError:
                # 如果无法打开文件，将 prompt_path 视为文本内容
                pt = prompt_path
        elif isinstance(prompt_path, list):  # 如果 prompt_path 是列表类型
            pt = "\n".join(prompt_path)
        else:  # 其他情况，默认将 prompt_path 视为文件路径
            with open(prompt_path, 'r') as f:
                pt = f.read()
        image = self.pipe(
            prompt=pt,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            processors=[
                ControlNetProcessor(self.controlnet_canny, load_image(pil_canny)),
                ControlNetProcessor(self.controlnet_seg, load_image(ori_mask)),
            ],
            generator=torch.Generator(device="cuda").manual_seed(0),
            num_inference_steps=steps,
            width=width,
            height=height,
        ).images[0]
        final_im = image
        # final_im=image.resize((np.array(ori_mask).shape[1],np.array(ori_mask).shape[0]))
        return final_im
    
def ddp_inference(rank, world_size, mask_list, prompt_list=None, mode = 0):
    model = Multicontrol_model()
    distributed_data_list = mask_list[rank::world_size]
    os.makedirs(f'/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0424_mode_{mode}/with_masactrl_controlnet_gen_img',exist_ok=True)
    os.makedirs(f'/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0424_mode_{mode}/without_masactrl_controlnet_gen_img',exist_ok=True)
    os.makedirs(f'/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0424_mode_{mode}/with_masactrl_concat_controlnet_gen_img',exist_ok=True)
    os.makedirs(f'/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0424_mode_{mode}/without_masactrl_concat_controlnet_gen_img',exist_ok=True)

    if prompt_list is None:
        for mask_path in distributed_data_list:
            basename = os.path.basename(mask_path)
            if mode == 0:
                prompt = f'{basename.split("#")[1]}, {basename.split("#")[3]}'
            elif mode == 1:
                prompt = f'{basename.split("#")[1]}'
            elif mode == 2:
                prompt = f'{basename.split("#")[3]}'
            b = model.generate(mask_path, prompt)
            if 'without_masactrl' in mask_path:
                save_dir = mask_path.replace('without_masactrl', 'without_masactrl_controlnet_gen_img').replace('0326',f'0424_mode_{mode}')
                # concat_save_dir = mask_path.replace('without_masactrl', 'concat_controlnet_gen_img')
                concat_save_dir = mask_path.replace('without_masactrl', 'without_masactrl_concat_controlnet_gen_img').replace('0326',f'0424_mode_{mode}')
            elif 'with_masactrl' in mask_path:
                save_dir = mask_path.replace('with_masactrl', 'with_masactrl_controlnet_gen_img').replace('0326',f'0424_mode_{mode}')
                # concat_save_dir = mask_path.replace('with_masactrl', 'concat_controlnet_gen_img')
                concat_save_dir = mask_path.replace('with_masactrl', 'with_masactrl_concat_controlnet_gen_img').replace('0326',f'0424_mode_{mode}')

            concat_img = Image.new('RGB', (b.size[0]*2, b.size[1]))
            concat_img.paste(b, (0, 0))
            concat_img.paste(Image.open(mask_path), (b.size[0], 0))
            # concat_save_dir = mask_path.replace('without_masactrl', 'concat_controlnet_gen_img')
            concat_img.save(concat_save_dir)
            b.save(save_dir)
            print(basename)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, default='/h3cstore_ns/DatasetDM/weights/SD1.4')
    parser.add_argument("--data_list", type=str, default='/h3cstore_ns/jcxie/seq/DIS5K/DIS*/gt/*.png')
    parser.add_argument("--out_dir_base", type=str, default="/h3cstore_ns/ydchen/mask_editting_test")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--filter_value", type=int, default=3)
    parser.add_argument("--mode", type=int, default=0)
    return parser.parse_args()


if __name__=="__main__":
    from glob import glob
    mask_list = []
    mask_list = glob('/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0326/with_masactrl/*png')
    mask_list += glob('/h3cstore_ns/ydchen/mask_edit/mask_editting3090_binary_0326/without_masactrl/*png')
    # sort list
    mask_list.sort()
    args = get_args()
    ddp_inference(args.local_rank, args.world_size, mask_list, mode = args.mode)
