import argparse
import torch
import numpy as np
import sys
import os,shutil
import dlib
sys.path.append(".")
sys.path.append("..")
from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
import cv2
import shutil
from editings.easy_edit import edit
import gradio as gr

opts=None
net=None
args=None

def init():
    global opts,net,args
    # Prepare and set-up model

    torch.nn.Module.dump_patches = True
    torch.multiprocessing.set_start_method('spawn')

    net, opts = setup_model(args)
    generator = net.decoder
    generator.eval()


def model_run(input_image,
        age,
        angle_horizontal,
        angle_pitch,
        beauty,
        emotion_angry,
        emotion_disgust,
        emotion_easy,
        emotion_fear,
        emotion_happy,
        emotion_sad,
        emotion_surprise,
        smile,
        eyes_open,
        face_shape,
        gender,
        glasses,
        race_black,
        race_white,
        race_yellow,
        height,
        width
        ):
    global opts,net,args
    
    # 创建临时图片保存的目录
    if os.path.exists(args.images_dir):
        shutil.rmtree(args.images_dir)
    os.makedirs(args.images_dir)

    # print(input_image)
    # 把图片复制到input里
    shutil.move(input_image,os.path.join(args.images_dir,os.path.basename(input_image)))

    args, data_loader = setup_data_loader(args, opts, parse_net=net.parser)

    is_cars = 'cars_' in opts.dataset_type

    generator = net.decoder
    generator.eval()

    # Get dlatents
    latents_file_path = os.path.join(args.images_dir, 'latents.pt')
    latent_codes = get_all_results(net, data_loader, args.n_sample, is_cars=is_cars, device=args.device)
    torch.save(latent_codes, latents_file_path)

    # Edit dlatents
    # latent_codes = edit(latent_codes, 'editings/latent_directions/'+editings+'.npy', strength=strength)
    for editing in [
                'age',
                'angle_horizontal',
                'angle_pitch',
                'beauty',
                'emotion_angry',
                'emotion_disgust','emotion_easy','emotion_fear','emotion_happy','emotion_sad',
                'emotion_surprise','eyes_open','face_shape','gender','glasses','height','race_black','race_white',
                'race_yellow','smile','width']:
        latent_codes = edit(latent_codes, 'editings/latent_directions/'+editing+'.npy', strength=eval(editing))
    

    # Generate new faces and merge back
    projected_images = generate_inversions(args, generator, latent_codes, is_cars=is_cars)
    
    result=projected_images[0]
    # print('projected_images',result)
    return result

# 在这里验证图片
def make_dataset(images_path,parse_net):
    images = []
    assert os.path.isdir(images_path), '%s is not a valid directory' % images_path
    for root, _, fnames in sorted(os.walk(images_path)):
        for fname in fnames:
            path = os.path.join(root, fname)
            is_image=None
            try:
                is_image=run_alignment(path,parse_net)
            except:
                # print("erro: "+path)
                is_image=None
            if is_image:
                images.append(path)
    return images


def setup_data_loader(args, opts, parse_net=None):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")

    # 在这里验证图片
    imgs=make_dataset(images_path,parse_net) 
    # print(imgs)                  

    test_dataset = InferenceDataset(root=images_path,
                                    paths=imgs,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=run_alignment,
                                    opts=opts,
                                    parse_net=parse_net)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_results(net, data_loader, n_images=None, is_cars=False, device='cuda', load_latents=None):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            img, x = batch
            if not load_latents:
                inputs = x.to(device).float()
                latents = get_latents(net, inputs, is_cars)
                all_latents.append(latents)
            i += 1
    if not load_latents:
        all_latents = torch.cat(all_latents)
    else:
        all_latents = torch.load(load_latents)
        all_latents = torch.cat(all_latents).to(device)
    return all_latents


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)
    return im_save_path

@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    aligned_images = []
    inversions_directory_path = args.images_dir
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(args.n_sample):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        im_save_path=save_image(imgs[0], inversions_directory_path, i + 1)
        aligned_images.append(im_save_path)
    return aligned_images

def run_alignment(image_path, parse_net):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    # print("to Aligned image: "+ image_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor, parse_net=parse_net)
    # print("Aligned image: " + image_path)
    return aligned_image



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--device", type=str, default='cuda', help="Inference on CPU or GPU")

    parser.add_argument("--images_dir", type=str, default='temp',
                        help="The directory of the images to be inverted")

    # parser.add_argument("--save_dir", type=str, default='temp_save',
    #                     help="The directory to save the latent codes and inversion images. (default: images_dir")

    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")

    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")

    parser.add_argument("--no_align", action="store_true", help="align face images before inference")

    parser.add_argument("--checkpoint_path_E", default="pretrained_models/encoder.pt", 
                            help="path to encoder checkpoint, optional: [encoder|encoder_without_pos].pt" )

    parser.add_argument("--checkpoint_path_D", default="pretrained_models/projector_EastAsian.pt", 
                            help="path to decoder checkpoint, optional: projector_[WestEuropean|EastAsian|NorthAfrican].pt")

    parser.add_argument("--checkpoint_path_P", default="pretrained_models/79999_iter.pth", help="path to parser checkpoint")

    # parser.add_argument("--editings",default="emotion_fear",help="path to editings")
    # parser.add_argument("--strength",default=6,help="editings strength")

    args = parser.parse_args()
    init()


    inputs=[
        # gr.inputs.Textbox(placeholder='请输入文字...',
            #                 default ='嗨，无界社区，你好~~',
            #                 label ='字幕',optional=False),
            gr.inputs.Image(shape=None, image_mode="RGB", invert_colors=False, source="upload", 
                            tool="editor", type="filepath", label=None, optional=False)
    ]
    for e in ['age',
                'angle_horizontal',
                'angle_pitch',
                'beauty',
                'emotion_angry',
                'emotion_disgust','emotion_easy','emotion_fear','emotion_happy','emotion_sad',
                'emotion_surprise','eyes_open','face_shape','gender','glasses','height','race_black','race_white',
                'race_yellow','smile','width']:

        inputs.append(
            gr.inputs.Slider(minimum=-100, maximum=100, step=1, default=1,label=e,optional=True)
        )


    ui = gr.Interface(
        article='人脸创作',
        theme='huggingface',
        allow_flagging ='auto',
        fn=model_run, 
        inputs=inputs,
        outputs=[gr.outputs.Image(type="auto")],
    )
    ui.launch()