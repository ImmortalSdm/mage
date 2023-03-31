import argparse
import math
import os
from PIL import Image
import numpy as np
import cv2
import torch

from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import models_mage
from util.datasets import *

imagenet_mean = torch.tensor([0.5, 0.5, 0.5])
imagenet_std = torch.tensor([0.5, 0.5, 0.5])

def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking


def gen_image(model, data, seed, num_iter=12, choice_temperature=4.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 64
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf

    B, C, H, W = data.shape
    sample_refer, sample_gt, refer, gt = data[...,:H//2,:H//2], data[...,:H//2,H//2:], data[...,H//2:,:H//2], data[...,H//2:,H//2:]
    refer = refer.cuda()
    gt = gt.cuda()
    sample_refer = sample_refer.cuda()
    sample_gt = sample_gt.cuda()

    refer_tokens = model.get_image_token(refer)
    B, N = refer_tokens.shape
    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(token_indices.shape[0], 8, 8, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)
    refer_emb = model.vqgan.quantize.get_codebook_entry(refer_tokens, (token_indices.shape[0],int(math.sqrt(N)),int(math.sqrt(N)),256))
    refer_img = model.vqgan.decode(refer_emb)

    gt_tokens = model.get_image_token(gt)
    gt_emb = model.vqgan.quantize.get_codebook_entry(gt_tokens, (token_indices.shape[0],int(math.sqrt(N)),int(math.sqrt(N)),256))
    gt_img = model.vqgan.decode(gt_emb)

    sample_refer_tokens = model.get_image_token(sample_refer)
    sample_refer_emb = model.vqgan.quantize.get_codebook_entry(sample_refer_tokens, (token_indices.shape[0],int(math.sqrt(N)),int(math.sqrt(N)),256))
    sample_refer_img = model.vqgan.decode(sample_refer_emb)

    sample_gt_tokens = model.get_image_token(sample_gt)
    sample_gt_emb = model.vqgan.quantize.get_codebook_entry(sample_gt_tokens, (token_indices.shape[0],int(math.sqrt(N)),int(math.sqrt(N)),256))
    sample_gt_img = model.vqgan.decode(sample_gt_emb)

    sample_pair = torch.cat((sample_refer_img,sample_gt_img),1)
    gt_pair = torch.cat((refer_img,gt_img),1)
    q_canvas = torch.cat((sample_pair,gt_pair),0)

    token_indices = torch.cat((
        sample_refer_tokens, 
        sample_gt_tokens, 
        refer_tokens), dim=-1)
    B, N = token_indices.shape
    initial_token_indices = mask_token_id * torch.ones(B, unknown_number_in_the_beginning).cuda()
    token_indices = torch.cat((token_indices, initial_token_indices), dim=-1)

    for step in range(num_iter):
        cur_ids = token_indices.clone().long()

        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        token_drop_mask = torch.zeros_like(token_indices)

        # token embedding
        input_embeddings = model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        # print(x.shape, token_drop_mask.shape, token_all_mask.shape)
        # decoder
        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]

        # get token prediction
        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()

        # get ids for next step
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter

        mask_ratio = np.cos(math.pi / 2. * ratio)

        # sample ids according to prediction confidence
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

        selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

        mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                 torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

        # Sample masking tokens for next iteration
        masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
        # Masks tokens with lower confidence.
        token_indices = torch.where(masking, mask_token_id, sampled_ids)
    # vqgan visualization
    sampled_ids = sampled_ids[:, N:]
    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(token_indices.shape[0], 8, 8, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)

    sample_pair = torch.cat((sample_refer_img,sample_gt_img),1)
    result_pair = torch.cat((refer_img,gen_images),1)
    r_canvas = torch.cat((sample_pair,result_pair),0)
    return q_canvas, r_canvas

def get_args_parser():
    parser = argparse.ArgumentParser('MAGE generation', add_help=False)
    parser.add_argument('-img', '--img_path', default='figures_dataset/original_999.png', type=str)
    parser.add_argument('-val', '--val_path', default='results/test_data', type=str)
    parser.add_argument('-list', '--data_list', default='/mnt/v-dsheng/data/ILSVRC_2012/imagenetcolor_train.txt', type=str)
    parser.add_argument('-type', '--data_type', default='test', type=str, choices=['CVF', 'inpaint', 'color', 'caption', 'test'])
    parser.add_argument('-size', '--input_size', default=256, type=int)

    parser.add_argument('--temp', default=4.5, type=float,
                        help='sampling temperature')
    parser.add_argument('--num_iter', default=12, type=int,
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=25, type=int,
                        help='batch size for generation')
    parser.add_argument('--ckpt', type=str,
                        help='checkpoint')
    parser.add_argument('--model', default='mage_vit_base_patch16', type=str,
                        help='model')

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    save_model_path = os.path.join('results', args.ckpt.split('/')[-2])
    save_path = os.path.join(save_model_path, args.val_path.split('/')[-1])
    print(save_model_path)
    print(save_path)
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    vqgan_ckpt_path = 'ckpts/vqgan_jax_strongaug.ckpt'

    model = models_mage.__dict__[args.model](norm_pix_loss=False,
                                            mask_ratio_mu=0.55, mask_ratio_std=0.25,
                                            mask_ratio_min=0.0, mask_ratio_max=1.0,
                                            vqgan_ckpt_path=vqgan_ckpt_path)
    model.to(0)

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transforms_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()])
    if args.data_type in ['CVF', 'test']:
        dataset_val = datasets.ImageFolder(args.val_path, transform=transforms_train)
    elif args.data_type == 'color':
        image_transform = transforms.Compose([
            transforms.CenterCrop((224 // 2, 224 // 2)),
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.CenterCrop((224 // 2, 224 // 2)),
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_val = DatasetColorization(args.data_list, args.val_path, image_transform, mask_transform, padding=0)
    elif args.data_type == 'caption':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=Image.BICUBIC),  # 384, 640
            transforms.ToTensor()])
        dataset_val = MyCOCOCaptionDataset('val', None, transform)
    else:
        dataset_val = MyInpaintingTrainDataset(args.val_path, transforms_train, args.samples_num)

    print(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    # dataset_val = glob(os.path.join(args.val_path, '*'))
    # for i in range(10):
    for i, batch in enumerate(tqdm(data_loader_val)):
        if args.data_type in ['CVF', 'test']:
            data = batch[0]
        else:
            data = batch

        with torch.no_grad():
            q_canvas, r_canvas = gen_image(model, data=data, seed=i, choice_temperature=args.temp, num_iter=args.num_iter)
        q_canvas, r_canvas = q_canvas.detach().cpu(), r_canvas.detach().cpu()

        # save img
        for b_id in range(args.batch_size):
            input_img = np.clip(q_canvas[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            input_img = input_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_path, 'input_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), input_img)

            gen_img = np.clip(r_canvas[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_path, 'output_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), gen_img)