import argparse

import clip
import torch
from tqdm import tqdm

from classes import imagenet_classes
from data_loader import data_loader
from save_predictions import save_to_file
from templates import imagenet_templates


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main(args):
    model, preprocess = clip.load("ViT-B/32")
    model.to(device())
    softmax = torch.nn.Softmax(dim=1)
    loader = data_loader(preprocess, args)
    model.eval()
    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)
    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(tqdm(loader)):
            images = images.to(device())

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            logits = softmax(logits)
            save_to_file(logits, targets, paths)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default=None, type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=20, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=2048, type=int,
                      help='Batch size (default: 64)')

    config = args.parse_args()
    main(config)
