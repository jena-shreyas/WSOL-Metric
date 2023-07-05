import os
import random
import subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from image_ops import load_and_resize, combine_image_and_heatmap
from similarity_ops import compute_spatial_similarity
from torchvision.io.image import read_image, ImageReadMode
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, resize
from torch.utils.data import Dataset

from src.dataset_norms import dataset_norms
from src.models import VisionTransformer
from xformer.visualize import visualize


# Set the seed value
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def sample_triplets(image_dir, num_triplets):

    '''Randomly sample triplets from a directory of images'''

    dirnames = os.listdir(image_dir)
    triplets = set()

    while len(triplets) < num_triplets:
        # Randomly sample an anchor image
        anchor_dirname = random.choice(dirnames)
        anchor_filename = random.choice(os.listdir(os.path.join(image_dir, anchor_dirname)))
        anchor_path = os.path.join(image_dir, anchor_dirname, anchor_filename)

        # Randomly sample a positive image (same class as anchor)
        positive_dirname = anchor_dirname
        positive_dir_files = os.listdir(os.path.join(image_dir, positive_dirname))
        positive_dir_files.remove(anchor_filename)
        positive_filename = random.choice(positive_dir_files)
        positive_path = os.path.join(image_dir, positive_dirname, positive_filename)

        # Randomly sample a negative image (different class from anchor)
        dirnames_ = dirnames.copy()
        dirnames_.remove(anchor_dirname)
        negative_dirname = random.choice(dirnames_)
        negative_filename = random.choice(os.listdir(os.path.join(image_dir, negative_dirname)))
        negative_path = os.path.join(image_dir, negative_dirname, negative_filename)

        triplets.add((anchor_path, positive_path, negative_path))

    triplets = list(triplets)
    return triplets


class ImageTripletDataset(Dataset):

    '''Dataset class for loading triplets of images'''

    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        anchor_path, positive_path, negative_path = self.triplets[index]

        anchor_image = read_image(anchor_path, mode=ImageReadMode.RGB) / 255.0
        positive_image = read_image(positive_path, mode=ImageReadMode.RGB) / 255.0
        negative_image = read_image(negative_path, mode=ImageReadMode.RGB) / 255.0

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.triplets)
    

# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # '''
        #  TODO: torch.dist computes frobenius norm, I want row-wise euclidean distance ! (CHANGE)
        # '''

        batch_size = anchor.shape[0]
        loss = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            distance_positive = torch.dist(anchor[i], positive[i], p=2)
            distance_negative = torch.dist(anchor[i], negative[i], p=2)
            loss[i] = torch.relu(distance_positive - distance_negative + self.margin)
        loss = torch.mean(loss)
        return loss
    

# Save model checkpoint
def checkpoint(model, epoch: int, model_path: str):

    print("Saving model checkpoint for epoch {}...".format(epoch))
    torch.save(model, model_path)


def create_class_dict(img_path: str):

    dir_list = os.listdir(img_path)
    class_dict = {}
    for dirname in dir_list:
        tokens = dirname.split(".")
        class_dict[tokens[1].lower()] = tokens[0]

    return class_dict

def create_annot_csv(annot_filename: str, dataset_path: str):

    imgs_path = dataset_path + "/images"
    imgfile_path = dataset_path + "/images.txt"
    bbox_path = dataset_path + "/bounding_boxes.txt"

    df = pd.DataFrame(columns=['img_id', 'img_path', 'img_name', 'gx', 'gy', 'gw', 'gh', 'correct'])

    with open(imgfile_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            l = line.split(' ')
            img_id, img_path = l
            df.loc[int(img_id)] = [int(img_id), imgs_path + "/" + img_path, img_path.split('/')[1], 0, 0, 0, 0, 0]

    with open(bbox_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            l = line.split(' ')
            img_id, gx, gy, gw, gh = l
            df.loc[int(img_id), ['gx', 'gy', 'gw', 'gh']] = [gx, gy, gw, gh]

    file_name = dataset_path + "/" + annot_filename
    df.to_csv(file_name, sep='\t', encoding='utf-8')


def get_output_features(model, out_tup):
    '''
        Function to get output features from model.
    '''
    if type(model) == VisionTransformer:
        output_feat, prepooled_tokens, attn_weights = out_tup
    else:
        # ResNet
        output_feat, prepooled_tokens = out_tup

    return output_feat

# function to compute the embeddings for each image in a input path using model defined above and save them in a output path
def compute_and_save_embeddings(model, dataset, transform, inp_path : str, out_path : str, device_ids: list):
    # create output directories
    os.makedirs(out_path + "/query", exist_ok=True)
    os.makedirs(out_path + "/gallery", exist_ok=True)

    query_path = inp_path + "/query"
    gallery_path = inp_path + "/gallery"

    # compute embeddings for query images
    query_files = os.listdir(query_path)
    for file in query_files:
        img_path = query_path + "/" + file
        input_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(f'cuda:{device_ids[0]}')
        out = get_output_features(model, model(input_tensor))
        print("Query ", out.shape)
        output_tensor = F.normalize(out, p=2, dim=1)
        torch.save(output_tensor, out_path + "/query/" + file[:-4] + ".pt")

    # compute embeddings for gallery images
    gallery_dirnames = os.listdir(gallery_path)
    for dirname in gallery_dirnames:
        os.makedirs(out_path + "/gallery" + "/" + dirname, exist_ok=True)
        gallery_files = os.listdir(gallery_path + "/" + dirname)
        
        for file in gallery_files:
            img_path = gallery_path + "/" + dirname + "/" + file
            input_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(f'cuda:{device_ids[0]}')
            out = get_output_features(model, model(input_tensor))
            print("Gallery ", out.shape)
            output_tensor = F.normalize(out, p=2, dim=1)
            torch.save(output_tensor, out_path + "/gallery/" + dirname + "/" + file[:-4] + ".pt")



def blackVis(model, model_type: str, dataset: str, img1_path: str, img2_path: str, vis_path: str, device_ids: list): 
    '''
        Separate definition for CUB dataset, with only query-heatmap overlay as output.
    '''

    #### PROLLY THIS CODE MIGHT NOT RUN NOW !!! ####
    device = 'gpu' if len(device_ids) > 0 else 'cpu'
    # subprocess.run(['python3', 'xformer/visualize.py', '--dataset', dataset, '--save_dir', vis_path, '--imageA', img1_path, '--imageB', img2_path, '--model_type', model_type, '--device', device])
    visualize(model, img1_path, img2_path, vis_path, device)


# function to retrieve the query embeddings, compute the cosine similarity with all the gallery embeddings, return the top 1 results and save whether top-1 class matches or not
def retrieve_visualize(model, model_type: str, dataset: str, img_path : str, emb_path : str, vis_path: str, class_dict: dict, csv_path: str, device_ids : list):

    # create output directories
    os.makedirs(vis_path + "/heatmaps", exist_ok=True)
    os.makedirs(vis_path + "/overlayed", exist_ok=True)

    query_path = emb_path + "/query"
    gallery_path = emb_path + "/gallery"

    # retrieve and visualize query images
    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
    
    query_files = os.listdir(query_path)
    for query_file in query_files:
        query_emb = torch.load(query_path + "/" + query_file).to(f'cuda:{device_ids[0]}')
        gallery_dirnames = os.listdir(gallery_path)
        max_sim = -1
        max_file_path = ""
        for dirname in gallery_dirnames:
            file_names = os.listdir(gallery_path + "/" + dirname)
            for file in file_names:
                gallery_emb = torch.load(gallery_path + "/" + dirname + "/" + file).to(f'cuda:{device_ids[0]}')
                sim = torch.cosine_similarity(query_emb, gallery_emb, dim=1)           # CHANGE !!!
                if sim > max_sim:
                    max_sim = sim
                    max_file_path = dirname + "/" + file

        correct = 0

        # save whether the retrieved image is of the correct class or not
        try:
            query_class = '_'.join(query_file[:-3].split('_')[:-2])
            query_class_id = int(class_dict[query_class.lower()])
            max_class = max_file_path.split('/')[0].split('.')[1]
            max_class_id = int(class_dict[max_class.lower()])

            
            if query_class_id == max_class_id:
                correct = 1

        except:
            pass

        query_imgname = query_file[:-3] + ".jpg"
        df.loc[df['img_name'] == query_imgname, 'correct'] = correct
        print("Query : {} | Top reference : {}".format(query_file, max_file_path))
        blackVis(model, model_type, dataset, img_path + "/query/" + query_file[:-3] + ".jpg", img_path + "/gallery/" + max_file_path[:-3] + ".jpg", vis_path, device_ids)
    
    df.to_csv(csv_path, sep='\t', encoding='utf-8')



def compute_bboxes_from_heatmaps(hmap_path : str, img_path: str, box_path: str, tau, area_frac):
    '''
        Function to compute bounding boxes from heatmaps, overlay them on images and store the bboxes and heatmaps.
    '''
    hmaps = os.listdir(hmap_path)
    for hmap_name in hmaps:
        bboxes = list()
        hmap = cv2.imread(hmap_path + "/" + hmap_name)
        hmap_cvt = cv2.cvtColor(hmap, cv2.COLOR_RGB2BGR)
        hmap_gray = cv2.cvtColor(hmap_cvt, cv2.COLOR_BGR2GRAY)

        hmap_thres = cv2.threshold(hmap_gray, tau * np.max(hmap_gray), 255, cv2.THRESH_BINARY_INV)[1]
        area_thres = area_frac * hmap_thres.shape[0] * hmap_thres.shape[1]

        cnts = cv2.findContours(hmap_thres.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        img_arr = load_and_resize(img_path + "/{}".format(hmap_name))
        
        for c in cnts:
            if cv2.contourArea(c) < area_thres: continue
            x,y,w,h = cv2.boundingRect(c)
            bboxes.append([x, y, w, h])
            img_arr = cv2.rectangle(img_arr, (x, y), (x + w, y + h), (10, 255, 34), 2)
            
        os.makedirs(box_path + "/overlayed", exist_ok=True)
        cv2.imwrite(box_path + "/overlayed/{}".format(hmap_name), img_arr)
        bboxes = np.array(bboxes)

        os.makedirs(box_path + "/boxes", exist_ok=True)
        bbox_filename = hmap_name[:-4] + ".npy"
        np.save(box_path + "/boxes/{}".format(bbox_filename), bboxes)



def compute_iou(bbox1, bbox2):
    '''
        Function to compute IoU between two bounding boxes.
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = w1 * h1
    boxBArea = w2 * h2

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



def resize_bboxes(bbox, w, h):
    '''
        Function to resize bounding boxes to original image size.
    '''
    x, y, bw, bh = bbox
    x = int(x / w * 224)
    y = int(y / h * 224)
    bw = int(bw / w * 224)
    bh = int(bh / h * 224)
    return [x, y, bw, bh]


def compute_box_acc(hmap_path : str, img_path: str, box_path: str, csv_path : str, tau, delta = 0.5, area_frac = 0.1, is_loc = False) -> float:
    '''
        Function to compute mean IoU between ground truth and predicted bounding boxes.
    '''

    # First, compute the bounding boxes from heatmaps
    compute_bboxes_from_heatmaps(hmap_path, img_path, box_path, tau, area_frac)

    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')

    bbox_path = box_path + "/boxes"
    bbox_names = os.listdir(bbox_path)
    mean_iou = 0

    for bbox_name in bbox_names:
        img_name = bbox_name[:-4] + ".jpg"
        gt_bbox_ = df[df['img_name'] == img_name][['gx', 'gy', 'gw', 'gh']].values[0]
        img = read_image(img_path + "/" + img_name, mode=ImageReadMode.RGB)
        _, h, w = img.shape
        gt_bbox = resize_bboxes(gt_bbox_, w, h)
        pred_bboxes = np.load(bbox_path + "/" + bbox_name)

        max_iou = 0

        for pred_bbox in pred_bboxes:
            if compute_iou(gt_bbox, pred_bbox) > max_iou: 
                max_iou = compute_iou(gt_bbox, pred_bbox)

        correct = df.loc[df['img_name'] == img_name]['correct'].values[0]
        if is_loc:
            max_iou = 1 if max_iou > delta and correct == 1 else 0
        else:
            max_iou = 1 if max_iou > delta else 0
        mean_iou += max_iou
    box_acc = mean_iou / len(bbox_names)
    # print("BoxAcc = {}".format(box_acc))
    return box_acc


def compute_max_box_acc(hmap_path : str, img_path: str, box_path: str, csv_path : str, delta = 0.5, area_frac = 0.1, is_loc = False, step = 0.05):
    max_tau, max_box_acc = 0, 0
    metric = "Top-1 Localization" if is_loc else "MaxBoxAcc"
    # print("Computing {} for delta = {} ...".format(metric, delta))

    for tau in np.arange(0, 1, step):
        box_acc = compute_box_acc(hmap_path, img_path, box_path, csv_path, tau, delta, area_frac, is_loc)
        # print("tau = {}, metric = {}".format(tau, box_acc))
        if box_acc > max_box_acc:
            max_box_acc = box_acc
            max_tau = tau
    print("\n{} = {} at tau = {}".format(metric, max_box_acc, max_tau))    
    # print("Re-computing results for tau = {} ... ".format(max_tau))
    compute_box_acc(hmap_path, img_path, box_path, csv_path, max_tau, delta, area_frac, is_loc)
    # print("Done.")

    return metric, max_box_acc
