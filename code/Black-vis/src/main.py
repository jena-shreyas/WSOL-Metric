import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse as ap
from collections import OrderedDict
from utils import sample_triplets, ImageTripletDataset, TripletLoss, checkpoint, compute_and_save_embeddings, create_class_dict, create_annot_csv, retrieve_visualize, compute_max_box_acc
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, resnet18, resnet101, resnet152
from src.models import VisionTransformer, ResNetV2
from collections import OrderedDict
from tqdm.notebook import tqdm
import time
import logging

from src.dataset_norms import dataset_norms

# Set the seed value
torch.manual_seed(42)

# Define the command line arguments
parser = ap.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/CUB_200_2011')
parser.add_argument('--retr_path', type=str, default='../data/img_retrieval_CUB_200_2011')
parser.add_argument('--emb_path', type=str, default='../embeddings/CUB_200_2011')   ## CHANGE !!!
parser.add_argument('--model_type', type=str, default='ViT-B16', help='"ViT-B{patch_size}" or "resnet-101"')
parser.add_argument('--model_path', type=str, default='../models/resnet50_finetuned_best.pth')  ## CHANGE !!!
parser.add_argument('--dataset', type=str, default='SOP', help='Choices: Hotels-50k, SOP, GLM (Datasets on which the model is trained)')
parser.add_argument('--vis_path', type=str, default='../visualizations/CUB_200_2011')  ## CHANGE !!!
parser.add_argument('--num_triplets', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0,1')
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--area_frac', type=float, default=0.1)
parser.add_argument('--is_loc', type=bool, default=False)
parser.add_argument('--step', type=float, default=0.05)
parser.add_argument('--log_path', type=str, default='logs/lr.txt')
parser.add_argument('--arg', type=str, default='lr')
parser.add_argument('--is_frozen', type=bool, default=False)


def create_logger(log_path, args):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the log level
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    with open(log_path, 'a') as f:
        if args.arg == 'lr':
            logger.debug(f"\n\n##### LEARNING RATE : {args.lr} ######\n")

        elif args.arg == 'bs':
            logger.debug(f"\n\n##### BATCH SIZE : {args.batch_size} ######\n")

        elif args.arg == 'num_epochs':
            logger.debug(f"\n\n##### NUMBER OF EPOCHS : {args.num_epochs} ######\n")

        elif args.arg == 'num_triplets':
            logger.debug(f"\n\n##### NUMBER OF TRIPLETS : {args.num_triplets} ######\n")

    return logger



# Define the training loop
def train(model, model_path, dataloader, optimizer, lr_scheduler, num_epochs, loss_fn, device_ids):

    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        min_loss = 1e6

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(f'cuda:{device_ids[0]}')             # [3, 224, 224]
            positive = positive.to(f'cuda:{device_ids[0]}')
            negative = negative.to(f'cuda:{device_ids[0]}')

            optimizer.zero_grad()
            anchor_embedding = F.normalize(model(anchor), p=2, dim=1)
            positive_embedding = F.normalize(model(positive), p=2, dim=1)
            negative_embedding = F.normalize(model(negative), p=2, dim=1) 

            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            '''
                # In case the previous cell to freeze model params is run, 
                the next line must be uncommented to solve the error : 
                RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

                TODO: 
                On freezing model params for layers 6, 7 (both blocks of conv, batchnorm layers), 8 (Avg Pool), 
                    the loss doesn't decrease at all!
            '''
            # loss.requires_grad = True #### (Only when freezing some model params)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if loss.item() < min_loss and epoch > 0:
            checkpoint(model, epoch, model_path)
            min_loss = loss.item()   

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")



# Parse the arguments
args = parser.parse_args()

if __name__ == '__main__':

    start = time.time()

    # Load the pretrained ResNet18 model
    # Set the image directory and other parameters
    data_dir = args.data_dir
    dataset = args.dataset
    image_dir = data_dir + '/images'
    # num_triplets = args.num_triplets
    # batch_size = args.batch_size
    model_type = args.model_type
    # model_path = args.model_path
    # log_path = args.log_path
    # is_frozen = args.is_frozen

    # logger = create_logger(log_path, args)

    # parse device string to get device ids
    device_ids = [int(s) for s in (args.device.split(':')[-1]).split(',')]

    model_weights = os.path.join('weights', dataset, f'{args.model_type}.pth')

    # Load the pretrained ResNet50 model
    model = None

    if 'vit' in args.model_type.lower():
        patch_size = int(args.model_type[-2:]) # Extract patch_size from model weights file
        config = {'output_size': 512, 'init_head': True, 'classifier': 'token', 'hidden_size': 768, 'img_size': 256,
                  'patch_size': (patch_size, patch_size), 'load_from': None, 'dropout_rate': 0, 'vis': True,
                  'num_layers': 12, 'mlp_dim': 3072, 'num_heads': 12, 'global_feature_embedding': 'mean',
                  'attention_dropout_rate': 0}
        model = VisionTransformer(config=config)

    elif args.model_type == 'resnet-101':
        config = {'arch': 'r101', 'width_factor': 1, 'output_size': 512, 'zero_head': False, 'weights_file': None}
        model = ResNetV2(config=config)

    else:
        raise ValueError(f"Invalid model_type argument: '{args.model_type}' "
                         "model_type argument must be either ViT-B/{32|16} or resnet-101")
    

    # Load the pretrained model weights
    weights = torch.load(model_weights)
    weights = {k.replace('module.', ''): weights[k] for k in weights.keys()}
    model.load_state_dict(weights)
    model.to(f'cuda:{device_ids[0]}')

    # dims = 512 # Dimensionality of the embedding space

    # # Create the finetuning model with the pretrained backbone
    # print("Putting model on GPUs {}...".format(device_ids))
    # finetuned_model = nn.Sequential(OrderedDict([*(list(pretrained_model.named_children())[:-1])]))
    # finetuned_model.add_module('flatten', nn.Flatten())  # Flatten the output of the last convolutional layer
    # finetuned_model.add_module('fc', nn.Linear(512, dims))  # Add the fully connected layer for retraining
    # print(finetuned_model.fc)
    # finetuned_model = nn.DataParallel(finetuned_model, device_ids=device_ids)
    # finetuned_model.to(f'cuda:{device_ids[0]}')

    # print("Finetuning the model...")
    # if is_frozen:
    #     # Freeze the model parameters
    #     for name, param in finetuned_model.module.named_parameters():
    #         if 'fc' not in name:
    #             param.requires_grad = False

    #     print("Model parameters frozen!")

    # # Sample image triplets
    # print("Sampling image triplets...")
    # triplets = sample_triplets(image_dir, num_triplets)

    # # Define the data transforms
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # # Create the image triplet dataset
    # dataset = ImageTripletDataset(triplets, transform=transform)

    # # Create the data loader
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # Define the triplet loss
    # triplet_loss = TripletLoss()

    # # Define the optimizer and learning rate scheduler
    # optimizer = optim.AdamW(finetuned_model.parameters(), lr=args.lr)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # # Training loop
    # num_epochs = args.num_epochs


    # # Train the model
    # print("Training the model...")
    # tr_start = time.time()
    # train(finetuned_model, model_path, dataloader, optimizer, lr_scheduler, num_epochs, triplet_loss, device_ids)
    # tr_end = time.time()
    # print(f"Training time: {tr_end - tr_start} seconds")

    # Set model to eval mode
    model.eval()

    # define image transforms
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_norms[dataset]['mean'], dataset_norms[dataset]['std'])
    ])

    # Compute and save embeddings
    retr_path = args.retr_path
    emb_path = args.emb_path + "/" + model_type
    # print("Computing and saving embeddings...")
    # emb_start = time.time()
    # compute_and_save_embeddings(model, dataset, transform, retr_path, emb_path, device_ids)
    # emb_end = time.time()
    # print(f"Embedding time: {emb_end - emb_start} seconds")

    # Create a dictionary mapping class names to indices
    print("Creating the class dictionary...")
    class_dict = create_class_dict(image_dir)

    # # Create the annotations CSV file
    # print("Creating the annotations CSV file...")
    # annot_start = time.time()
    annot_filename = 'annotations.csv'
    # create_annot_csv(annot_filename, data_dir)
    # annot_end = time.time()
    # print(f"Annotation time: {annot_end - annot_start} seconds")

    # # Create a sub-model with the backbone and the average pooling layer for Stylianou approach
    # stylianou_model = nn.Sequential(OrderedDict([*(list(finetuned_model.module.named_children())[:-3])]))       # -3 bcoz last 3 layers are avgpool, flatten, fc
    # stylianou_model = nn.DataParallel(stylianou_model, device_ids=device_ids)
    # stylianou_model.to(f'cuda:{device_ids[0]}')

    # Retrieve and visualize images
    vis_path = args.vis_path + "/" + model_type
    hmap_path = vis_path + "/heatmaps"
    csv_path = data_dir + "/" + annot_filename
    # print("Retrieving and visualizing images...")
    # retr_start = time.time()
    # retrieve_visualize(model, model_type, dataset, retr_path, emb_path, vis_path, class_dict, csv_path, device_ids)  # Retrieve and visualize images
    # retr_end = time.time()
    # print(f"Retrieval and visualization time: {retr_end - retr_start} seconds")

    # Compute the maximum box accuracy
    img_path = retr_path + "/query"
    box_path = vis_path + "/bboxes"
    delta = args.delta
    area_frac = args.area_frac
    is_loc = args.is_loc
    step = args.step

    # Compute max box accuracy
    print("Computing metrics ...")
    metric, acc = compute_max_box_acc(hmap_path, img_path, box_path, csv_path, delta, area_frac, is_loc, step)

    # # Log the metric and accuracy
    # with open(log_path, 'a') as f:
    #     logger.debug(f"{metric} : {acc}")
    # print(f"{metric} : {acc}")

    end = time.time()
    print(f"\nTotal time: {end - start} seconds")

