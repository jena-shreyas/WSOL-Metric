import torch
import torch.nn as nn
import argparse as ap
from collections import OrderedDict
from utils import sample_triplets, ImageTripletDataset, TripletLoss, checkpoint, create_class_dict, create_annot_csv, retrieve_visualize, compute_max_box_acc
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from collections import OrderedDict
from tqdm._tqdm_notebook import tqdm

# Set the seed value
torch.manual_seed(42)

# Define the command line arguments
parser = ap.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/CUB_200_2011')
parser.add_argument('--retr_path', type=str, default='dataset/img_retrieval_CUB_200_2011_ft')
parser.add_argument('--emb_path', type=str, default='embeddings_CUB_200_2011_ft')
parser.add_argument('--model_path', type=str, default='../models/resnet18_finetuned_best.pth')
parser.add_argument('--vis_path', type=str, default='visualizations_CUB_200_2011_ft')
parser.add_argument('--num_triplets', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--area_frac', type=float, default=0.1)
parser.add_argument('--is_loc', type=bool, default=False)
parser.add_argument('--step', type=float, default=0.05)


# Define the training loop
def train(model, dataloader, optimizer, lr_scheduler, num_epochs, loss_fn, device_ids):

    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        min_loss = 1e6

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(f'cuda:{device_ids[0]}')             # [3, 224, 224]
            positive = positive.to(f'cuda:{device_ids[0]}')
            negative = negative.to(f'cuda:{device_ids[0]}')

            optimizer.zero_grad()
            anchor_embedding = finetuned_model(anchor).flatten()       # [512, ]
            positive_embedding = finetuned_model(positive).flatten()
            negative_embedding = finetuned_model(negative).flatten()

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
            checkpoint(finetuned_model, epoch)
            min_loss = loss.item()   

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")



# Parse the arguments
args = parser.parse_args()

if __name__ == '__main__':

    # Load the pretrained ResNet18 model
    # Set the image directory and other parameters
    data_dir = args.data_dir
    image_dir = data_dir + '/images'
    num_triplets = args.num_triplets
    batch_size = args.batch_size

    # Sample image triplets
    triplets = sample_triplets(image_dir, num_triplets)

    # Define the data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the image triplet dataset
    dataset = ImageTripletDataset(triplets, transform=transform)

    # Create the data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # parse device string to get device ids
    device_ids = [int(s) for s in (args.device.split(':')[-1]).split(',')]

    # Load the pretrained ResNet18 model
    pretrained_model = resnet18(pretrained=True)

    # Create the finetuning model with the pretrained backbone
    finetuned_model = nn.Sequential(OrderedDict([*(list(pretrained_model.named_children())[:-1])]))
    finetuned_model = nn.DataParallel(finetuned_model, device_ids=device_ids)
    finetuned_model.to(f'cuda:{device_ids[0]}')

    # Define the triplet loss
    triplet_loss = TripletLoss()

    # Define the optimizer and learning rate scheduler
    optimizer = optim.AdamW(finetuned_model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = args.num_epochs

    # Train the model
    train(finetuned_model, dataloader, optimizer, lr_scheduler, num_epochs, triplet_loss, device_ids)

    # Create a dictionary mapping class names to indices
    class_dict = create_class_dict(image_dir)

    # Create the annotations CSV file
    create_annot_csv(data_dir)

    # Create a sub-model with the backbone and the average pooling layer for Stylianou approach
    stylianou_model = nn.Sequential(OrderedDict([*(list(finetuned_model.module.named_children())[:-1])]))
    stylianou_model = nn.DataParallel(stylianou_model, device_ids=device_ids)
    stylianou_model.to(f'cuda:{device_ids[0]}')

    retr_path = args.retr_path
    emb_path = args.emb_path
    vis_path = args.vis_path
    hmap_path = vis_path + "/heatmaps"
    csv_path = data_dir + "/annotations_ft.csv"
    retrieve_visualize(retr_path, emb_path, hmap_path, csv_path)  # Retrieve and visualize images

    # Compute the maximum box accuracy
    img_path = retr_path + "/query"
    box_path = vis_path + "/bboxes"
    csv_path = data_dir + "/annotations_ft.csv"   
    bbox_path = box_path + "/boxes"
    delta = args.delta
    area_frac = args.area_frac
    is_loc = args.is_loc
    step = args.step

    # Compute max box accuracy
    compute_max_box_acc(hmap_path, img_path, vis_path, csv_path, bbox_path, delta, area_frac, is_loc, step)