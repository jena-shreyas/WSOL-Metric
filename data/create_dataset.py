# write a program where given a path to a directory, it will randomly sample 'n' images from each class
# One of each of n images per class will go to a 'query' directory and the rest will go to a 'gallery' directory

# import the necessary packages
import os
import random
import shutil
from pathlib import Path
import argparse

# define the function to create the dataset
def create_dataset(path, new_path, m, n):
    # create the new directories
    Path(new_path).mkdir(parents=True, exist_ok=True)
    Path(new_path + '/query').mkdir(parents=True, exist_ok=True)
    Path(new_path + '/gallery').mkdir(parents=True, exist_ok=True)

    # get the list of classes
    classes = os.listdir(path)

    # for each class, randomly sample n images and move them to the query directory
    for c in classes:
        # create the class directory in the new path
        
        Path(new_path + '/gallery/' + c).mkdir(parents=True, exist_ok=True)

        # get the list of images in the class directory
        images = os.listdir(path + '/' + c)

        # randomly sample n images
        random_images = random.sample(images, m+n)
        query_images = random_images[:m]
        gallery_images = random_images[m:]

        # move the randomly sampled images to the query directory
        for i in query_images:
            shutil.copy(path + '/' + c + '/' + i, new_path + '/query/')

        # move the rest of the images to the gallery directory
        for i in gallery_images:
            shutil.copy(path + '/' + c + '/' + i, new_path + '/gallery/' + c)



if __name__ == "__main__":

    #argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to the dataset")
    parser.add_argument("-n", "--new_path", required=True, help="path to the new dataset")
    parser.add_argument("-q", "--num_query", required=True, help="number of query images sampled from each class")
    parser.add_argument("-g", "--num_gallery", required=True, help="number of gallery images to be sampled from each class")
    args = parser.parse_args()

    # define the path to the dataset
    path = args.path

    # define the new path
    new_path = args.new_path

    # define the number of images to be sampled from each class
    m = int(args.num_query)

    # define the number of images to be sampled from each class
    n = int(args.num_gallery)

    # call the function to create the dataset
    create_dataset(path, new_path, m, n)
