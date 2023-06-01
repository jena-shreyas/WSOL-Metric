import numpy as np
from numpy import matlib as mb # matlib must be imported separately

def compute_spatial_similarity(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    # K = 7
    # if conv1 = (49, 512)
    pool1 = np.mean(conv1,axis=0) # 1, 512
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0]))) # (sqrt(49), sqrt(49)) = (7, 7)
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0] # conv1 / norm{beta_1} / K^2 = (49, 512)
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0] # conv2 / norm{beta_2} / K^2 = (49, 512)
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0])) # (49, 49)
    for zz in range(conv1_normed.shape[0]): # range(49)
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1) # repmat((1, 512), 49, 1) = (49, 512) (512 length vector corresponding to each pixel position is repeated 49 times to form a 49x512 matrix)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1) # (1, 49) = (49, 512) * (49, 512) = (49, 512).sum(axis=1) = (49, 1)
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz) # 7, 7 
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    return similarity1, similarity2
