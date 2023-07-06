import torch.nn as nn
import torchreid
from collections import OrderedDict
from code.Black_vis.src.xformer.src.visualization_functions import compute_spatial_similarity
# TODO : 
# set up this model for eval
# check its input dimensions, make it 512 so as to be compatible with Black et al. approach
# also, try to replace their predefined loss function with the one from Black (i.e, ArcFace, ProxyAnchor)
# test results, if doesn't work, try to integrate ViT into torchreid and try to train it on market1501

datamanager = torchreid.data.ImageDataManager(
    root="../data/dataset/",
    sources="market1501",
    targets="market1501",
    height=128,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=["random_flip", "random_crop"],
)

train_loader = datamanager.train_loader
test_loader = datamanager.test_loader

# # for x in train_loader:
# #     print(x['dsetid'])
# #     break

# # print(train_loader)
# # print(test_loader)

query_loader = test_loader['market1501']['query']
gallery_loader = test_loader['market1501']['gallery']

model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1501, loss='triplet', pretrained=True)
print(model.global_avgpool)

# for name, module in model.named_children():
#     print(name)

# print(model.conv5)

submodel = nn.Sequential(OrderedDict([*(list(model.named_children())[:-3])]))

for name, module in submodel.named_children():
    print(name)

for batch in train_loader:
    img = batch['img'][0].unsqueeze(0)
    print(img.shape)

    out = submodel(img).squeeze(0)
    print(out.size())

    avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    out_avg = avgpool(out)
    print(out_avg.size())

    out = out.view(-1, out.size(0)).cpu().detach().numpy()
    print(out.shape)

    out_avg = out_avg.view(out_avg.size(0)).cpu().detach().numpy()
    print(out_avg.shape)

    break

