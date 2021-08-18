import pandas as pd
import torch
from tqdm import tqdm
import joblib

from conf import *
from utils import *
from data import FBDataset
from models import Net
from loss import *

DEVICE = torch.device("cuda:3")

# Build model:
model = Net(args)
model.to(DEVICE)
model.eval()
# print(model)

# Load dataset:
# df_train = pd.read_csv('df_train.csv')
# data_train = FBDataset(df_train)
# a = data_train.__getitem__(0)
#
# np_zeros = np.zeros((2, 3, 440, 440))
# np_zeros[0] = a['input']
# np_zeros[1] = a['input']
# np_zeros = torch.from_numpy(np_zeros.astype(np.float32))
# print(np_zeros.shape)
#
# output = model(np_zeros, get_embeddings=True)
# print(output['embeddings'])
# # print(output['logits'])

# Load dataset:
batch_size = 2
df_train = pd.read_csv('reference_image.csv')
data_train = FBDataset(df_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False)

tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
np_refer = np.zeros((1000000, 512))

for i, batch in tk0:
    i_start = i*batch_size
    i_end = (i+1)*batch_size
    print(batch['input'].shape)
    inputs = batch['input'].to(DEVICE)
    output = model(inputs, get_embeddings=True)
    np_refer[i_start: i_end] = output['embeddings'].detach().numpy()
    if i == 3:
        break
# print(np_refer[:6])

# joblib.dump(np_refer, 'np_refer.joblib')