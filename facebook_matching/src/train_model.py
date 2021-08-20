import pandas as pd
# import torch
from tqdm import tqdm
# import joblib
from torch.optim import AdamW
# from torch import nn

# from conf import *
from utils import *
from data import FBDataset
from models import Net
from loss import ArcFaceLoss

is_dgx = False
use_data_re_id = True

# if is_dgx:
#     # Load dataset:
#     df_train = pd.read_csv('train_truth_10k_dgx.csv')
#     DEVICE = torch.device("cuda:3")
#     batch_size = 12
#     num_epoch = 20
# else:
#     # Load dataset:
#     df_train = pd.read_csv('train_truth_10k.csv')
#     DEVICE = torch.device("cpu")
#     batch_size = 2
#     num_epoch = 2

if use_data_re_id:
    # Config image folder
    root_image_folder = ""

    DEVICE = torch.device("cuda:3")
    batch_size = 12
    num_epoch = 20

    df_train = pd.read_csv('train_1K.txt', header=None, delimiter=" ")
    df_valid = pd.read_csv('val_200.txt', header=None, delimiter=" ")
    df_test = pd.read_csv('test_200.txt', header=None, delimiter=" ")

    df_train.columns = ['img_folder', 'label']
    df_train['query_id'] = df_train['img_folder'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    df_train['img_folder'] = df_train['img_folder'].apply(lambda x: root_image_folder + "/" + "/".join(x.split('/')[:-1]))

    df_valid.columns = ['img_folder', 'label']
    df_valid['query_id'] = df_valid['img_folder'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    df_valid['img_folder'] = df_valid['img_folder'].apply(lambda x: root_image_folder + "/" + "/".join(x.split('/')[:-1]))

    df_test.columns = ['img_folder', 'label']
    df_test['query_id'] = df_test['img_folder'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    df_test['img_folder'] = df_test['img_folder'].apply(lambda x: root_image_folder + "/" + "/".join(x.split('/')[:-1]))

data_train = FBDataset(df_train, normalization=args.normalization, aug=args.tr_aug)
data_valid = FBDataset(df_valid, normalization=args.normalization, aug=args.val_aug)
data_test = FBDataset(df_test, normalization=args.normalization, aug=args.val_aug)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)


args.n_classes = df_train['label'].nunique()
# Training model:
if args.class_weights == "log":
    val_counts = df_train['label'].value_counts().sort_index().values
    class_weights = 1 / np.log1p(val_counts)
    class_weights = (class_weights / class_weights.sum()) * args.n_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
else:
    class_weights = None

metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit="bce", weight=None, reduction="sum")

# Build model:
model = Net(args)
model.to(DEVICE)
model.train()

# Optimizer
BASE_LR = 0.001
optimizer = AdamW(model.parameters(), lr=BASE_LR)

# for epoch:
for epoch in range(num_epoch):
    _epoch = epoch + 1
    print(f"Start training epoch: {_epoch}")
    running_loss = 0
    tk_train = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in tk_train:
        optimizer.zero_grad()

        inputs = batch['input'].to(DEVICE)
        targets = batch['target'].to(DEVICE)

        output = model(inputs, get_embeddings=False)
        loss = metric_crit(output['logits'], targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('\nEpoch: %d loss: %.3f' %
          (epoch + 1, running_loss / (len(data_train))))
    torch.save(model.state_dict(), f'{args.backbone}_epoch{_epoch}.pth')
    # compute train accuracy:
    print("\nBegin compute accuracy on valid images: ")
    correct = 0
    total = 0
    with torch.no_grad():
        batch_preds = []
        batch_labels = []

        tk_valid = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, batch in tk_valid:
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs['logits'], 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            # AUC:
            batch_preds.append(predicted.detach().cpu().numpy())
            batch_labels.append(targets.detach().cpu().numpy())
    predictions = np.concatenate(batch_preds)
    labels = np.concatenate(batch_labels)
    auc = roc_auc_score(y_true=labels, y_score=predictions, multi_class='ovo', average='macro')

    print('\nPrecission of valid images: %d %%' % (correct / total))
    print(f'AUC of valid images:{auc}')