# with open('reference_image_dgx.csv', 'w') as fw:
#     fw.write('query_id,reference_id,img_folder\n')
#     for i in range(1000000):
#         idx = str(i).zfill(6)
#         # fw.write(f'R{idx},R{idx},D:\\Driven_Data\\Facebook\\data\\reference_images\n')
#         fw.write(f'R{idx},R{idx},/tf/facebook/data/reference_images/\n')
#
# with open('query_image_dgx.csv', 'w') as fw:
#     fw.write('query_id,reference_id,img_folder\n')
#     for i in range(50000):
#         idx = str(i).zfill(5)
#         # fw.write(f'R{idx},R{idx},D:\\Driven_Data\\Facebook\\data\\reference_images\n')
#         fw.write(f'Q{idx},Q{idx},/tf/facebook/data/query_images/\n')

import pandas as pd
import numpy as np

is_dgx = False

pd_truth = pd.read_csv('public_ground_truth.csv')
pd_truth = pd_truth[pd_truth['reference_id'].notna()].reset_index()

query_id = pd_truth['query_id'].unique()
query_lb = np.arange(len(query_id))

reference_id = pd_truth['reference_id'].unique()
reference_lb = np.arange(len(reference_id))

train_image = np.concatenate((query_id, reference_id))
train_lb = np.concatenate((query_lb, reference_lb))

df_train = pd.DataFrame({"query_id":train_image, "label": train_lb})
if is_dgx:
    df_train['img_folder'] = df_train.apply(lambda x: '/tf/facebook/data/reference_images/' if x['query_id'].startswith('R') else '/tf/facebook/data/query_images/', axis=1)
    df_train.to_csv('train_truth_10k_dgx.csv', index=False)
else:
    df_train['img_folder'] = df_train.apply(lambda x: 'D:/Driven_Data/Facebook/data/reference_images/' if x['query_id'].startswith('R') else 'D:/Driven_Data/Facebook/data/query_images/', axis=1)
    df_train.to_csv('train_truth_10k.csv', index=False)