import numpy as np
import matplotlib.pyplot as plt

w = 10
h = 10
fig = plt.figure(figsize=(9, 13))
columns = 4
rows = 5

xs = np.linspace(0, 2*np.pi, 60)
ys = np.abs(np.sin(xs))
ax = []
for i in range(columns*rows):
    img = np.random.randint(10, size=(h,w))
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("ax:"+str(i))  # set title
    plt.imshow(img, alpha=0.25)

plt.show()

# import pandas as pd
# df = pd.read_csv('test_200.txt', header=None, delimiter=" ")
# df.columns = ['img_folder', 'label']
# df['query_id'] = df['img_folder'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
# df['img_folder'] = df['img_folder'].apply(lambda x: "/".join(x.split('/')[:-1]))
# print(df)

# a = "0001/0001/9640_CAM_FRONT_000007.jpg"
# a.replace("/")