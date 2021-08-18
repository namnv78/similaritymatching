with open('reference_image_dgx.csv', 'w') as fw:
    fw.write('query_id,reference_id,img_folder\n')
    for i in range(1000000):
        idx = str(i).zfill(6)
        # fw.write(f'R{idx},R{idx},D:\\Driven_Data\\Facebook\\data\\reference_images\n')
        fw.write(f'R{idx},R{idx},/tf/facebook/data/reference_images/\n')

with open('query_image_dgx.csv', 'w') as fw:
    fw.write('query_id,reference_id,img_folder\n')
    for i in range(50000):
        idx = str(i).zfill(5)
        # fw.write(f'R{idx},R{idx},D:\\Driven_Data\\Facebook\\data\\reference_images\n')
        fw.write(f'Q{idx},Q{idx},/tf/facebook/data/query_images/\n')
