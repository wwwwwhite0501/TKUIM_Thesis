import os
import shutil
import random

def move_random_files(src_folder, dst_folder, num_files):
    os.makedirs(dst_folder, exist_ok=True)
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    if num_files > len(files):
        num_files = len(files)
    selected = random.sample(files, num_files)
    for f in selected:
        shutil.move(os.path.join(src_folder, f), os.path.join(dst_folder, f))
    print(f"已隨機剪下 {len(selected)} 個檔案從 {src_folder} 到 {dst_folder}")

src = r'C:\Users\USER\OneDrive\桌面\專題\BabyCryDataset3371\xxxxx'
dst = r'C:\Users\USER\OneDrive\桌面\專題\BabyCryDataset\train val test\xxxxx'
move_random_files(src, dst, xx)  #隨機剪下??個檔案