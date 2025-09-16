import os
import numpy as np
from torchvggish import vggish, vggish_input
import torch

splits = ['train', 'val', 'test']
base_dir = r'C:\Users\USER\OneDrive\桌面\專題\BabyCryDataset'

model = vggish()
model.eval()

for split in splits:
    input_root = os.path.join(base_dir, split)
    features = []
    labels = []
    for label in os.listdir(input_root):
        label_path = os.path.join(input_root, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith('.wav'):
                continue
            audio_path = os.path.join(label_path, fname)
            print(f"處理中: {audio_path}")
            try:
                examples = vggish_input.wavfile_to_examples(audio_path)
                with torch.no_grad():
                    emb = model(examples)
                feature = emb.mean(dim=0).numpy()
                if feature.shape == (128,):
                    features.append(feature)
                    labels.append(label)
                else:
                    print(f"特徵維度錯誤，已跳過：{audio_path}，shape={feature.shape}")
            except Exception as e:
                print(f"檔案處理失敗，已跳過：{audio_path}，錯誤訊息：{e}")
    features = np.array(features)
    labels = np.array(labels)
    np.save(fr'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_features_{split}.npy', features)
    np.save(fr'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_labels_{split}.npy', labels)
    print(f"{split} 特徵儲存完成，共 {len(labels)} 筆")