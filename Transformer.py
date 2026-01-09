# train_transformer_full.py
# 需求: torch, torchaudio, numpy, matplotlib
# pip install torch torchaudio numpy matplotlib

import os
import glob
import random
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt

import torchaudio
from scipy.io import wavfile
# 優先使用soundfile backend(避免需要torchcodec)
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    try:
        torchaudio.set_audio_backend("sox_io")
    except Exception:
        print("Warning: no suitable torchaudio backend available. Install SoundFile or torchcodec.")
# ---------------------------
# CONFIG
# ---------------------------
CFG = {
    "train_dir": "BabyCryDataset/train",
    "val_dir": "BabyCryDataset/val",
    "test_dir": "BabyCryDataset/test",
    "sample_rate": 16000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "max_audio_seconds": 4.0,     # 音檔會被裁/補到此長度
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "Models/transformer_model_001.pth",
    "plot_path": "Plots/training_curve01.png",
    # Transformer hyperparams
    "d_input": 64,    # = n_mels
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CFG["plot_path"]), exist_ok=True)

# ---------------------------
# Utility: detect classes
# ---------------------------
def detect_classes_from_dir(train_dir: str) -> List[str]:
    p = Path(train_dir)
    classes = [d.name for d in p.iterdir() if d.is_dir()]
    classes = sorted(classes)
    if len(classes) == 0:
        raise ValueError(f"No class folders found in {train_dir}")
    return classes

# ---------------------------
# Dataset (固定長度 + Mel)
# ---------------------------
class BabyCryDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], cfg=CFG, augment=False):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.cfg = cfg
        self.augment = augment

        self.sr = cfg["sample_rate"]
        self.max_len = int(self.sr * cfg["max_audio_seconds"])  # samples

        self.files = []
        for idx, cname in enumerate(classes):
            folder = self.root_dir / cname
            if not folder.exists():
                continue
            for p in glob.glob(str(folder / "*.wav")):
                self.files.append((p, idx))
        if len(self.files) == 0:
            raise ValueError(f"No wav files found under {root_dir} for classes {classes}")

        self.mel = MelSpectrogram(sample_rate=self.sr, n_fft=cfg["n_fft"],
                                  hop_length=cfg["hop_length"], n_mels=cfg["n_mels"])
        self.atdb = AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def _load_and_fix(self, path):
        # use scipy to read wav to avoid torchaudio->torchcodec path
        sr, data = wavfile.read(path)
        # convert to float32 tensor in range [-1, 1] if integer
        if isinstance(data, np.ndarray):
            wav = torch.tensor(data, dtype=torch.float32)
        else:
            wav = torch.tensor([data], dtype=torch.float32)
        if wav.dim() > 1:
            # average channels to mono
            wav = wav.mean(dim=-1)
        # ensure shape (1, L)
        wav = wav.unsqueeze(0)
        # normalize integer PCM to [-1,1] if necessary
        if wav.dtype == torch.float32 and wav.abs().max() > 1.0:
            # assume int16 or int32
            wav = wav / (2 ** 15)
        # resample if needed
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # pad or crop
        if wav.shape[1] > self.max_len:
            if self.augment:
                start = random.randint(0, wav.shape[1] - self.max_len)
            else:
                start = 0
            wav = wav[:, start:start + self.max_len]
        else:
            pad = self.max_len - wav.shape[1]
            wav = nn.functional.pad(wav, (0, pad))
        return wav  # (1, max_len)

    def _spec_augment(self, mel):
        if not self.augment:
            return mel
        # simple spec augment (freq & time mask)
        n_mels, t = mel.size()
        # freq mask
        f_param = max(1, int(n_mels * 0.12))
        f = random.randint(0, f_param)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0+f, :] = 0
        # time mask
        t_param = max(1, int(t * 0.12))
        tt = random.randint(0, t_param)
        t0 = random.randint(0, max(0, t - tt))
        mel[:, t0:t0+tt] = 0
        return mel

    def __getitem__(self, idx):
        path, label = self.files[idx]
        wav = self._load_and_fix(path)  # (1, L)
        mel = self.mel(wav)             # (1, n_mels, T)
        mel = mel.squeeze(0)            # (n_mels, T)
        mel = self.atdb(mel)            # dB
        # per-sample normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = self._spec_augment(mel)
        # Transformer expects (T, C) -> we will return (T, C)
        mel = mel.permute(1, 0)  # (T, n_mels)
        return mel.float(), label

# ---------------------------
# Transformer model
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, C)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class BabyCryTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        # project input dim -> d_model (here we keep d_model = input_dim for simplicity)
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.pos_enc = PositionalEncoding(input_dim, max_len= int(CFG["max_audio_seconds"] * (CFG["sample_rate"] / CFG["hop_length"]) + 10))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x)       # (B, T, d)
        x = self.pos_enc(x)          # (B, T, d)
        x = self.encoder(x)          # (B, T, d)
        x = x.mean(dim=1)            # global avg pool over time -> (B, d)
        x = self.norm(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Training / Eval utilities
# ---------------------------
def compute_class_weights_from_dirs(train_dir: str, classes: List[str]):
    counts = []
    for c in classes:
        files = list(Path(train_dir).joinpath(c).glob("*.wav"))
        counts.append(len(files))
    counts = np.array(counts) + 1e-6
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        # x: (B, T, C) ; y: (B,)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    train_dir = CFG["train_dir"]
    val_dir = CFG["val_dir"]
    test_dir = CFG["test_dir"]

    classes = detect_classes_from_dir(train_dir)
    num_classes = len(classes)
    print("Detected classes:", classes)

    # datasets
    train_ds = BabyCryDataset(train_dir, classes, cfg=CFG, augment=True)
    val_ds = BabyCryDataset(val_dir, classes, cfg=CFG, augment=False)
    test_ds = BabyCryDataset(test_dir, classes, cfg=CFG, augment=False)

    # dataloaders (use num_workers=0 to avoid torchaudio backend issues in worker processes)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

    device = CFG["device"]
    model = BabyCryTransformer(input_dim=CFG["n_mels"],
                               num_classes=num_classes,
                               nhead=CFG["nhead"],
                               num_layers=CFG["num_layers"],
                               dim_feedforward=CFG["dim_feedforward"],
                               dropout=CFG["dropout"]).to(device)

    # optional: class weights if dataset imbalance
    class_weights = compute_class_weights_from_dirs(train_dir, classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    final_train_acc = 0.0
    final_val_acc = 0.0

    print("Start training on device:", device)
    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        final_train_acc = train_acc
        final_val_acc = val_acc

        print("=" * 60)
        print(f"Epoch {epoch}/{CFG['epochs']}")
        print(f"  Train -> loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   -> loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print("=" * 60)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "classes": classes
            }, CFG["save_path"])
            print(f"✔ Saved best model (val_acc={val_acc:.4f}) to {CFG['save_path']}")

    print("\n==================== Training Finished ====================")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Train Accuracy:     {final_train_acc:.4f}")
    print(f"Final Val Accuracy:       {final_val_acc:.4f}")

    # -----------------------------------------
    # 最後→ 在 test set 額外評估並獨立顯示
    # -----------------------------------------
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print("\n==================== Test Set Evaluation ====================")
    print(f"Test Set Loss:     {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print("==============================================================")

    # ========= Plot training curves =========
    plt.figure(figsize=(10, 6))
    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    # Acc subplot
    plt.subplot(2, 1, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG["plot_path"], dpi=300)
    plt.show()
    print(f"Training curve saved -> {CFG['plot_path']}")