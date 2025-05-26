import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_metadata(csv_path):
    import pandas as pd
    meta = pd.read_csv(csv_path)
    return meta

class ESC50Dataset(Dataset):
    def __init__(self, meta, audio_dir, sr=22050, n_fft=1024, hop_length=512, n_mels=128):
        self.meta = meta
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        y, sr = librosa.load(file_path, sr=self.sr)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        tensor = torch.FloatTensor(log_mel[np.newaxis, ...])
        label = torch.LongTensor([row['target']])[0]
        return tensor, label


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ResNet-like CNN
def build_resnet(block, layers, num_classes=50):
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.in_channels = 64
            self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNet()


if __name__ == '__main__':
    meta_csv = 'ESC-50-master/meta/esc50.csv'
    audio_dir = 'ESC-50-master/audio'
    meta = load_metadata(meta_csv)
    train_meta, val_meta = train_test_split(meta, test_size=0.2, stratify=meta['target'], random_state=42)
    # Datasets
    train_ds = ESC50Dataset(train_meta, audio_dir)
    val_ds = ESC50Dataset(val_meta, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = build_resnet(ResidualBlock, [2,2,2,2], num_classes=50).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    epochs = 30
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
        acc = correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best validation accuracy: {best_acc:.4f}')
