import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Paramètres Globaux ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DATA_DIR = Path('Data')
TOP_K = 500 # 3.274 avec TOP_K = 500
CHUNK_SIZE = 5000

print('Imports OK - PyTorch version:', torch.__version__)

# ==========================================
# 1. Chargement des données
# ==========================================
# --- Charger annotations ---
ind = pd.read_csv(DATA_DIR / 'annot_projet.csv')
ind = ind.dropna(subset=['age', 'Sample_description']).copy()
ind['Sample_description'] = ind['Sample_description'].astype(str)
ind = ind.set_index('Sample_description')

raw_path = DATA_DIR / 'c_sample.csv'
sample_header = pd.read_csv(raw_path, nrows=0)
all_sample_ids = list(sample_header.columns)
common_ids = [s for s in all_sample_ids if s in ind.index]
y = ind.loc[common_ids, 'age'].values.astype(np.float32)
print(f'Échantillons communs : {len(common_ids)}')

# --- Corrélations NaN-tolerant par chunks ---
print(f'\nCalcul des corrélations sur données brutes...')
y_centered = y - y.mean()
y_den = np.sqrt(np.sum(y_centered ** 2))

all_corrs = []
for chunk in pd.read_csv(raw_path, usecols=common_ids, chunksize=CHUNK_SIZE):
    x = chunk.to_numpy(dtype=np.float32)
    if np.isnan(x).any():
        row_means = np.nanmean(x, axis=1, keepdims=True)
        row_means = np.where(np.isnan(row_means), 0, row_means)
        x = np.where(np.isnan(x), row_means, x)
    x_c = x - x.mean(axis=1, keepdims=True)
    num = x_c @ y_centered
    den = np.sqrt(np.sum(x_c ** 2, axis=1)) * y_den
    corr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    all_corrs.append(np.abs(corr))

all_corrs = np.concatenate(all_corrs)
top_k_indices = np.argsort(all_corrs)[::-1][:TOP_K]

indices_to_load = np.sort(top_k_indices)
rows = []
start = 0
for chunk in pd.read_csv(raw_path, usecols=common_ids, chunksize=CHUNK_SIZE):
    end = start + len(chunk)
    pos_s = np.searchsorted(indices_to_load, start)
    pos_e = np.searchsorted(indices_to_load, end)
    local = indices_to_load[pos_s:pos_e] - start
    if len(local) > 0:
        rows.append(chunk.iloc[local].values)
    start = end

X_raw = np.vstack(rows).T.astype(np.float32)
load_order = {idx: pos for pos, idx in enumerate(indices_to_load)}
reorder = np.array([load_order[i] for i in top_k_indices])
X_raw = X_raw[:, reorder]

print(f'\nShape initial X_raw (patients, CpGs): {X_raw.shape}')
print(f'Nombre de NaN total : {np.isnan(X_raw).sum()}')

# ==========================================
# 2. Fonction de Transformation en Images 2D
# ==========================================
IMG_H, IMG_W = 23, 22
FEATURES_TOTAL = IMG_H * IMG_W

def transform_to_images(X):
    """
    Transforme un vecteur de méthylation en image 2D (batch_size, 1, H, W)
    En ajoutant un padding avec des zéros si nécessaire.
    """
    pad_size = FEATURES_TOTAL - X.shape[1]
    X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
    return X_padded.reshape(-1, 1, IMG_H, IMG_W)

print(f"Dimension de l'image préparée : {IMG_H}x{IMG_W} (Total features requis: {FEATURES_TOTAL})")

# ==========================================
# 3. Architecture du Modèle CNN et PyTorch
# ==========================================
class CpGImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CpG_CNN(nn.Module):
    def __init__(self):
        super(CpG_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.conv_layers(x)
        age_pred = self.regression_head(features)
        return age_pred.squeeze(1)

# ==========================================
# 4. Entraînement 5-Folds
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Appareil : {device}")

    N_FOLDS = 5
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae, cv_r2 = [], []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_raw)):
        print(f"\nFold {fold_i + 1}/{N_FOLDS}")
        
        # Sélection des données brutes pour le split (avec NaN)
        X_train_raw, y_tr = X_raw[train_idx], y[train_idx]
        X_test_raw, y_te = X_raw[test_idx], y[test_idx]
        
        # Imputation ciblée au fold
        print("  -> Imputation des données (fit sur train, transform sur train & test)...")
        t0 = time()
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=RANDOM_STATE,
            skip_complete=True
        )
        X_train_imp = imputer.fit_transform(X_train_raw).astype(np.float32)
        X_test_imp = imputer.transform(X_test_raw).astype(np.float32)
        print(f"  -> Imputation terminée en {time()-t0:.1f}s")
        
        # Validation par transformation en images
        X_tr_img = transform_to_images(X_train_imp)
        X_te_img = transform_to_images(X_test_imp)
        
        # DataLoaders
        train_loader = DataLoader(CpGImageDataset(X_tr_img, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(CpGImageDataset(X_te_img, y_te), batch_size=BATCH_SIZE, shuffle=False)
        
        # Modèle, Loss et Optimiseur
        model = CpG_CNN().to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Modération avec weight decay
        
        # Boucle d'entrainement du Fold
        for epoch in range(EPOCHS):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
        # Évaluation du Fold
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                preds = model(batch_X.to(device))
                all_preds.extend(preds.cpu().numpy())
                
        mae = mean_absolute_error(y_te, all_preds)
        r2 = r2_score(y_te, all_preds)
        cv_mae.append(mae)
        cv_r2.append(r2)
        print(f"  -> Résultat du Fold - MAE: {mae:.3f} | R²: {r2:.3f}")

    print(f"\n{'-'*40}")
    print(f"Bilan Global PyTorch CNN (MICE par Fold)")
    print(f"Global MAE: {np.mean(cv_mae):.3f} +/- {np.std(cv_mae):.3f}")
    print(f"Global R²: {np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}")

if __name__ == "__main__":
    main()
