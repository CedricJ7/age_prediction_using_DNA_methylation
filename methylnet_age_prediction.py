import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Paramètres Globaux
# ==========================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DATA_DIR = Path('Data')
TOP_K = 1000 # Nombre de CpGs à conserver
CHUNK_SIZE = 5000

print('Imports OK - PyTorch version:', torch.__version__)


# ==========================================
# 1. Chargement des Données
# ==========================================
def load_data():
    ind = pd.read_csv(DATA_DIR / 'annot_projet.csv')
    ind = ind.dropna(subset=['age', 'Sample_description']).copy()
    ind['Sample_description'] = ind['Sample_description'].astype(str)
    ind = ind.set_index('Sample_description')

    raw_path = DATA_DIR / 'c_sample.csv'
    sample_header = pd.read_csv(raw_path, nrows=0)
    all_sample_ids = list(sample_header.columns)
    common_ids = [s for s in all_sample_ids if s in ind.index]
    y = ind.loc[common_ids, 'age'].values.astype(np.float32)
    print(f'Échantillons communs trouvés : {len(common_ids)}')

    print(f'\nCalcul des corrélations sur données brutes pour sélectionner les top {TOP_K} CpGs...')
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
    
    return X_raw, y


# ==========================================
# 2. Architecture de type MethylNet
# ==========================================

class MethylationDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Module 1 : L'Autoencodeur pour la réduction de dimension non supervisée
class MethylNetAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(MethylNetAutoEncoder, self).__init__()
        
        # Encodage
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Décodage
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim),
            nn.Sigmoid() # Assumes input data is min-max scaled between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# Module 2 : Le Modèle de Régression basé sur l'Encodeur
class MethylNetRegressor(nn.Module):
    def __init__(self, encoder, latent_dim=256):
        super(MethylNetRegressor, self).__init__()
        self.encoder = encoder
        
        # Le réseau de prédiction de l'âge (MLP) attaché à l'encodeur
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # On passe à travers l'encodeur
        latent = self.encoder(x)
        # Puis à travers le régresseur
        age_pred = self.regressor(latent)
        return age_pred.squeeze(1)


# ==========================================
# 3. Entraînement complet du Modèle
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Appareil pour l'entraînement : {device}")

    # Chargement
    X_raw, y = load_data()
    
    # Paramètres d'entraînement
    N_FOLDS = 5
    PRETRAIN_EPOCHS = 50
    TRAIN_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LATENT_DIM = 256

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae, cv_r2 = [], []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_raw)):
        print(f"\n{'='*20} Fold {fold_i + 1}/{N_FOLDS} {'='*20}")
        
        X_train_raw, y_tr = X_raw[train_idx], y[train_idx]
        X_test_raw, y_te = X_raw[test_idx], y[test_idx]
        
        print("  -> Imputation des données manquantes en cours...")
        t0 = time()
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=RANDOM_STATE,
            skip_complete=True
        )
        X_train_imp = imputer.fit_transform(X_train_raw).astype(np.float32)
        X_test_imp = imputer.transform(X_test_raw).astype(np.float32)
        
        # Scaling simple pour le Sigmoid de l'AutoEncodeur (0 to 1 scaling array-wide)
        x_min, x_max = X_train_imp.min(), X_train_imp.max()
        X_train_imp = (X_train_imp - x_min) / (x_max - x_min + 1e-8)
        X_test_imp = (X_test_imp - x_min) / (x_max - x_min + 1e-8)
        print(f"  -> Imputation terminée en {time()-t0:.1f}s")
        
        # DataLoaders
        train_loader = DataLoader(MethylationDataset(X_train_imp, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(MethylationDataset(X_test_imp, y_te), batch_size=BATCH_SIZE, shuffle=False)
        
        # --- Étape A : Pré-entraînement de l'Autoencodeur ---
        print("\n  [Phase 1] Pré-entraînement de l'Autoencodeur (Non supervisé)")
        autoencoder = MethylNetAutoEncoder(input_dim=TOP_K, latent_dim=LATENT_DIM).to(device)
        ae_criterion = nn.MSELoss()
        ae_optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(PRETRAIN_EPOCHS):
            autoencoder.train()
            total_ae_loss = 0
            for batch_X, _ in train_loader:
                batch_X = batch_X.to(device)
                ae_optimizer.zero_grad()
                _, reconstructed = autoencoder(batch_X)
                loss = ae_criterion(reconstructed, batch_X)
                loss.backward()
                ae_optimizer.step()
                total_ae_loss += loss.item()
            if (epoch + 1) % 25 == 0:
                print(f"    AE Epoch {epoch+1}/{PRETRAIN_EPOCHS} - Loss: {total_ae_loss/len(train_loader):.4f}")

        # --- Étape B : Entraînement du Régresseur ---
        print("\n  [Phase 2] Entraînement au Régresseur d'Âge supervisé")
        # On extrait la partie encodeur de l'autoencodeur
        encoder_network = autoencoder.encoder
        methylnet_model = MethylNetRegressor(encoder=encoder_network, latent_dim=LATENT_DIM).to(device)
        
        reg_criterion = nn.L1Loss() # MAE Loss
        reg_optimizer = optim.Adam(methylnet_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        for epoch in range(TRAIN_EPOCHS):
            methylnet_model.train()
            total_reg_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                reg_optimizer.zero_grad()
                preds = methylnet_model(batch_X)
                loss = reg_criterion(preds, batch_y)
                loss.backward()
                reg_optimizer.step()
                total_reg_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"    Reg Epoch {epoch+1}/{TRAIN_EPOCHS} - L1 Loss (MAE): {total_reg_loss/len(train_loader):.4f}")
                
        # --- Étape C : Évaluation du Split ---
        methylnet_model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                preds = methylnet_model(batch_X.to(device))
                all_preds.extend(preds.cpu().numpy())
                
        mae = mean_absolute_error(y_te, all_preds)
        r2 = r2_score(y_te, all_preds)
        cv_mae.append(mae)
        cv_r2.append(r2)
        print(f"\n  -> ** Résultat du Fold {fold_i + 1} - MAE: {mae:.3f} | R²: {r2:.3f} **")

    print(f"\n{'-'*50}")
    print(f"Bilan Global PyTorch MethylNet (AE + MLP)")
    print(f"Global MAE: {np.mean(cv_mae):.3f} +/- {np.std(cv_mae):.3f}")
    print(f"Global R²: {np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}")

if __name__ == "__main__":
    main()
