import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ================== Preprocessing ==================

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < data.shape[-1]]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped)
    pca_components = 30 if dataset_name != 'indian' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    return data_pca, gt, h, w, pca_components

# ================== Patch Extraction ==================

def extract_patches(data, gt, patch_size):
    h, w, _ = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')
    patches, labels, coords = [], [], []
    for i in range(margin, margin + h):
        for j in range(margin, margin + w):
            patch = padded_data[i - margin:i + margin, j - margin:j + margin, :]
            label = padded_gt[i, j]
            if label != 0:
                patches.append(patch)
                labels.append(label)
                coords.append((i - margin, j - margin))
    return np.array(patches), np.array(labels), np.array(coords), h, w

# ================== Autoencoder ==================

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class SimpleTransformer(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        if z.shape[-1] != self.attn.embed_dim:
            raise ValueError("Latent dim mismatch with transformer input.")
        attn_out, _ = self.attn(z, z, z)
        squeezed = attn_out.squeeze(1)
        if squeezed.shape[-1] != self.linear[0].in_features:
            raise ValueError("Transformer output mismatch with linear layer.")
        scores = self.linear(squeezed).squeeze()
        return scores

# ================== Pipeline for uploaded files ==================

def run_pipeline_with_files(hsi_path, gt_path, dataset_name, patch_size=16, latent_dim=32, num_epochs=10):
    # Load data from uploaded .mat files
    data = sio.loadmat(hsi_path)
    gt = sio.loadmat(gt_path)

    # Debug prints to check keys and types
    print(f"HSI file keys: {list(data.keys())}")
    print(f"GT file keys: {list(gt.keys())}")

    # Extract the first variable in each mat file (excluding metadata keys)
    data_keys = [key for key in data.keys() if not key.startswith('__')]
    gt_keys = [key for key in gt.keys() if not key.startswith('__')]

    if not data_keys or not gt_keys:
        raise ValueError("No valid variable keys found in .mat files")

    data_array = data[data_keys[0]]
    gt_array = gt[gt_keys[0]]

    print(f"Data array type: {type(data_array)}, shape: {getattr(data_array, 'shape', 'N/A')}")
    print(f"GT array type: {type(gt_array)}, shape: {getattr(gt_array, 'shape', 'N/A')}")

    # Preprocess
    data_pca, gt_processed, h, w, pca_dim = preprocess(data_array, gt_array, dataset_name)
    input_dim = patch_size * patch_size * pca_dim

    # Extract patches
    patches, labels, coords, h, w = extract_patches(data_pca, gt_processed, patch_size=patch_size)

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)

    patches_tensor = torch.tensor(patches).permute(0, 3, 1, 2).float()
    patches_tensor = patches_tensor.reshape(-1, input_dim)
    train_loader = DataLoader(TensorDataset(patches_tensor), batch_size=256, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train autoencoder
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output, _ = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        _, latent_z = model(patches_tensor.to(device))

    # Transformer
    transformer = SimpleTransformer(dim=latent_dim).to(device)
    trans_scores = []
    batch_size = 256
    for i in range(0, latent_z.shape[0], batch_size):
        batch = latent_z[i:i+batch_size]
        with torch.no_grad():
            scores = transformer(batch.to(device)).cpu().numpy()
        trans_scores.extend(scores)
    trans_scores = np.array(trans_scores)

    # Train SVM classifier
    X = latent_z.cpu().numpy()
    y = np.array(labels)

    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        svm = SVC(kernel='rbf', class_weight='balanced')
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Create visualizations
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        conf_matrix_path = os.path.join(os.path.dirname(hsi_path), 'confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()

        # Create t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
        plt.title('t-SNE Visualization of Latent Space')
        tsne_path = os.path.join(os.path.dirname(hsi_path), 'tsne_visualization.png')
        plt.savefig(tsne_path)
        plt.close()

        # Create anomaly score map
        anomaly_map = np.zeros((h, w))
        for (i, j), score in zip(coords, trans_scores):
            anomaly_map[i, j] = score
        plt.figure(figsize=(10, 8))
        plt.imshow(anomaly_map, cmap='hot')
        plt.colorbar(label='Anomaly Score')
        plt.title('Anomaly Score Map')
        anomaly_map_path = os.path.join(os.path.dirname(hsi_path), 'anomaly_map.png')
        plt.savefig(anomaly_map_path)
        plt.close()

        # Format results for frontend
        results = {
            'stats': {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            },
            'images': [
                {
                    'url': f'/uploads/confusion_matrix.png',
                    'name': 'Confusion Matrix',
                    'description': 'Visualization of model predictions vs true labels'
                },
                {
                    'url': f'/uploads/tsne_visualization.png',
                    'name': 't-SNE Visualization',
                    'description': '2D visualization of the latent space'
                },
                {
                    'url': f'/uploads/anomaly_map.png',
                    'name': 'Anomaly Score Map',
                    'description': 'Spatial distribution of anomaly scores'
                }
            ],
            'info': f'Analysis completed for {dataset_name} dataset with {len(y)} samples'
        }
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        results = {
            'stats': {
                'error': str(e)
            },
            'images': [],
            'info': 'Analysis failed'
        }

    return results
