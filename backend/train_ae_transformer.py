import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model.ae_transformer import AETransformer
from model.utils import preprocess_data, train_model
import os

def main():
    # Configuration
    dataset_name = 'indian'  # Change as needed: 'pavia', 'salinas', 'indian'
    data_file = f'uploads/Indian_pines_corrected.mat'  # Update path if needed
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    model_save_path = 'backend/model/ae_transformer_model.pth'

    # Preprocess data
    print(f"Loading and preprocessing dataset: {dataset_name}")
    data = preprocess_data(data_file, dataset_name)
    h, w, c = data.shape
    data_reshaped = data.reshape(-1, c)

    # Create DataLoader
    tensor_data = torch.tensor(data_reshaped, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = c
    latent_dim = 32
    model = AETransformer(input_dim=input_dim, latent_dim=latent_dim)

    # Train model
    print("Starting training...")
    trained_model = train_model(model, train_loader, epochs=epochs, lr=learning_rate)

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
