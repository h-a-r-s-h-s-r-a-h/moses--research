
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gzip

plt.style.use('seaborn-v0_8-darkgrid')

def load_moses_dataset():
    """
    Try to load the MOSES dataset from various possible locations
    """
    possible_paths = [
        'data/train.csv',
        'data/train.csv.gz',
        'moses/dataset/data/train.csv.gz',
        'moses/dataset/data/train.csv'
    ]
    
    for path in possible_paths:
        try:
            if path.endswith('.gz'):
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(path)
            
            if 'version https://git-lfs.github.com/spec/v1' in df.iloc[0, 0]:
                print(f"File {path} is a Git LFS pointer, skipping...")
                continue
                
            if 'SMILES' not in df.columns:
                print(f"File {path} does not contain required 'SMILES' column, skipping...")
                continue
            
            print(f"Successfully loaded dataset from {path}")
            print(f"Dataset shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to load from {path}: {str(e)}")
            continue
    
    return None

def create_sample_dataset():
    """
    Create a sample dataset with diverse molecules
    """
    print("Creating sample dataset...")
    sample_smiles = [
        'CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'c1ccccc1Cl', 'CC(C)CC',
        'CCOC(=O)C', 'c1ccccc1N', 'CCCBr', 'CC(C)(C)C', 'CCOCC', 'CCNCC',
        'COC(=O)C', 'CC#N', 'CNC(=O)C', 'CC(=O)NC', 'CN=C=O', 'c1cc(O)ccc1',
        'CC(C)=O', 'CCS', 'CCC(=O)O', 'CC(C)O', 'CCCC', 'c1cnc[nH]1',
        'c1ccc2ccccc2c1', 'CS(=O)(=O)C', 'CC1CCCCC1', 'C1CCNCC1',
        'c1ccc(F)cc1', 'c1cc(Cl)c(Cl)cc1', 'c1cc(O)c(O)cc1', 'CCC(N)C(=O)O',
        'CC(C)c1ccccc1', 'CC(=O)c1ccccc1', 'c1cccs1', 'c1ccncc1',
        'c1cc(C(=O)O)ccc1', 'CNC', 'c1cc(N)ccc1', 'C1CCCCC1'
    ]
    
    
    np.random.seed(42)
    boiling_points = np.random.normal(100, 50, len(sample_smiles)) 
    for i, smiles in enumerate(sample_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            weight = Descriptors.MolWt(mol)
            atoms = mol.GetNumAtoms()
            rings = mol.GetRingInfo().NumRings()
            boiling_points[i] += weight * 0.2 + atoms * 2 + rings * 15
    
    df = pd.DataFrame({
        'SMILES': sample_smiles,
        'BoilingPoint': boiling_points
    })
    print(f"Sample dataset created with shape: {df.shape}")
    return df

# Try to load the MOSES dataset first
print("Attempting to load MOSES dataset...")
df = load_moses_dataset()

# If MOSES dataset is not available, create sample dataset
if df is None:
    print("\nCould not load MOSES dataset. Creating a sample dataset...")
    df = create_sample_dataset()

print("\nDataset preview:")
print(df.head())

# Function to calculate fingerprints
def get_morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fingerprint)

# Calculate fingerprints for all molecules
print("\nCalculating molecular fingerprints...")
fingerprints = []
valid_indices = []

for i, smiles in enumerate(df['SMILES']):
    fp = get_morgan_fingerprint(smiles)
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(i)

# Convert fingerprints to numpy array
X = np.array(fingerprints)
y = df['BoilingPoint'].values[valid_indices]

print(f"Created {len(fingerprints)} valid fingerprints")
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")


class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MoleculeDataset(X_train_scaled, y_train)
test_dataset = MoleculeDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


class MoleculeNet(nn.Module):
    def __init__(self, input_size):
        super(MoleculeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize the model
model = MoleculeNet(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Testing
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            test_loss += loss.item()
    
    # Record losses
    train_losses.append(train_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}')

# Plot training and testing losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Losses')
plt.legend()
plt.savefig('training_losses.png')
plt.close()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Boiling Point')
plt.ylabel('Predicted Boiling Point')
plt.title('Actual vs Predicted Boiling Points')
plt.savefig('predictions.png')
plt.close() 