# Neural Network Model for Molecular Property Prediction

This notebook demonstrates a neural network approach to predicting molecular properties using Morgan fingerprints.

## Features

- **Neural Network Model**: Uses TensorFlow/Keras to build a deep neural network for property prediction
- **Morgan Fingerprints**: Represents molecules as fingerprint vectors for machine learning
- **3D Visualization**: Includes 3D plots showing structure-property relationships
- **Interactive Prediction**: Allows prediction on new molecules with visualization
- **Error Analysis**: Detailed analysis of prediction errors with molecule visualization

## Requirements

To run this notebook, you'll need the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- rdkit
- scikit-learn
- tensorflow (or tensorflow-gpu)
- jupyter

Installation:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter tensorflow
pip install rdkit
```

## Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `neural_network_model.ipynb`

## What Sets This Model Apart

Compared to the previous Random Forest model, this neural network approach:

1. Uses Morgan fingerprints instead of descriptors, capturing more structural information
2. Employs deep learning to find complex patterns in molecular structure
3. Provides more sophisticated visualizations, including 3D plots
4. Focuses on boiling point prediction as a demonstration
5. Shows interactive prediction capabilities for new molecules

## Model Architecture

The neural network model uses several dense layers with dropout for regularization:
- Input layer: 1024 features (Morgan fingerprint bits)
- Hidden layers: 512 → 256 → 128 → 64 neurons
- Output layer: 1 neuron (predicted property)

## Example Usage

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('boiling_point_model.h5')

# Function to convert SMILES to fingerprint
def get_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp).reshape(1, -1)

# Predict boiling point
def predict(smiles):
    fp = get_fingerprint(smiles)
    if fp is None:
        return None
    # Note: in real use, you would need to apply the scaler transform/inverse_transform
    prediction = model.predict(fp)[0][0]
    return prediction

# Example
smiles = "CCO"  # Ethanol
bp = predict(smiles)
print(f"Predicted boiling point for {smiles}: {bp:.2f}°C")
``` 