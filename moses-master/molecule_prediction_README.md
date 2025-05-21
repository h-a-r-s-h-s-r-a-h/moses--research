# Molecular Property Prediction Notebook

This repository contains a Jupyter notebook for predicting molecular properties using the MOSES dataset and machine learning.

## Requirements

To run this notebook, you'll need the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- rdkit
- scikit-learn
- jupyter

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install rdkit
```

## Dataset

The notebook is designed to work with the MOSES dataset, which contains molecular structures in SMILES format. If the dataset is not available, the notebook will create a sample dataset for demonstration purposes.

If you have Git LFS installed and want to use the full MOSES dataset:

```bash
git lfs pull
```

## Running the Notebook

1. Clone this repository
2. Navigate to the repository directory
3. Start Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```

4. Open the `molecule_prediction_model.ipynb` notebook

## What the Notebook Does

The notebook performs the following steps:

1. Loads molecular data (MOSES dataset or sample data)
2. Calculates molecular descriptors using RDKit
3. Performs exploratory data analysis with visualizations
4. Builds a Random Forest model to predict molecular properties
5. Analyzes feature importance
6. Visualizes predictions and errors
7. Displays representative molecules

## Output

The notebook will generate:
- Visualizations of molecular properties
- A trained Random Forest model saved as `molecule_property_model.pkl`
- A list of model features saved as `model_features.csv`

## Using the Model for Predictions

After running the notebook, you can load the saved model to make predictions on new molecules:

```python
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load the model
with open('molecule_property_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the features list
features_df = pd.read_csv('model_features.csv')
features = features_df['Feature'].tolist()

# Function to calculate descriptors for a new molecule
def calc_descriptors_for_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {}
    descriptors['MolWt'] = Descriptors.MolWt(mol)
    descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['NumAromaticRings'] = Chem.Lipinski.NumAromaticRings(mol)
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['NumAtoms'] = mol.GetNumAtoms()
    
    return descriptors

# Example prediction
smiles = "CCO"  # Ethanol
descriptors = calc_descriptors_for_prediction(smiles)
if descriptors:
    X = pd.DataFrame([descriptors])[features]
    prediction = model.predict(X)[0]
    print(f"Predicted property for {smiles}: {prediction:.2f}")
``` 