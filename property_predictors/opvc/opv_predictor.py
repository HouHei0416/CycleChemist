import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class OPVPredictor:
    """Simplified OPV molecule classifier"""

    def __init__(self, model_path=None):
        """Initialize and load the model"""
        if model_path is None:
            model_dir = Path(__file__).parent / 'model_output'
            model_path = model_dir / 'ovp_random_forest_model.pkl'
        else:
            model_path = Path(model_path)

        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded successfully")

    def extract_features(self, smiles):
        """Extract molecular features"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fingerprint = np.array(fp)

        # Molecular descriptors
        descriptors = [
            #Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol)
        ]

        # Combine features
        features = np.hstack([fingerprint, descriptors])
        return features.reshape(1, -1)

    def predict_single(self, smiles):
        """Predict whether the SMILES is an OPV material"""
        # Validate SMILES
        if not smiles or len(smiles) > 400:
            return {"error": "Invalid or too long SMILES"}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES format"}

        # Extract features
        features = self.extract_features(smiles)
        if features is None:
            return {"error": "Feature extraction failed"}

        # Prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        result = {
            "prediction": "OPV material" if prediction == 1 else "Non-OPV material",
            "ovp_probability": float(probability[1]),
            "confidence": float(max(probability))
        }

        return result

    def predict(self, smiles_list):
        """
        Predict OPV classification for a list of SMILES strings.

        Args:
            smiles_list (list of str): List of SMILES strings.

        Returns:
            tuple: (predictions, probabilities)
                - predictions: np.ndarray of shape (len(smiles_list),), with -1 for errors
                - probabilities: np.ndarray of shape (len(smiles_list), 2), with np.nan for errors
        """
        n = len(smiles_list)
        predictions = np.full(n, -1, dtype=int)
        probabilities = np.full((n, 2), np.nan, dtype=float)
        features_list = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            if not smiles or len(smiles) > 400:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            features = self.extract_features(smiles)
            if features is None:
                continue
            features_list.append(features[0])  # features is (1, N)
            valid_indices.append(idx)

        if features_list:
            features_array = np.vstack(features_list)
            preds = self.model.predict(features_array)
            probs = self.model.predict_proba(features_array)
            for i, idx in enumerate(valid_indices):
                predictions[idx] = preds[i]
                probabilities[idx] = probs[i]

        return predictions, probabilities

def main():
    """Main function"""
    print("OPV Molecule Classifier")
    print("-" * 40)
    
    # Initialize predictor
    predictor = OPVPredictor()

    # --- Test predict_single on a few SMILES ---
    print("\nTesting predict_single on a few SMILES:")
    smiles = ["CN1C(=O)C(=CC2=CC=C(c3cnc(-c4ccc(C(=O)c5ccoc5)s4)nc3)C3N=C(F)C(F)=NC23)C(=O)N(C)C1=O"]
    result, prob = predictor.predict(smiles)
    print("Result:", result, prob[0][1])


if __name__ == "__main__":
    main() 