import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, roc_auc_score, precision_recall_curve,
                           accuracy_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Chemistry
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

#plt.rcParams['font.sans-serif'] = ['Arial']  # Support English display
#plt.rcParams['axes.unicode_minus'] = False

def smiles_to_fingerprint(smiles, n_bits=1024):
    """
    Convert SMILES string to Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def calculate_molecular_descriptors(smiles):
    """
    Calculate molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 6
    
    return [
        Descriptors.MolLogP(mol),         # LogP
        Descriptors.NumHDonors(mol),      # Number of H-bond donors
        Descriptors.NumHAcceptors(mol),   # Number of H-bond acceptors
        Descriptors.NumRotatableBonds(mol), # Number of rotatable bonds
        Descriptors.TPSA(mol),            # Topological polar surface area
        Descriptors.NumAromaticRings(mol) # Number of aromatic rings
    ]

def train_ovp_classifier():
    """
    Train OPV molecule random forest classifier
    """
    
    print("="*80)
    print("OPV Molecule Random Forest Classifier Training")
    print("="*80)
    
    # Set paths
    dataset_dir = Path(__file__).parent / 'dataset'
    output_dir = Path(__file__).parent / 'model_output'
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Load data ---
    print(f"\n" + "-"*60)
    print(f"Loading dataset")
    print(f"-"*60)
    
    train_data = pd.read_csv(dataset_dir / 'train_dataset.csv')
    test_data = pd.read_csv(dataset_dir / 'test_dataset.csv')
    
    print(f"Training set: {len(train_data):,} samples")
    print(f"Test set: {len(test_data):,} samples")
    
    # Check label distribution
    print(f"\nTraining set label distribution:")
    train_label_counts = train_data['Label'].value_counts()
    for label, count in train_label_counts.items():
        label_name = "OPV material" if label == 1 else "Non-OPV material"
        print(f"   • {label_name} (label={label}): {count:,} samples ({count/len(train_data)*100:.1f}%)")
    
    # --- 2. Feature extraction ---
    print(f"\n" + "-"*60)
    print(f"Molecular feature extraction")
    print(f"-"*60)
    
    print(f"Extracting Morgan fingerprint features...")
    
    # Extract Morgan fingerprints
    print(f"   • Extracting Morgan fingerprints for training set...")
    X_train_fp = np.array([smiles_to_fingerprint(smiles) for smiles in train_data['SMILES']])
    print(f"   • Extracting Morgan fingerprints for test set...")
    X_test_fp = np.array([smiles_to_fingerprint(smiles) for smiles in test_data['SMILES']])
    
    # Extract molecular descriptors
    print(f"   • Extracting molecular descriptors for training set...")
    X_train_desc = np.array([calculate_molecular_descriptors(smiles) for smiles in train_data['SMILES']])
    print(f"   • Extracting molecular descriptors for test set...")
    X_test_desc = np.array([calculate_molecular_descriptors(smiles) for smiles in test_data['SMILES']])
    
    # Combine features
    X_train = np.hstack([X_train_fp, X_train_desc])
    X_test = np.hstack([X_test_fp, X_test_desc])
    
    y_train = train_data['Label'].values
    y_test = test_data['Label'].values
    
    print(f"Feature extraction completed:")
    print(f"   • Morgan fingerprint dimension: {X_train_fp.shape[1]}")
    print(f"   • Molecular descriptor dimension: {X_train_desc.shape[1]}")
    print(f"   • Total feature dimension: {X_train.shape[1]}")
    
    # --- 3. Train Random Forest model ---
    print(f"\n" + "-"*60)
    print(f"Training Random Forest Classifier")
    print(f"-"*60)
    
    # Train model
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    print(f"Training model...")
    rf_classifier.fit(X_train, y_train)
    print(f"Model training completed!")
    
    # --- 4. Model evaluation ---
    print(f"\n" + "-"*60)
    print(f"Model performance evaluation")
    print(f"-"*60)
    
    # Prediction
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    y_train_proba = rf_classifier.predict_proba(X_train)[:, 1]
    y_test_proba = rf_classifier.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Performance metrics:")
    print(f"   • Training set accuracy: {train_accuracy:.4f}")
    print(f"   • Test set accuracy: {test_accuracy:.4f}")
    print(f"   • Training set F1 score: {train_f1:.4f}")
    print(f"   • Test set F1 score: {test_f1:.4f}")
    print(f"   • Training set AUC: {train_auc:.4f}")
    print(f"   • Test set AUC: {test_auc:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed classification report (test set):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Non-OPV material', 'OPV material']))
    
    # --- 5. Save model ---
    print(f"\n" + "-"*60)
    print(f"Saving model and weights")
    print(f"-"*60)
    
    # Save full model
    model_file = output_dir / 'ovp_random_forest_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(rf_classifier, f)
    print(f"Model saved: {model_file}")
    
    # --- 6. Visualization ---
    print(f"\n" + "-"*60)
    print(f"Generating visualization charts")
    print(f"-"*60)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Confusion matrix
    plt.subplot(3, 3, 1)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-OPV material', 'OPV material'],
                yticklabels=['Non-OPV material', 'OPV material'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    # 2. ROC curve
    plt.subplot(3, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    plt.subplot(3, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    plt.plot(recall, precision, color='darkgreen', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # 4. Feature importance (Top 20)
    plt.subplot(3, 3, 4)
    feature_names = [f'fp_{i}' for i in range(1024)] + ['LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'NumAromaticRings']
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    plt.bar(range(20), importances[indices])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Feature Rank')
    plt.ylabel('Importance')
    plt.xticks(range(20), [feature_names[i][:10] for i in indices], rotation=45)
    
    # 5. Prediction probability distribution
    plt.subplot(3, 3, 5)
    plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, label='Non-OPV material', color='skyblue')
    plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, label='OPV material', color='lightcoral')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Learning curve (number of trees vs performance)
    plt.subplot(3, 3, 6)
    n_estimators_range = range(10, 201, 20)
    train_scores = []
    test_scores = []
    
    print(f"Calculating learning curve...")
    for n_est in [50, 100, 150, 200]:  # Simplified calculation
        rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf_temp.fit(X_train, y_train)
        train_scores.append(rf_temp.score(X_train, y_train))
        test_scores.append(rf_temp.score(X_test, y_test))
    
    plt.plot([50, 100, 150, 200], train_scores, 'o-', label='Training set', color='blue')
    plt.plot([50, 100, 150, 200], test_scores, 'o-', label='Test set', color='red')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. PCA visualization
    plt.subplot(3, 3, 7)
    print(f"Calculating PCA dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test)
    
    scatter = plt.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], 
                         c='skyblue', label='Non-OPV material', alpha=0.6, s=20)
    scatter = plt.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], 
                         c='lightcoral', label='OPV material', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. t-SNE visualization
    plt.subplot(3, 3, 8)
    print(f"Calculating t-SNE dimensionality reduction...")
    # For speed, use only part of the data
    n_samples = min(1000, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_test_subset = X_test[indices]
    y_test_subset = y_test[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_test_tsne = tsne.fit_transform(X_test_subset)
    
    plt.scatter(X_test_tsne[y_test_subset == 0, 0], X_test_tsne[y_test_subset == 0, 1], 
               c='skyblue', label='Non-OPV material', alpha=0.6, s=20)
    plt.scatter(X_test_tsne[y_test_subset == 1, 0], X_test_tsne[y_test_subset == 1, 1], 
               c='lightcoral', label='OPV material', alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Molecular descriptor distribution
    plt.subplot(3, 3, 9)
    desc_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'NumAromaticRings']
    ovp_desc = X_test_desc[y_test == 1]
    non_ovp_desc = X_test_desc[y_test == 0]
    
    # Compare molecular weight distribution
    plt.hist(non_ovp_desc[:, 0], bins=30, alpha=0.7, label='Non-OPV material', color='skyblue')
    plt.hist(ovp_desc[:, 0], bins=30, alpha=0.7, label='OPV material', color='lightcoral')
    plt.xlabel('MolWt')
    plt.ylabel('Count')
    plt.title('Molecular Weight Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = output_dir / 'ovp_classifier_visualization.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"Visualization charts saved: {viz_file}")
    plt.show()
    
    # --- 7. Generate detailed report ---
    print(f"\n" + "-"*60)
    print(f"Generating model report")
    print(f"-"*60)
    
    report_file = output_dir / 'model_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("OPV Molecule Random Forest Classifier Training Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Dataset information:\n")
        f.write(f"- Training set samples: {len(train_data):,}\n")
        f.write(f"- Test set samples: {len(test_data):,}\n")
        f.write(f"- Feature dimension: {X_train.shape[1]:,}\n\n")
        
        f.write("Model parameters:\n")
        for param, value in rf_classifier.get_params().items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("Performance metrics:\n")
        f.write(f"- Training set accuracy: {train_accuracy:.4f}\n")
        f.write(f"- Test set accuracy: {test_accuracy:.4f}\n")
        f.write(f"- Training set F1 score: {train_f1:.4f}\n")
        f.write(f"- Test set F1 score: {test_f1:.4f}\n")
        f.write(f"- Training set AUC: {train_auc:.4f}\n")
        f.write(f"- Test set AUC: {test_auc:.4f}\n\n")
        
        f.write("Top 10 important features:\n")
        feature_names = [f'fp_{i}' for i in range(1024)] + ['LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'NumAromaticRings']
        importances = rf_classifier.feature_importances_
        top_features = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(top_features):
            f.write(f"{i+1:2d}. {feature_names[idx]}: {importances[idx]:.6f}\n")
        
        f.write(f"\nGenerated files:\n")
        f.write(f"- ovp_random_forest_model.pkl: Trained model\n")
        f.write(f"- ovp_classifier_visualization.png: Visualization charts\n")
        f.write(f"- model_report.txt: This report\n")
    
    print(f"Model report saved: {report_file}")
    
    # --- 8. Prediction examples ---
    print(f"\n" + "-"*60)
    print(f"Model prediction examples")
    print(f"-"*60)
    
    # Select a few samples for prediction demonstration
    sample_indices = np.random.choice(len(test_data), 5, replace=False)
    for i, idx in enumerate(sample_indices):
        smiles = test_data.iloc[idx]['SMILES']
        true_label = test_data.iloc[idx]['Label']
        pred_label = y_test_pred[idx]
        pred_proba = y_test_proba[idx]
        
        true_name = "OPV material" if true_label == 1 else "Non-OPV material"
        pred_name = "OPV material" if pred_label == 1 else "Non-OPV material"
        status = "CORRECT" if true_label == pred_label else "INCORRECT"
        
        print(f"{i+1}. {status} True: {true_name} | Predicted: {pred_name} | Probability: {pred_proba:.3f}")
        print(f"   SMILES: {smiles[:80]}...")
    
    print(f"\n" + "="*80)
    print(f"Model training and visualization completed!")
    print(f"All files saved to: {output_dir.absolute()}")
    print(f"="*80)

if __name__ == "__main__":
    train_ovp_classifier() 