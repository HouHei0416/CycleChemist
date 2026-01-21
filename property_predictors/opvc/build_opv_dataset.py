import pandas as pd
import numpy as np
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

def compute_molwt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    return Descriptors.MolWt(mol)

def build_ovp_dataset():
    """
    Build OPV molecule classification dataset
    - Positive samples: Donor and Acceptor SMILES from OPV data
    - Negative samples: PubChem SMILES, sampled to match the molecular weight distribution of positives
    """
    print("="*80)
    print("Building OPV molecule classification dataset")
    print("="*80)

    # File paths
    base_dir = Path(__file__).parent.parent
    ovp_file = base_dir.parent / 'data' / 'exp_dataset.csv'
    pubchem_file = base_dir.parent / 'data' / 'pubchem_len20-290_no_ions_no_multi_random5M_123.csv'
    output_dir = Path(__file__).parent / 'dataset'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"Input files:")
    print(f"   • OPV data: {ovp_file}")
    print(f"   • PubChem data: {pubchem_file}")
    print(f"Output directory: {output_dir}")

    # --- 1. Read OPV data (positive samples) ---
    print(f"\n" + "-"*60)
    print(f"Reading OPV data (positive samples)")
    print(f"-"*60)

    ovp_df = pd.read_csv(ovp_file)
    print(f"OPV data loaded: {len(ovp_df)} rows")

    # Extract Donor and Acceptor SMILES
    donor_smiles = ovp_df['Donor SMILES'].dropna().unique()
    acceptor_smiles = ovp_df['Acceptor SMILES'].dropna().unique()

    print(f"   • Unique Donor SMILES: {len(donor_smiles)}")
    print(f"   • Unique Acceptor SMILES: {len(acceptor_smiles)}")

    # Combine all positive SMILES, remove duplicates
    positive_smiles = list(set(list(donor_smiles) + list(acceptor_smiles)))
    print(f"   • Combined unique positive samples: {len(positive_smiles)}")

    # Compute molecular weights for positive samples
    print(f"   • Calculating molecular weights for positive samples...")
    pos_molwts = [compute_molwt(smi) for smi in positive_smiles]
    pos_df = pd.DataFrame({'SMILES': positive_smiles, 'MolWt': pos_molwts})
    pos_df = pos_df.dropna().reset_index(drop=True)
    print(f"   • Valid positive samples with MolWt: {len(pos_df)}")

    # --- 2. Read PubChem data (negative samples) ---
    print(f"\n" + "-"*60)
    print(f"Reading PubChem data (negative samples)")
    print(f"-"*60)

    # Read all PubChem SMILES (may be large, so use chunks)
    chunk_size = 10000
    pubchem_smiles = []
    print(f"Reading PubChem SMILES and calculating molecular weights...")
    try:
        chunk_iter = pd.read_csv(pubchem_file, chunksize=chunk_size)
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=['SMILES'])
            chunk['MolWt'] = chunk['SMILES'].apply(compute_molwt)
            chunk = chunk.dropna(subset=['MolWt'])
            pubchem_smiles.extend(chunk[['SMILES', 'MolWt']].values.tolist())
            print(f"   Read {len(pubchem_smiles)} PubChem molecules so far...")
            if len(pubchem_smiles) > 20000:  # Limit for speed, adjust as needed
                break
        print(f"PubChem negative candidates loaded: {len(pubchem_smiles)}")
    except Exception as e:
        print(f"Error reading PubChem data: {e}")
        return
    pubchem_df = pd.DataFrame(pubchem_smiles, columns=['SMILES', 'MolWt'])

    # --- 3. Align negative sample MolWt distribution to positive ---
    print(f"\n" + "-"*60)
    print(f"Aligning negative sample molecular weight distribution to positive samples")
    print(f"-"*60)

    # Bin positive samples by MolWt
    n_bins = 20
    pos_hist, bin_edges = np.histogram(pos_df['MolWt'], bins=n_bins)
    pos_df['bin'] = np.digitize(pos_df['MolWt'], bin_edges, right=True)
    pubchem_df['bin'] = np.digitize(pubchem_df['MolWt'], bin_edges, right=True)

    neg_samples = []
    for b in range(1, n_bins+1):
        n_needed = (pos_df['bin'] == b).sum()
        neg_bin = pubchem_df[pubchem_df['bin'] == b]
        if len(neg_bin) == 0:
            print(f"   Bin {b}: positive={n_needed}, negative available=0, selected=0 (SKIPPED: no negatives in this bin!)")
            continue
        elif len(neg_bin) >= n_needed:
            chosen = neg_bin.sample(n=n_needed, random_state=42)
        else:
            # Not enough negatives in this bin, sample with replacement
            chosen = neg_bin.sample(n=n_needed, replace=True, random_state=42)
        neg_samples.append(chosen)
        print(f"   Bin {b}: positive={n_needed}, negative available={len(neg_bin)}, selected={len(chosen)}")
    neg_df = pd.concat(neg_samples, ignore_index=True)
    neg_df = neg_df.drop_duplicates(subset='SMILES').reset_index(drop=True)

    # If after deduplication we have fewer negatives than positives, sample more from the full negatives to fill up
    if len(neg_df) < len(pos_df):
        extra_needed = len(pos_df) - len(neg_df)
        # Exclude already used negatives
        available_neg = pubchem_df[~pubchem_df['SMILES'].isin(neg_df['SMILES'])]
        if len(available_neg) == 0:
            # If all are used, sample with replacement from all negatives
            available_neg = pubchem_df
        extra_neg = available_neg.sample(n=extra_needed, replace=True, random_state=42)
        neg_df = pd.concat([neg_df, extra_neg], ignore_index=True)
    neg_df = neg_df.sample(n=len(pos_df), random_state=42).reset_index(drop=True)
    print(f"   • Final negative samples: {len(neg_df)}")

    # --- 4. Build dataset ---
    print(f"\n" + "-"*60)
    print(f"Building unified dataset")
    print(f"-"*60)

    # Create positive sample DataFrame
    positive_df = pd.DataFrame({
        'SMILES': pos_df['SMILES'],
        'Label': 1,  # 1 means OPV material
        'Type': 'OPV_Material'
    })

    # Create negative sample DataFrame
    negative_df = pd.DataFrame({
        'SMILES': neg_df['SMILES'],
        'Label': 0,  # 0 means non-OPV material
        'Type': 'Non_OPV_Material'
    })

    # Combine datasets
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset built:")
    print(f"   • Positive samples (OPV): {len(positive_df):,}")
    print(f"   • Negative samples (Non-OPV): {len(negative_df):,}")
    print(f"   • Total: {len(combined_df):,}")
    print(f"   • Positive/Negative ratio: {len(positive_df)/len(negative_df):.3f}")

    # --- 5. Data quality check ---
    print(f"\n" + "-"*60)
    print(f"Data quality check")
    print(f"-"*60)

    # Check nulls
    null_count = combined_df['SMILES'].isnull().sum()
    print(f"   • Null SMILES: {null_count}")

    # Check duplicates
    duplicate_count = combined_df['SMILES'].duplicated().sum()
    print(f"   • Duplicate SMILES: {duplicate_count}")
    if duplicate_count > 0:
        print(f"   Removing duplicates...")
        combined_df = combined_df.drop_duplicates(subset='SMILES').reset_index(drop=True)
        print(f"   After deduplication: {len(combined_df):,}")

    # Check SMILES length distribution
    smiles_lengths = combined_df['SMILES'].str.len()
    print(f"   • SMILES length stats:")
    print(f"     - Min: {smiles_lengths.min()} chars")
    print(f"     - Max: {smiles_lengths.max()} chars")
    print(f"     - Mean: {smiles_lengths.mean():.1f} chars")
    print(f"     - Median: {smiles_lengths.median():.1f} chars")

    # Filter SMILES longer than 400
    long_smiles_count = (smiles_lengths > 400).sum()
    print(f"   • SMILES longer than 400 chars: {long_smiles_count}")
    if long_smiles_count > 0:
        print(f"   Removing SMILES longer than 400 chars...")
        combined_df = combined_df[combined_df['SMILES'].str.len() <= 400].reset_index(drop=True)
        print(f"   After filtering: {len(combined_df):,}")
        # Recompute stats
        smiles_lengths = combined_df['SMILES'].str.len()
        print(f"   • After filtering SMILES length stats:")
        print(f"     - Min: {smiles_lengths.min()} chars")
        print(f"     - Max: {smiles_lengths.max()} chars")
        print(f"     - Mean: {smiles_lengths.mean():.1f} chars")
        print(f"     - Median: {smiles_lengths.median():.1f} chars")
        # Recompute positive/negative stats
        final_positive_count = len(combined_df[combined_df['Label'] == 1])
        final_negative_count = len(combined_df[combined_df['Label'] == 0])
        print(f"   • After filtering sample stats:")
        print(f"     - Positive (OPV): {final_positive_count:,}")
        print(f"     - Negative (Non-OPV): {final_negative_count:,}")
        print(f"     - Positive/Negative ratio: {final_positive_count/final_negative_count:.3f}")

    # --- 6. Save dataset ---
    print(f"\n" + "-"*60)
    print(f"Saving dataset")
    print(f"-"*60)

    # Save full dataset
    full_dataset_file = output_dir / 'ovp_classification_dataset.csv'
    combined_df.to_csv(full_dataset_file, index=False)
    print(f"Full dataset saved: {full_dataset_file}")

    # Save train and test sets
    from sklearn.model_selection import train_test_split
    X = combined_df['SMILES']
    y = combined_df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Build train and test DataFrames
    train_df = pd.DataFrame({
        'SMILES': X_train,
        'Label': y_train,
        'Type': combined_df.loc[X_train.index, 'Type']
    }).reset_index(drop=True)
    test_df = pd.DataFrame({
        'SMILES': X_test,
        'Label': y_test,
        'Type': combined_df.loc[X_test.index, 'Type']
    }).reset_index(drop=True)
    # Save train and test sets
    train_file = output_dir / 'train_dataset.csv'
    test_file = output_dir / 'test_dataset.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Training set saved: {train_file} ({len(train_df):,} samples)")
    print(f"Test set saved: {test_file} ({len(test_df):,} samples)")

    # --- 7. Generate dataset report ---
    print(f"\n" + "-"*60)
    print(f"Generating dataset report")
    print(f"-"*60)
    report_file = output_dir / 'dataset_report.txt'
    # Compute final positive/negative counts
    final_positive_count = len(combined_df[combined_df['Label'] == 1])
    final_negative_count = len(combined_df[combined_df['Label'] == 0])
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("OPV Molecule Classification Dataset Report\n")
        f.write("="*50 + "\n\n")
        f.write("Dataset overview:\n")
        f.write(f"- Total samples: {len(combined_df):,}\n")
        f.write(f"- Positive (OPV): {final_positive_count:,}\n")
        f.write(f"- Negative (Non-OPV): {final_negative_count:,}\n")
        f.write(f"- Positive/Negative ratio: {final_positive_count/final_negative_count:.3f}\n")
        f.write(f"- Max SMILES length: 400 chars (filtered)\n\n")
        f.write("Data split:\n")
        f.write(f"- Training set: {len(train_df):,} ({len(train_df)/len(combined_df)*100:.1f}%)\n")
        f.write(f"- Test set: {len(test_df):,} ({len(test_df)/len(combined_df)*100:.1f}%)\n\n")
        f.write("SMILES length stats:\n")
        f.write(f"- Min: {smiles_lengths.min()} chars\n")
        f.write(f"- Max: {smiles_lengths.max()} chars\n")
        f.write(f"- Mean: {smiles_lengths.mean():.1f} chars\n")
        f.write(f"- Median: {smiles_lengths.median():.1f} chars\n\n")
        f.write("File list:\n")
        f.write(f"- ovp_classification_dataset.csv: Full dataset\n")
        f.write(f"- train_dataset.csv: Training set\n")
        f.write(f"- test_dataset.csv: Test set\n")
        f.write(f"- dataset_report.txt: This report\n")
    print(f"Dataset report saved: {report_file}")

    # --- 8. Show sample preview ---
    print(f"\n" + "-"*60)
    print(f"Dataset sample preview")
    print(f"-"*60)
    print("\nPositive sample examples (first 5):")
    pos_samples = combined_df[combined_df['Label'] == 1].head(5)
    for i, (_, row) in enumerate(pos_samples.iterrows()):
        print(f"   {i+1}. {row['SMILES'][:60]}...")
    print("\nNegative sample examples (first 5):")
    neg_samples = combined_df[combined_df['Label'] == 0].head(5)
    for i, (_, row) in enumerate(neg_samples.iterrows()):
        print(f"   {i+1}. {row['SMILES'][:60]}...")
    print(f"\n" + "="*80)
    print(f"Dataset building completed!")
    print(f"All files saved to: {output_dir.absolute()}")
    print(f"="*80)

if __name__ == "__main__":
    build_ovp_dataset() 