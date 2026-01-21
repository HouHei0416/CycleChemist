import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def preprocess_smiles(data, smiles_column="SMILES", cid_column=None, filter_ions=True, filter_multiples=True):
    """
    Preprocess data: filter invalid SMILES, optionally drop ionic and multi-molecule entries.

    Args:
        data (DataFrame): Raw data containing SMILES, may include CID.
        smiles_column (str): Column name for SMILES strings.
        cid_column (str or None): Column name for CID, None if absent.
        filter_ions (bool): Drop molecules with charged atoms if True.
        filter_multiples (bool): Drop multi-molecule SMILES containing '.' if True.

    Returns:
        DataFrame: Valid, standardized SMILES with CID preserved when available.
    """
    valid_data = []
    print(f"Checking validity of SMILES strings... (Filter ions: {filter_ions}, Filter multi-molecule: {filter_multiples})")

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing SMILES"):
        smi = row[smiles_column]
        cid = row[cid_column] if cid_column else None  # 只有當 cid_column 存在時才取值

        if isinstance(smi, str):  # ensure string
            # Filter out multi-molecule systems containing '.' when enabled
            if filter_multiples and "." in smi:
                continue
            
            mol = Chem.MolFromSmiles(smi)  # parse SMILES
            if mol:
                try:
                    # Filter ionic molecules when enabled
                    if filter_ions:
                        has_ion = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
                        if has_ion:
                            continue

                    mol = Chem.RemoveHs(mol)  # remove hydrogens
                    std_smiles = Chem.MolToSmiles(mol)  # standardize SMILES
                    if std_smiles:  # ensure still valid
                        if cid_column:
                            valid_data.append((cid, std_smiles))
                        else:
                            valid_data.append((std_smiles,))
                except:
                    pass  # treat as invalid if standardization fails

    # Build new DataFrame
    columns = [cid_column, smiles_column] if cid_column else [smiles_column]
    processed_data = pd.DataFrame(valid_data, columns=columns)
    return processed_data


def downsample_smiles(
    data,
    output_file: str,
    smiles_column: str = "SMILES",
    cid_column: str = None,
    min_length: int = 15,
    max_length: int = 1200,
    pre_sample_size: int = 1200000,
    final_sample_size: int = 1000000,
    random_state: int = 42,
    filter_ions: bool = True,
    filter_multiples: bool = True,
):
    """
    Pipeline:
    1) Filter by SMILES length (min_length ~ max_length).
    2) Randomly sample pre_sample_size from the length-filtered set.
    3) Filter ions and multi-molecule entries.
    4) Randomly sample final_sample_size from the filtered set and save.
    """

    print("Filtering SMILES based on length...")
    length_filtered_df = data[
        (data[smiles_column].str.len() >= min_length) & 
        (data[smiles_column].str.len() <= max_length)
    ]
    print(f"Total rows after length filtering: {len(length_filtered_df)}")

    print(f"Randomly selecting {pre_sample_size} samples from length-filtered dataset...")
    sampled_df = length_filtered_df.sample(n=min(pre_sample_size, len(length_filtered_df)), random_state=random_state)

    # Preprocess: filter ions and multi-molecule systems
    valid_data = preprocess_smiles(sampled_df, smiles_column, cid_column, filter_ions, filter_multiples)
    print(f"Total valid rows after preprocessing: {len(valid_data)}")

    # Final sampling
    print(f"Final sampling {final_sample_size} SMILES from the filtered dataset...")
    sample_df = valid_data.sample(n=min(final_sample_size, len(valid_data)), random_state=random_state)

    # Save to CSV
    sample_df.to_csv(output_file, index=False)
    print(f"Randomly sampled {len(sample_df)} SMILES saved as: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Downsample SMILES dataset with filtering and preprocessing")
    
    # Input/Output
    parser.add_argument("--input", "-i", type=str, default="CID-SMILES.csv",
                        help="Input CSV file path (default: CID-SMILES.csv)")
    parser.add_argument("--output", "-o", type=str, 
                        default="pubchem_len20-290_no_ions_no_multi_random5M_123.csv",
                        help="Output CSV file path")
    
    # Column names
    parser.add_argument("--smiles_column", type=str, default="SMILES",
                        help="Column name for SMILES strings (default: SMILES)")
    parser.add_argument("--cid_column", type=str, default=None,
                        help="Column name for CID (default: None, auto-detect)")
    
    # Length filtering
    parser.add_argument("--min_length", type=int, default=20,
                        help="Minimum SMILES length (default: 20)")
    parser.add_argument("--max_length", type=int, default=290,
                        help="Maximum SMILES length (default: 290)")
    
    # Sampling parameters
    parser.add_argument("--pre_sample_size", type=int, default=6000000,
                        help="Number of samples to select before preprocessing (default: 6000000)")
    parser.add_argument("--final_sample_size", type=int, default=5000000,
                        help="Final number of samples after preprocessing (default: 5000000)")
    parser.add_argument("--random_state", type=int, default=123,
                        help="Random seed for reproducibility (default: 123)")
    
    # Filtering options
    parser.add_argument("--filter_ions", action="store_true", default=True,
                        help="Filter out ionic molecules (default: True)")
    parser.add_argument("--no_filter_ions", dest="filter_ions", action="store_false",
                        help="Do not filter ionic molecules")
    parser.add_argument("--filter_multiples", action="store_true", default=True,
                        help="Filter out multi-molecule SMILES (default: True)")
    parser.add_argument("--no_filter_multiples", dest="filter_multiples", action="store_false",
                        help="Do not filter multi-molecule SMILES")
    
    # CSV reading options
    parser.add_argument("--header", type=int, default=None,
                        help="Row to use as column names (default: None, use first row)")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="Column names if header=None (default: None)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    if args.names:
        data = pd.read_csv(args.input, header=args.header, names=args.names)
    else:
        data = pd.read_csv(args.input, header=args.header)
    
    # Detect CID column if not specified
    if args.cid_column is None:
        if "CID" in data.columns:
            cid_col = "CID"
        else:
            cid_col = None
    else:
        cid_col = args.cid_column if args.cid_column in data.columns else None
    
    print(f"Detected columns: {list(data.columns)}")
    if cid_col:
        print(f"Using CID column: {cid_col}")
    else:
        print("No CID column detected or specified")
    
    # Run downsampling
    downsample_smiles(
        data,
        output_file=args.output,
        smiles_column=args.smiles_column,
        cid_column=cid_col,
        min_length=args.min_length,
        max_length=args.max_length,
        pre_sample_size=args.pre_sample_size,
        final_sample_size=args.final_sample_size,
        random_state=args.random_state,
        filter_ions=args.filter_ions,
        filter_multiples=args.filter_multiples,
    )


if __name__ == "__main__":
    main()
