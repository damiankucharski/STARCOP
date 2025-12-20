#!/usr/bin/env python3
"""
Create OKAPI fusion set with test-like distribution.

This script creates a held-out set from the training data that matches
the test set distribution. This set will be used by OKAPI to learn
how to fuse base model predictions and should NOT be used for training
the base models.

Created files:
- okapi_fusion.csv: 340 samples matching test distribution
- train_without_okapi.csv: Remaining training samples for base models
- train_easy_without_okapi.csv: Easy samples without OKAPI overlap

Distribution matching:
- Plume/No-plume ratio: ~48.5% / 51.5%
- Easy/Hard plume ratio (qplume > 1000): ~34% / 66%
- Difficulty distribution: ~58% hard, ~26% random, ~17% easy
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def create_okapi_fusion_set(
    data_root: str,
    target_size: int = 340,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create OKAPI fusion set with test-like distribution.

    Args:
        data_root: Path to STARCOP data directory
        target_size: Target number of samples for OKAPI set
        seed: Random seed for reproducibility
        verbose: Print distribution information

    Returns:
        Tuple of (okapi_df, train_without_okapi_df, train_easy_without_okapi_df)
    """
    np.random.seed(seed)

    data_root = Path(data_root)

    # Load datasets
    test_df = pd.read_csv(data_root / 'test.csv')
    train_df = pd.read_csv(data_root / 'train.csv')
    train_easy_df = pd.read_csv(data_root / 'train_easy.csv')

    # Calculate test set distribution
    test_total = len(test_df)
    test_plume = test_df['has_plume'].sum()
    test_no_plume = (~test_df['has_plume']).sum()

    test_plume_df = test_df[test_df['has_plume']]
    test_easy = (test_plume_df['qplume'] > 1000).sum()
    test_hard = (test_plume_df['qplume'] <= 1000).sum()

    if verbose:
        print("=== Target Distribution (Test Set) ===")
        print(f"Total: {test_total}")
        print(f"Plume: {test_plume} ({100*test_plume/test_total:.1f}%)")
        print(f"No plume: {test_no_plume} ({100*test_no_plume/test_total:.1f}%)")
        print(f"Plume - Easy (qplume > 1000): {test_easy} ({100*test_easy/test_plume:.1f}%)")
        print(f"Plume - Hard (qplume <= 1000): {test_hard} ({100*test_hard/test_plume:.1f}%)")

    # Calculate target counts to match test distribution
    okapi_plume_count = int(target_size * test_plume / test_total)
    okapi_no_plume_count = target_size - okapi_plume_count
    okapi_plume_easy = int(okapi_plume_count * test_easy / test_plume)
    okapi_plume_hard = okapi_plume_count - okapi_plume_easy

    if verbose:
        print(f"\n=== Target OKAPI Set ===")
        print(f"Total target: {target_size}")
        print(f"Plume target: {okapi_plume_count} (easy: {okapi_plume_easy}, hard: {okapi_plume_hard})")
        print(f"No plume target: {okapi_no_plume_count}")

    # Split train data by categories
    train_no_plume = train_df[~train_df['has_plume']]
    train_plume = train_df[train_df['has_plume']]
    train_plume_easy = train_plume[train_plume['qplume'] > 1000]
    train_plume_hard = train_plume[train_plume['qplume'] <= 1000]

    if verbose:
        print(f"\n=== Available in Train Set ===")
        print(f"No plume: {len(train_no_plume)}")
        print(f"Plume easy: {len(train_plume_easy)}")
        print(f"Plume hard: {len(train_plume_hard)}")

    # Sample from each category
    okapi_no_plume = train_no_plume.sample(
        n=min(okapi_no_plume_count, len(train_no_plume)),
        random_state=seed
    )
    okapi_plume_easy_samples = train_plume_easy.sample(
        n=min(okapi_plume_easy, len(train_plume_easy)),
        random_state=seed
    )
    okapi_plume_hard_samples = train_plume_hard.sample(
        n=min(okapi_plume_hard, len(train_plume_hard)),
        random_state=seed
    )

    # Combine into OKAPI fusion set
    okapi_df = pd.concat([okapi_no_plume, okapi_plume_easy_samples, okapi_plume_hard_samples])
    okapi_df = okapi_df.sample(frac=1, random_state=seed)  # Shuffle

    if verbose:
        print(f"\n=== Actual OKAPI Set ===")
        print(f"Total: {len(okapi_df)}")
        print(f"Plume: {okapi_df['has_plume'].sum()}")
        print(f"No plume: {(~okapi_df['has_plume']).sum()}")
        okapi_plume_check = okapi_df[okapi_df['has_plume']]
        print(f"Plume easy: {(okapi_plume_check['qplume'] > 1000).sum()}")
        print(f"Plume hard: {(okapi_plume_check['qplume'] <= 1000).sum()}")

    # Create training set without OKAPI samples
    okapi_ids = set(okapi_df['id'].values)
    train_without_okapi = train_df[~train_df['id'].isin(okapi_ids)]
    train_easy_without_okapi = train_easy_df[~train_easy_df['id'].isin(okapi_ids)]

    if verbose:
        print(f"\n=== Train Without OKAPI ===")
        print(f"Total: {len(train_without_okapi)}")
        print(f"Plume: {train_without_okapi['has_plume'].sum()}")
        print(f"No plume: {(~train_without_okapi['has_plume']).sum()}")
        print(f"\n=== Train Easy Without OKAPI ===")
        print(f"Total: {len(train_easy_without_okapi)}")

    return okapi_df, train_without_okapi, train_easy_without_okapi


def verify_no_overlap(
    test_df: pd.DataFrame,
    okapi_df: pd.DataFrame,
    train_wo_df: pd.DataFrame,
) -> bool:
    """Verify there is no overlap between the three sets."""
    test_ids = set(test_df['id'])
    okapi_ids = set(okapi_df['id'])
    train_wo_ids = set(train_wo_df['id'])

    overlaps = {
        'Test ∩ OKAPI': len(test_ids & okapi_ids),
        'Test ∩ Train_wo': len(test_ids & train_wo_ids),
        'OKAPI ∩ Train_wo': len(okapi_ids & train_wo_ids),
    }

    print("\n=== Overlap Check ===")
    for name, count in overlaps.items():
        print(f"{name}: {count}")

    return all(v == 0 for v in overlaps.values())


def main():
    parser = argparse.ArgumentParser(
        description='Create OKAPI fusion set with test-like distribution'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=os.environ.get('STARCOP_DATA_ROOT', '/home/s/Data/STARCOP'),
        help='Path to STARCOP data directory'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=340,
        help='Target number of samples for OKAPI set'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print distributions without saving files'
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Create the fusion set
    okapi_df, train_wo_df, train_easy_wo_df = create_okapi_fusion_set(
        data_root=data_root,
        target_size=args.target_size,
        seed=args.seed,
        verbose=True,
    )

    # Verify no overlap
    test_df = pd.read_csv(data_root / 'test.csv')
    if not verify_no_overlap(test_df, okapi_df, train_wo_df):
        raise ValueError("Overlap detected between sets!")

    if not args.dry_run:
        # Save files
        okapi_path = data_root / 'okapi_fusion.csv'
        train_wo_path = data_root / 'train_without_okapi.csv'
        train_easy_wo_path = data_root / 'train_easy_without_okapi.csv'

        okapi_df.to_csv(okapi_path, index=False)
        train_wo_df.to_csv(train_wo_path, index=False)
        train_easy_wo_df.to_csv(train_easy_wo_path, index=False)

        print(f"\n=== Files Saved ===")
        print(f"{okapi_path} ({len(okapi_df)} samples)")
        print(f"{train_wo_path} ({len(train_wo_df)} samples)")
        print(f"{train_easy_wo_path} ({len(train_easy_wo_df)} samples)")
    else:
        print("\n=== Dry run - no files saved ===")


if __name__ == '__main__':
    main()
