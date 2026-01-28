"""
Data loading functions for DNA methylation age prediction.

This module provides functions to load methylation data, annotations,
and CpG site information from CSV files.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_annotations(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate sample annotations.

    Args:
        data_dir: Directory containing the annotation file (str or Path)

    Returns:
        DataFrame with sample annotations indexed by Sample_description

    Raises:
        FileNotFoundError: If annotation file doesn't exist
        ValueError: If required columns are missing

    Example:
        >>> annot = load_annotations(Path("Data"))
        >>> print(annot.columns)
        Index(['Sample_name', 'female', 'ethnicity', 'age'], dtype='object')
    """
    data_dir = Path(data_dir)  # Convert to Path if string
    annot_path = data_dir / "annot_projet.csv"

    if not annot_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    logger.info(f"Loading annotations from {annot_path}")

    annot = pd.read_csv(annot_path)

    # Validate required columns
    required_cols = ["age", "Sample_description"]
    missing_cols = [col for col in required_cols if col not in annot.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out samples with missing age or description
    initial_count = len(annot)
    annot = annot.dropna(subset=required_cols).copy()
    filtered_count = initial_count - len(annot)

    if filtered_count > 0:
        logger.warning(f"Filtered {filtered_count} samples with missing age or description")

    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")

    logger.info(f"Loaded {len(annot)} samples")

    return annot


def load_cpg_names(data_dir: Union[str, Path]) -> list[str]:
    """
    Load CpG site names from file.

    Args:
        data_dir: Directory containing CpG names file (str or Path)

    Returns:
        List of CpG site names

    Raises:
        FileNotFoundError: If CpG names file doesn't exist

    Example:
        >>> cpg_names = load_cpg_names(Path("Data"))
        >>> print(len(cpg_names))
        894353
    """
    data_dir = Path(data_dir)  # Convert to Path if string
    cpg_path = data_dir / "cpg_names_projet.csv"

    if not cpg_path.exists():
        raise FileNotFoundError(f"CpG names file not found: {cpg_path}")

    logger.info(f"Loading CpG names from {cpg_path}")

    cpg = pd.read_csv(cpg_path, usecols=["cpg_names"])
    cpg_names = cpg["cpg_names"].astype(str).tolist()

    logger.info(f"Loaded {len(cpg_names)} CpG sites")

    return cpg_names


def load_clock_cpgs(path: Optional[Union[str, Path]]) -> set[str]:
    """
    Load a predefined list of CpG sites (e.g., Horvath clock).

    Args:
        path: Path to file containing CpG site names (one per line)

    Returns:
        Set of CpG site names, or empty set if path is None

    Example:
        >>> clock_cpgs = load_clock_cpgs(Path("horvath_cpgs.txt"))
        >>> len(clock_cpgs)
        353
    """
    if path is None:
        return set()

    path = Path(path)  # Convert to Path if string
    if not path.exists():
        logger.warning(f"Clock CpG file not found: {path}")
        return set()

    logger.info(f"Loading clock CpGs from {path}")

    cpgs = pd.read_csv(path, header=None)
    clock_set = set(cpgs.iloc[:, 0].astype(str).tolist())

    logger.info(f"Loaded {len(clock_set)} clock CpG sites")

    return clock_set


def load_selected_cpgs(
    data_path: Union[str, Path],
    sample_ids: list[str],
    selected_indices: list[int],
    selected_names: list[str],
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """
    Load only the selected CpG sites from the full methylation data.

    This function efficiently loads a subset of CpG sites by reading the file in chunks
    and extracting only the rows corresponding to selected indices.

    Args:
        data_path: Path to the full methylation CSV file
        sample_ids: List of sample IDs to load
        selected_indices: Row indices of selected CpG sites
        selected_names: Names corresponding to selected indices
        chunk_size: Number of rows to read at a time (default: 2000)

    Returns:
        DataFrame with selected CpG sites as rows and samples as columns

    Example:
        >>> df = load_selected_cpgs(
        ...     Path("Data/c_sample.csv"),
        ...     sample_ids=["S1", "S2"],
        ...     selected_indices=[0, 100, 500],
        ...     selected_names=["cg1", "cg2", "cg3"]
        ... )
    """
    data_path = Path(data_path)  # Convert to Path if string
    if not data_path.exists():
        raise FileNotFoundError(f"Methylation data file not found: {data_path}")

    logger.info(f"Loading {len(selected_indices)} selected CpG sites from {data_path}")

    indices = np.array(sorted(selected_indices))
    rows = []
    start = 0

    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        end = start + len(chunk)

        # Find which selected indices fall in this chunk
        pos_start = np.searchsorted(indices, start)
        pos_end = np.searchsorted(indices, end)

        # Convert global indices to local chunk indices
        local = indices[pos_start:pos_end] - start

        if len(local) > 0:
            rows.append(chunk.iloc[local])

        start = end

    if not rows:
        raise ValueError("No CpG sites were selected")

    selected = pd.concat(rows, axis=0)
    selected.index = selected_names

    logger.info(f"Loaded shape: {selected.shape}")

    return selected
