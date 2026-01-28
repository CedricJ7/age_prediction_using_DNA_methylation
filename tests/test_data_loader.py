"""
Unit tests for data loading module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.data_loader import (
    load_annotations,
    load_cpg_names,
    load_clock_cpgs,
    load_selected_cpgs
)


def test_load_annotations_success(temp_data_dir):
    """Test successful annotation loading."""
    annot = load_annotations(temp_data_dir)

    assert isinstance(annot, pd.DataFrame)
    assert len(annot) == 3
    assert annot.index.name == "Sample_description"
    assert "age" in annot.columns
    assert "female" in annot.columns


def test_load_annotations_filters_missing_age(tmp_path):
    """Test that samples with missing age are filtered out."""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    # Create annotation with missing age
    annot = pd.DataFrame({
        'Sample_name': ['S1', 'S2', 'S3'],
        'age': [25.0, None, 30.0],
        'Sample_description': ['D1', 'D2', 'D3']
    })
    annot.to_csv(data_dir / "annot_projet.csv", index=False)

    result = load_annotations(data_dir)

    assert len(result) == 2
    assert 'D2' not in result.index


def test_load_annotations_missing_file(tmp_path):
    """Test error handling when annotation file is missing."""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_annotations(data_dir)


def test_load_cpg_names_success(temp_data_dir):
    """Test successful CpG names loading."""
    cpg_names = load_cpg_names(temp_data_dir)

    assert isinstance(cpg_names, list)
    assert len(cpg_names) == 1000
    assert all(isinstance(name, str) for name in cpg_names)
    assert cpg_names[0].startswith('cg')


def test_load_cpg_names_missing_file(tmp_path):
    """Test error handling when CpG names file is missing."""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_cpg_names(data_dir)


def test_load_clock_cpgs_success(tmp_path):
    """Test loading clock CpG list."""
    cpg_file = tmp_path / "clock_cpgs.txt"
    pd.DataFrame(['cg00000001', 'cg00000002', 'cg00000003']).to_csv(
        cpg_file, header=False, index=False
    )

    clock_cpgs = load_clock_cpgs(cpg_file)

    assert isinstance(clock_cpgs, set)
    assert len(clock_cpgs) == 3
    assert 'cg00000001' in clock_cpgs


def test_load_clock_cpgs_none_path():
    """Test that None path returns empty set."""
    clock_cpgs = load_clock_cpgs(None)

    assert isinstance(clock_cpgs, set)
    assert len(clock_cpgs) == 0


def test_load_clock_cpgs_missing_file(tmp_path):
    """Test behavior when clock file doesn't exist."""
    cpg_file = tmp_path / "nonexistent.txt"

    clock_cpgs = load_clock_cpgs(cpg_file)

    assert isinstance(clock_cpgs, set)
    assert len(clock_cpgs) == 0


def test_load_selected_cpgs_success(temp_data_dir):
    """Test loading selected CpG sites."""
    sample_ids = ['D1', 'D2', 'D3']
    selected_indices = [0, 10, 50, 100, 500]
    selected_names = [f'cg{i:07d}' for i in selected_indices]

    df = load_selected_cpgs(
        data_path=temp_data_dir / "c_sample.csv",
        sample_ids=sample_ids,
        selected_indices=selected_indices,
        selected_names=selected_names,
        chunk_size=100
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (len(selected_indices), len(sample_ids))
    assert list(df.columns) == sample_ids
    assert list(df.index) == selected_names


def test_load_selected_cpgs_missing_file(tmp_path):
    """Test error when methylation file is missing."""
    data_path = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        load_selected_cpgs(
            data_path=data_path,
            sample_ids=['D1'],
            selected_indices=[0],
            selected_names=['cg0000000'],
            chunk_size=100
        )
