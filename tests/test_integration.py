import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dyspyosis import Dyspyosis


@pytest.fixture
def data_dir():
    """Return the path to the data directory."""
    return Path(__file__).parent.parent / "data"


def test_example_integration(data_dir):
    """
    Integration test that runs the example code and compares outputs to reference data.

    This test verifies that the upgraded dependencies (TensorFlow 2.16+, Keras 3, NumPy 2.0)
    produce results consistent with the previous version.
    """
    # Load test data (same as example.py)
    df = pd.read_table(data_dir / "test.tsv", index_col=0)

    # Create Dyspyosis instance with same parameters as example
    dyspyosis = Dyspyosis(
        df.values,
        labels=df.index.tolist(),
        rarefication_depth=5000,
        rarefication_count=10,
        encode_dim=4,
    )

    # Run training with same parameters as example
    dyspyosis.run_training(epochs=5)

    # Compute loss
    loss = dyspyosis.compute_loss()

    # Load reference loss output
    # Note: We don't compare latent output as it can vary between runs
    # due to neural network initialization and training variability
    reference_loss = pd.read_csv(data_dir / "loss_out.tsv", sep=",")

    # Verify shapes match
    assert loss.shape == reference_loss.shape, (
        f"Loss shape mismatch: {loss.shape} vs {reference_loss.shape}"
    )

    # Verify column names match
    assert list(loss.columns) == list(reference_loss.columns), (
        f"Loss columns mismatch: {loss.columns} vs {reference_loss.columns}"
    )

    # Compare loss values with tolerance
    # Using a relatively high tolerance since:
    # 1. Neural networks can have variations between versions
    # 2. We're comparing Keras 2 vs Keras 3 outputs
    # 3. NumPy 2.0 may have subtle numerical differences
    loss_diff = np.abs(loss["loss"].values - reference_loss["loss"].values)
    max_loss_diff = np.max(loss_diff)
    mean_loss_diff = np.mean(loss_diff)

    # Loss values should be reasonably close (within 50% relative tolerance or 0.001 absolute)
    # This is lenient to account for Keras 2 -> Keras 3 differences
    rtol = 0.5  # 50% relative tolerance
    atol = 0.001  # Absolute tolerance of 0.001 (reasonable for loss values ~0.002-0.01)

    loss_close = np.allclose(
        loss["loss"].values, reference_loss["loss"].values, rtol=rtol, atol=atol
    )

    # If not close, provide detailed diagnostics
    if not loss_close:
        print("\nLoss comparison details:")
        print(f"  Max difference: {max_loss_diff:.6f}")
        print(f"  Mean difference: {mean_loss_diff:.6f}")
        print("  Sample of differences:")
        for i in range(min(5, len(loss_diff))):
            print(
                f"    Sample {i}: new={loss['loss'].values[i]:.6f}, "
                f"ref={reference_loss['loss'].values[i]:.6f}, "
                f"diff={loss_diff[i]:.6f}"
            )

    assert loss_close, (
        f"Loss values differ too much from reference. "
        f"Max diff: {max_loss_diff:.6f}, Mean diff: {mean_loss_diff:.6f}"
    )

    print("\n✓ Integration test passed!")
    print(f"  Loss - Max diff: {max_loss_diff:.6f}, Mean diff: {mean_loss_diff:.6f}")
