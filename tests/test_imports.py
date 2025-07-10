"""Test basic package imports."""

def test_package_imports():
    """Test that main package components can be imported."""
    try:
        from hpc_inference import ParquetImageDataset, ImageFolderDataset
        from hpc_inference import __version__, load_config, profiling
        assert __version__ is not None
        print("✅ Package imports successful!")
    except ImportError as e:
        raise AssertionError(f"Failed to import package components: {e}")


def test_specific_module_imports():
    """Test that specific modules can be imported."""
    try:
        from hpc_inference.datasets import ParquetImageDataset, ImageFolderDataset
        from hpc_inference.utils import load_config, decode_image, save_emb_to_parquet, profiling
        print("✅ Specific module imports successful!")
    except ImportError as e:
        raise AssertionError(f"Failed to import specific modules: {e}")


def test_feature_detection():
    """Test the feature detection functions."""
    try:
        import hpc_inference
        features = hpc_inference.list_available_features()
        print(f"✅ Available features: {features}")
        print("\n" + "="*50)
        hpc_inference.print_installation_guide()
        print("="*50)
    except Exception as e:
        raise AssertionError(f"Feature detection failed: {e}")


if __name__ == "__main__":
    test_package_imports()
    test_specific_module_imports()
    test_feature_detection()
    print("🎉 All tests passed!")
