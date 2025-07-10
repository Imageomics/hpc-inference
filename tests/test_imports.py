"""Test basic package imports."""

def test_package_imports():
    """Test that main package components can be imported."""
    try:
        from hpc_inference import ParquetImageDataset, ImageFolderDataset
        from hpc_inference import __version__, load_config
        assert __version__ is not None
        print("✅ Package imports successful!")
    except ImportError as e:
        raise AssertionError(f"Failed to import package components: {e}")


def test_specific_module_imports():
    """Test that specific modules can be imported."""
    try:
        from hpc_inference.datasets import ParquetImageDataset, ImageFolderDataset
        from hpc_inference.utils import load_config, decode_image, save_emb_to_parquet
        from hpc_inference.inference.embed.open_clip_embed import main
        print("✅ Specific module imports successful!")
    except ImportError as e:
        raise AssertionError(f"Failed to import specific modules: {e}")


if __name__ == "__main__":
    test_package_imports()
    test_specific_module_imports()
    print("🎉 All tests passed!")
