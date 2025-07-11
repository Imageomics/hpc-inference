#!/usr/bin/env python3
"""
Test script to validate import/export structure of the HPC inference package.
Run this script to ensure all modules are properly structured and accessible.

Usage:
    python test_package_structure.py
"""

import sys
import traceback
from pathlib import Path

# Add src to path for testing
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

def test_main_imports():
    """Test that main package imports work correctly."""
    print("🔍 Testing main package imports...")
    
    try:
        import hpc_inference
        
        # Test version info
        assert hasattr(hpc_inference, '__version__')
        assert hasattr(hpc_inference, '__author__')
        assert hasattr(hpc_inference, '__email__')
        
        # Test __all__ exports
        assert hasattr(hpc_inference, '__all__')
        assert len(hpc_inference.__all__) > 0
        
        # Test all exported items are accessible
        for item in hpc_inference.__all__:
            assert hasattr(hpc_inference, item), f"Missing export: {item}"
        
        print(f"✅ Main package imported successfully (v{hpc_inference.__version__})")
        print(f"   Exports: {len(hpc_inference.__all__)} items")
        return True
        
    except Exception as e:
        print(f"❌ Main package import failed: {e}")
        return False

def test_dataset_classes():
    """Test dataset class imports and basic instantiation."""
    print("\n🔍 Testing dataset classes...")
    
    try:
        from hpc_inference import ParquetImageDataset, ImageFolderDataset
        from hpc_inference.datasets import multi_model_collate
        
        # Test that classes are actually classes
        import inspect
        assert inspect.isclass(ParquetImageDataset)
        assert inspect.isclass(ImageFolderDataset)
        assert callable(multi_model_collate)
        
        print("✅ Dataset classes imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataset class import failed: {e}")
        return False

def test_utility_functions():
    """Test utility function imports."""
    print("\n🔍 Testing utility functions...")
    
    try:
        from hpc_inference import (
            decode_image, save_emb_to_parquet, format_time, load_config,
            assign_files_to_rank, assign_indices_to_rank, get_distributed_info
        )
        
        # Test that all are callable
        functions = [
            decode_image, save_emb_to_parquet, format_time, load_config,
            assign_files_to_rank, assign_indices_to_rank, get_distributed_info
        ]
        
        for func in functions:
            assert callable(func), f"{func.__name__} should be callable"
        
        print(f"✅ Utility functions imported successfully ({len(functions)} functions)")
        return True
        
    except Exception as e:
        print(f"❌ Utility function import failed: {e}")
        return False

def test_profiling_module():
    """Test profiling module import."""
    print("\n🔍 Testing profiling module...")
    
    try:
        from hpc_inference.utils import profiling
        
        # Test some key functions exist
        expected_functions = [
            'log_computing_specs', 'start_usage_logging', 
            'save_usage_plots', 'save_batch_stats'
        ]
        
        for func_name in expected_functions:
            assert hasattr(profiling, func_name), f"Missing function: {func_name}"
        
        print(f"✅ Profiling module imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Profiling module import failed: {e}")
        return False

def test_embedding_script():
    """Test that embedding script can be imported."""
    print("\n🔍 Testing embedding script...")
    
    try:
        from hpc_inference.inference.embed.open_clip_embed import main
        assert callable(main)
        
        print("✅ Embedding script imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Embedding script import failed: {e}")
        return False

def test_distributed_functions():
    """Test distributed utility functions with dummy data."""
    print("\n🔍 Testing distributed functions...")
    
    try:
        from hpc_inference.utils.distributed import (
            assign_files_to_rank, get_distributed_info, multi_model_collate
        )
        
        # Test get_distributed_info
        rank, world_size = get_distributed_info()
        assert isinstance(rank, int)
        assert isinstance(world_size, int)
        
        # Test assign_files_to_rank with dummy data
        dummy_files = ['file1.jpg', 'file2.jpg', 'file3.jpg']
        assigned = assign_files_to_rank(0, 2, dummy_files, evenly_distribute=False)
        assert isinstance(assigned, list)
        assert len(assigned) > 0
        
        print("✅ Distributed functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Distributed function test failed: {e}")
        return False

def test_no_circular_imports():
    """Test that there are no circular import issues."""
    print("\n🔍 Testing for circular imports...")
    
    modules_to_test = [
        'hpc_inference',
        'hpc_inference.utils',
        'hpc_inference.utils.common',
        'hpc_inference.utils.distributed',
        'hpc_inference.utils.profiling',
        'hpc_inference.datasets',
        'hpc_inference.datasets.parquet_dataset',
        'hpc_inference.datasets.image_folder_dataset',
        'hpc_inference.inference.embed.open_clip_embed',
    ]
    
    try:
        import importlib
        for module_name in modules_to_test:
            importlib.import_module(module_name)
        
        print("✅ No circular import issues detected")
        return True
        
    except Exception as e:
        print(f"❌ Circular import detected: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🧪 HPC Inference Package Structure Tests")
    print("=" * 45)
    
    tests = [
        test_main_imports,
        test_dataset_classes,
        test_utility_functions,
        test_profiling_module,
        test_embedding_script,
        test_distributed_functions,
        test_no_circular_imports,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 All tests passed! Package structure is correct.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
