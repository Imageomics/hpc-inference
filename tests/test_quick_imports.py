#!/usr/bin/env python3
"""
Quick import/export verification script.
Run this to quickly check if all imports and exports work.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

def test_basic_imports():
    """Test basic package imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import hpc_inference
        print(f"✅ Main package imported. Version: {hpc_inference.__version__}")
    except Exception as e:
        print(f"❌ Failed to import main package: {e}")
        return False
        
    return True

def test_main_exports():
    """Test main package exports."""
    print("\n🔍 Testing main package exports...")
    
    try:
        import hpc_inference
        
        exports_to_test = [
            "ParquetImageDataset", "ImageFolderDataset",
            "decode_image", "format_time", "load_config",
            "assign_files_to_rank", "get_distributed_info",
            "profiling"
        ]
        
        for export in exports_to_test:
            if hasattr(hpc_inference, export):
                print(f"✅ {export}")
            else:
                print(f"❌ {export} - not found")
                return False
                
    except Exception as e:
        print(f"❌ Error testing exports: {e}")
        return False
        
    return True

def test_utils_imports():
    """Test utils module imports."""
    print("\n🔍 Testing utils imports...")
    
    try:
        from hpc_inference.utils import (
            decode_image, format_time, load_config,
            assign_files_to_rank, profiling
        )
        print("✅ Utils imports successful")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
        
    return True

def test_dataset_imports():
    """Test dataset imports."""
    print("\n🔍 Testing dataset imports...")
    
    try:
        from hpc_inference.datasets import (
            ParquetImageDataset, ImageFolderDataset, multi_model_collate
        )
        print("✅ Dataset imports successful")
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False
        
    return True

def test_distributed_imports():
    """Test distributed utils imports."""
    print("\n🔍 Testing distributed utils imports...")
    
    try:
        from hpc_inference.utils.distributed import (
            assign_files_to_rank, assign_indices_to_rank,
            get_distributed_info, validate_distributed_setup,
            multi_model_collate
        )
        print("✅ Distributed utils imports successful")
    except Exception as e:
        print(f"❌ Distributed utils import failed: {e}")
        return False
        
    return True

def test_embedding_script():
    """Test embedding script can be imported."""
    print("\n🔍 Testing embedding script imports...")
    
    try:
        from hpc_inference.inference.embed.open_clip_embed import main
        print("✅ Embedding script import successful")
    except Exception as e:
        print(f"❌ Embedding script import failed: {e}")
        return False
        
    return True

def check_for_redundant_functions():
    """Check for potentially redundant functions."""
    print("\n🔍 Checking for redundant functions...")
    
    try:
        import inspect
        import hpc_inference.utils.common as common
        import hpc_inference.utils.distributed as distributed
        import hpc_inference.utils.profiling as profiling
        
        common_funcs = [name for name, obj in inspect.getmembers(common, inspect.isfunction) 
                       if not name.startswith('_')]
        distributed_funcs = [name for name, obj in inspect.getmembers(distributed, inspect.isfunction) 
                            if not name.startswith('_')]
        profiling_funcs = [name for name, obj in inspect.getmembers(profiling, inspect.isfunction) 
                          if not name.startswith('_')]
        
        print(f"📊 Function counts:")
        print(f"   Common: {len(common_funcs)}")
        print(f"   Distributed: {len(distributed_funcs)}")  
        print(f"   Profiling: {len(profiling_funcs)}")
        
        # Look for overlaps
        all_funcs = common_funcs + distributed_funcs + profiling_funcs
        unique_funcs = set(all_funcs)
        
        if len(all_funcs) > len(unique_funcs):
            duplicates = [func for func in unique_funcs if all_funcs.count(func) > 1]
            print(f"⚠️  Potential duplicates: {duplicates}")
        else:
            print("✅ No duplicate function names detected")
            
    except Exception as e:
        print(f"❌ Error checking functions: {e}")
        return False
        
    return True

def main():
    """Run all quick tests."""
    print("🚀 Quick Import/Export Verification")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_main_exports, 
        test_utils_imports,
        test_dataset_imports,
        test_distributed_imports,
        test_embedding_script,
        check_for_redundant_functions,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    if all(results):
        print("🎉 All quick tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Run the comprehensive test suite for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
