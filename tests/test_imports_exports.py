"""
Comprehensive tests for imports and exports across the HPC inference package.

This test suite verifies:
1. All __init__.py files properly export their modules
2. No circular imports exist
3. All exported functions/classes are accessible
4. No redundant or conflicting exports
"""

import sys
import unittest
import importlib
from pathlib import Path
from typing import Any, Dict, List, Set

# Add src to path for testing
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestImportsExports(unittest.TestCase):
    """Test all imports and exports work correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.import_errors = []
        self.export_errors = []
        
    def test_main_package_import(self):
        """Test main package can be imported."""
        try:
            import hpc_inference
            self.assertTrue(hasattr(hpc_inference, '__version__'))
            self.assertTrue(hasattr(hpc_inference, '__all__'))
            print(f"✅ Main package imported. Version: {hpc_inference.__version__}")
        except ImportError as e:
            self.fail(f"Failed to import main package: {e}")

    def test_main_package_exports(self):
        """Test all exports in main package __all__ are accessible."""
        import hpc_inference
        
        expected_exports = [
            "__version__",
            "ParquetImageDataset",
            "ImageFolderDataset", 
            "decode_image",
            "save_emb_to_parquet",
            "format_time",
            "load_config",
            "assign_files_to_rank",
            "assign_indices_to_rank",
            "get_distributed_info",
            "profiling",
        ]
        
        for export in expected_exports:
            with self.subTest(export=export):
                self.assertIn(export, hpc_inference.__all__, 
                             f"{export} not in __all__")
                self.assertTrue(hasattr(hpc_inference, export),
                               f"{export} not accessible in main package")
        
        print(f"✅ All {len(expected_exports)} main package exports verified")

    def test_utils_module_imports(self):
        """Test utils module and its submodules."""
        from hpc_inference.utils import (
            decode_image, save_emb_to_parquet, format_time, load_config,
            assign_files_to_rank, assign_indices_to_rank, 
            get_distributed_info, validate_distributed_setup,
            multi_model_collate, profiling
        )
        
        # Test that functions are callable
        self.assertTrue(callable(decode_image))
        self.assertTrue(callable(save_emb_to_parquet))
        self.assertTrue(callable(format_time))
        self.assertTrue(callable(load_config))
        self.assertTrue(callable(assign_files_to_rank))
        self.assertTrue(callable(assign_indices_to_rank))
        self.assertTrue(callable(get_distributed_info))
        self.assertTrue(callable(validate_distributed_setup))
        self.assertTrue(callable(multi_model_collate))
        
        print("✅ All utils functions imported and callable")

    def test_distributed_utils_imports(self):
        """Test distributed utils specifically."""
        from hpc_inference.utils.distributed import (
            assign_files_to_rank,
            assign_indices_to_rank, 
            get_distributed_info,
            validate_distributed_setup,
            multi_model_collate
        )
        
        # Test function signatures exist
        import inspect
        self.assertTrue(len(inspect.signature(assign_files_to_rank).parameters) >= 3)
        self.assertTrue(len(inspect.signature(assign_indices_to_rank).parameters) >= 3)
        self.assertTrue(len(inspect.signature(get_distributed_info).parameters) == 0)
        self.assertTrue(len(inspect.signature(validate_distributed_setup).parameters) == 2)
        self.assertTrue(len(inspect.signature(multi_model_collate).parameters) == 1)
        
        print("✅ All distributed utils functions have correct signatures")

    def test_dataset_imports(self):
        """Test dataset classes and their dependencies."""
        from hpc_inference.datasets import (
            ParquetImageDataset,
            ImageFolderDataset,
            multi_model_collate
        )
        
        # Test classes are actually classes
        import inspect
        self.assertTrue(inspect.isclass(ParquetImageDataset))
        self.assertTrue(inspect.isclass(ImageFolderDataset))
        self.assertTrue(callable(multi_model_collate))
        
        print("✅ All dataset classes imported successfully")

    def test_profiling_module(self):
        """Test profiling module functions."""
        from hpc_inference.utils import profiling
        
        # Check expected functions exist
        expected_functions = [
            'log_computing_specs',
            'start_usage_logging', 
            'save_batch_stats',
            'save_usage_log',
            'save_usage_plots',
            'save_batch_timings_plot'
        ]
        
        for func_name in expected_functions:
            with self.subTest(function=func_name):
                self.assertTrue(hasattr(profiling, func_name),
                               f"{func_name} not found in profiling module")
        
        print("✅ Profiling module functions verified")

    def test_no_circular_imports(self):
        """Test for circular import dependencies."""
        import sys
        
        # Clear any previously imported modules to test fresh imports
        modules_to_test = [
            'hpc_inference',
            'hpc_inference.utils',
            'hpc_inference.utils.common',
            'hpc_inference.utils.distributed', 
            'hpc_inference.utils.profiling',
            'hpc_inference.datasets',
            'hpc_inference.datasets.parquet_dataset',
            'hpc_inference.datasets.image_folder_dataset',
        ]
        
        for module_name in modules_to_test:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Try importing each module individually
        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                    print(f"✅ {module_name} imported without circular dependencies")
                except ImportError as e:
                    self.fail(f"Circular import or missing dependency in {module_name}: {e}")

    def test_embedding_script_imports(self):
        """Test that embedding scripts can import their dependencies."""
        try:
            from hpc_inference.inference.embed import open_clip_embed
            self.assertTrue(hasattr(open_clip_embed, 'main'))
            print("✅ OpenCLIP embedding script imports work")
        except ImportError as e:
            self.fail(f"Failed to import embedding script: {e}")

    def test_redundant_function_detection(self):
        """Check for any redundant function definitions."""
        import ast
        import inspect
        
        # Get all modules
        import hpc_inference.utils.common as common
        import hpc_inference.utils.distributed as distributed
        import hpc_inference.utils.profiling as profiling
        
        # Get all function names from each module
        common_funcs = set([name for name, obj in inspect.getmembers(common) 
                           if inspect.isfunction(obj) and not name.startswith('_')])
        distributed_funcs = set([name for name, obj in inspect.getmembers(distributed) 
                                if inspect.isfunction(obj) and not name.startswith('_')])
        profiling_funcs = set([name for name, obj in inspect.getmembers(profiling) 
                              if inspect.isfunction(obj) and not name.startswith('_')])
        
        # Check for overlaps
        common_distributed_overlap = common_funcs.intersection(distributed_funcs)
        common_profiling_overlap = common_funcs.intersection(profiling_funcs)
        distributed_profiling_overlap = distributed_funcs.intersection(profiling_funcs)
        
        print(f"📊 Function counts - Common: {len(common_funcs)}, "
              f"Distributed: {len(distributed_funcs)}, Profiling: {len(profiling_funcs)}")
        
        if common_distributed_overlap:
            print(f"⚠️  Common-Distributed overlap: {common_distributed_overlap}")
        if common_profiling_overlap:
            print(f"⚠️  Common-Profiling overlap: {common_profiling_overlap}")
        if distributed_profiling_overlap:
            print(f"⚠️  Distributed-Profiling overlap: {distributed_profiling_overlap}")
            
        if not any([common_distributed_overlap, common_profiling_overlap, distributed_profiling_overlap]):
            print("✅ No redundant function definitions detected")

    def test_all_init_files_valid(self):
        """Test all __init__.py files are valid Python and properly export."""
        init_files = [
            SRC_DIR / "hpc_inference" / "__init__.py",
            SRC_DIR / "hpc_inference" / "utils" / "__init__.py", 
            SRC_DIR / "hpc_inference" / "datasets" / "__init__.py",
            SRC_DIR / "hpc_inference" / "inference" / "__init__.py",
            SRC_DIR / "hpc_inference" / "inference" / "embed" / "__init__.py",
        ]
        
        for init_file in init_files:
            with self.subTest(file=str(init_file)):
                self.assertTrue(init_file.exists(), f"__init__.py missing: {init_file}")
                
                # Check if file is valid Python
                with open(init_file, 'r') as f:
                    content = f.read()
                
                try:
                    compile(content, str(init_file), 'exec')
                    print(f"✅ {init_file.relative_to(PROJECT_ROOT)} is valid Python")
                except SyntaxError as e:
                    self.fail(f"Syntax error in {init_file}: {e}")

    def test_optional_dependencies_handling(self):
        """Test that the package gracefully handles missing optional dependencies."""
        import hpc_inference
        
        # Test feature detection
        features = hpc_inference.list_available_features()
        self.assertIsInstance(features, dict)
        self.assertIn('core', features)
        self.assertTrue(features['core'])  # Core should always be available
        
        print(f"✅ Feature detection works. Available: {features}")

    def test_import_performance(self):
        """Test that imports are reasonably fast (no heavy computation during import)."""
        import time
        import sys
        
        # Clear modules
        modules_to_clear = [mod for mod in sys.modules.keys() 
                           if mod.startswith('hpc_inference')]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        # Time the main import
        start_time = time.time()
        import hpc_inference
        import_time = time.time() - start_time
        
        # Should be fast (< 1 second on reasonable hardware)
        self.assertLess(import_time, 2.0, 
                       f"Import took too long: {import_time:.2f}s")
        print(f"✅ Main package import time: {import_time:.3f}s")


class TestImportStructure(unittest.TestCase):
    """Test the overall import structure and organization."""
    
    def test_public_api_accessibility(self):
        """Test that users can access key functionality through clean imports."""
        # Test that users can do the most common imports
        test_imports = [
            "from hpc_inference import ParquetImageDataset, ImageFolderDataset",
            "from hpc_inference import assign_files_to_rank, format_time",
            "from hpc_inference.utils import profiling",
            "from hpc_inference.datasets import multi_model_collate",
        ]
        
        for import_stmt in test_imports:
            with self.subTest(import_statement=import_stmt):
                try:
                    exec(import_stmt)
                    print(f"✅ {import_stmt}")
                except Exception as e:
                    self.fail(f"Failed to execute: {import_stmt}\nError: {e}")

    def test_namespace_organization(self):
        """Test that the namespace is well organized."""
        import hpc_inference
        
        # Check main exports are properly categorized
        main_exports = set(hpc_inference.__all__)
        
        # Dataset classes
        dataset_classes = {'ParquetImageDataset', 'ImageFolderDataset'}
        utility_functions = {'decode_image', 'save_emb_to_parquet', 'format_time', 'load_config'}
        distributed_functions = {'assign_files_to_rank', 'assign_indices_to_rank', 'get_distributed_info'}
        modules = {'profiling'}
        meta = {'__version__'}
        
        expected_all = dataset_classes | utility_functions | distributed_functions | modules | meta
        
        self.assertEqual(main_exports, expected_all,
                        f"Unexpected exports. Got: {main_exports}, Expected: {expected_all}")
        print("✅ Namespace is well organized")


def run_import_export_tests():
    """Run all import/export tests and provide a summary."""
    print("🧪 Running comprehensive import/export tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImportsExports))
    suite.addTests(loader.loadTestsFromTestCase(TestImportStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📋 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, error in result.failures:
            print(f"   - {test}: {error}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, error in result.errors:
            print(f"   - {test}: {error}")
    
    if result.wasSuccessful():
        print("\n🎉 All import/export tests passed!")
    else:
        print(f"\n⚠️  Some tests failed. Please review and fix the issues.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_import_export_tests()
    sys.exit(0 if success else 1)
