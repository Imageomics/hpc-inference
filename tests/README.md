# Tests Directory

This directory contains all test files for the HPC Inference package.

## Test Files

### Core Import/Export Tests
- **`test_imports_exports.py`** - Comprehensive test suite using unittest framework that validates:
  - All package imports work correctly
  - No circular import dependencies exist
  - All exported functions/classes are accessible
  - Import performance is acceptable
  - Namespace organization is correct

### Quick Validation Tests  
- **`test_package_structure.py`** - Quick validation script that tests basic functionality:
  - Main package imports
  - Dataset classes
  - Utility functions
  - Profiling module
  - Embedding script imports
  - Distributed functions

- **`test_quick_imports.py`** - Fast verification script for development use:
  - Basic package imports
  - Export availability
  - Function redundancy check

### Analysis Tools
- **`final_verification.py`** - Comprehensive package analysis tool that provides:
  - Complete package structure overview
  - Detailed submodule analysis
  - Common usage pattern testing
  - Import consistency verification
  - Redundancy detection

### Legacy Tests
- **`test_imports.py`** - Basic import tests (legacy, but still functional)

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# Or run individual test files
python tests/test_imports_exports.py
python tests/test_package_structure.py
```

### Quick Validation
```bash
# Fast check during development
python tests/test_quick_imports.py
```

### Comprehensive Analysis
```bash
# Detailed package analysis
python tests/final_verification.py
```

## Test Coverage

These tests cover:
- ✅ Import/export functionality
- ✅ Package structure organization
- ✅ Circular import detection
- ✅ Function redundancy checking
- ✅ Performance validation
- ✅ Public API accessibility
- ✅ Cross-module consistency

All tests should pass before committing changes to the package structure.
