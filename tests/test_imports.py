import pytest

# Test if all required modules can be imported successfully
def test_imports():
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import lightgbm
        import catboost
        import tensorflow
        import matplotlib
        import seaborn
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
