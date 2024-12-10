import pytest
from lightnumpy.python_api import core, gpu_operations, tpu_operations

def test_add():
    assert core.add([1, 2], [3, 4]) == [4, 6]

def test_gpu_add():
    assert gpu_operations.gpu_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]

def test_tpu_add():
    assert tpu_operations.tpu_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
