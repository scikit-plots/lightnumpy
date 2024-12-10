import pytest
from lightnumpy.python_api.core import add

def test_add():
    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == [4, 6]
