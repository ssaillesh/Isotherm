import pytest

from config import load_cluster
from physics import Cluster


@pytest.fixture
def spec():
    return load_cluster()


@pytest.fixture
def cluster(spec):
    return Cluster(spec)
