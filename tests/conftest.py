"""Shared fixtures. Tests run offline with no API key, on the bundled sample."""
import pandas as pd
import pytest

from autods.pipeline.loader import load_data, profile
from autods.pipeline.task_detect import detect_task

SAMPLE = "data/sample_customers.csv"


@pytest.fixture(scope="session")
def df():
    return load_data(SAMPLE)


@pytest.fixture(scope="session")
def prof(df):
    return profile(df)


@pytest.fixture(scope="session")
def task(df):
    return detect_task(df, target="churned")
