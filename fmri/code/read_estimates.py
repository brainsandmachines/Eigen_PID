"""Read one real estimate and one shuffled control per participant and target."""

import numpy as np
import pandas as pd


def read_demo_estimates(results):
    data = pd.read_csv(results / "estimates.csv")
    assert (((data.condition == "real") & (data.replicate == 0)) |
            ((data.condition == "shuffled") & (data.replicate == 1))).all()
    keys = ["target", "condition", "method", "component"]
    assert not data.duplicated(["subject", *keys]).any()
    assert np.isfinite(data.bits).all()
    assert set(data.subject) == set(range(1, 9))
    assert set(data.target) == {"lh", "rh"}
    assert set(data.condition) == {"real", "shuffled"}
    assert set(data.method) == {"raw", "combined", "permutation"}
    assert set(data.component) == {"mix", "miy", "mixy", "union_info", "uix", "uiy", "red", "syn"}
    assert (data.groupby(keys).size() == 8).all()
    assert len(data) == 8 * 2 * 2 * 3 * 8
    return data
