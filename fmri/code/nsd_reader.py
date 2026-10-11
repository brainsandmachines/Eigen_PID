"""Read Algonauts responses in their supplied image order; no PCA or scaling."""

from pathlib import Path

import numpy as np


def load_region(subject_folder, region):
    """Return responses and vertex counts, removing only exact duplicate columns."""
    subject_folder = Path(subject_folder)
    hemisphere, name = region.split(":")
    assert hemisphere in {"lh", "rh"}, region
    masks = subject_folder / "roi_masks"
    # Pickled mappings are loaded only from the trusted official dataset.
    mapping = np.load(masks / "mapping_prf-visualrois.npy", allow_pickle=True).item()
    names = {"V1v", "V1d"} if name == "V1" else {name}
    assert names <= set(mapping.values()), name
    ids = [key for key, label in mapping.items() if label in names]
    labels = np.load(masks / f"{hemisphere}.prf-visualrois_challenge_space.npy")
    fmri = np.load(subject_folder / "training_split/training_fmri" / f"{hemisphere}_training_fmri.npy", mmap_mode="r")
    assert fmri.ndim == 2 and labels.shape == (fmri.shape[1],), "ROI/data shape mismatch"
    responses = fmri[:, np.isin(labels, ids)].astype(np.float64)
    assert responses.shape[0] > 1 and responses.shape[1] > 0, region
    assert np.isfinite(responses).all(), f"Nonfinite responses: {region}"
    _, indices = np.unique(responses.T, axis=0, return_index=True)
    counts = dict(vertices=responses.shape[1], retained=len(indices), duplicates=responses.shape[1] - len(indices))
    return responses[:, np.sort(indices)], counts
