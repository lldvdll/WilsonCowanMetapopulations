"""
Prepare connectivity data for Next Gen NMM simulations.

Sources (all from ENIGMA Toolbox; Larivière et al. 2021, Nat Methods):
  - SC weights:     HCP 207-subject average, MRtrix3 + ACT + CSD + SIFT2
  - FC (empirical): HCP resting-state fMRI BOLD, group-averaged
  - Region labels:  Desikan-Killiany atlas, 68 cortical regions (DK68)
  - Centroids:      Computed from ENIGMA's fsaverage5 cortical surface mesh
                    (fsa5_lh.gii + fsa5_rh.gii) using ENIGMA's per-vertex
                    aparc parcellation (aparc_fsa5.csv). Coordinates in
                    fsaverage5 mm space.
  - Distances:      Euclidean centroid distances (mm). Used in modelling with
                    an empirically calibrated scale factor (s = 1.25, see
                    distance scale sweep) to approximate true white-matter
                    pathway curvature.
"""


import os
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_enigma_sc():
    """
    Download HCP structural connectivity from ENIGMA Toolbox.
    Returns DK68 cortico-cortical SC matrix (68x68) and region labels.
    """
    print("=== Step 1: Download SC from ENIGMA Toolbox ===")
    from enigmatoolbox.datasets import load_sc

    SC, labels, _, _ = load_sc(parcellation='aparc')
    print(f"  SC shape: {SC.shape}, non-zero: {np.count_nonzero(SC)}/{SC.size}")
    print(f"  SC range: [{SC.min():.4f}, {SC.max():.4f}]")
    print(f"  Labels: {labels[:3]} ... {labels[-3:]}")

    sc_path = os.path.join(DATA_DIR, 'hcp_sc_68.npy')
    labels_path = os.path.join(DATA_DIR, 'hcp_labels_68.npy')
    np.save(sc_path, SC)
    np.save(labels_path, labels)
    print(f"  Saved: {sc_path}")
    print(f"  Saved: {labels_path}")
    return SC, labels


def download_enigma_fc():
    """
    Download HCP functional connectivity from ENIGMA Toolbox.
    Returns DK68 cortico-cortical FC matrix (68x68).
    """
    print("\n=== Step 1b: Download FC from ENIGMA Toolbox ===")
    from enigmatoolbox.datasets import load_fc

    FC, labels, _, _ = load_fc(parcellation='aparc')
    print(f"  FC shape: {FC.shape}")
    print(f"  FC range: [{FC.min():.4f}, {FC.max():.4f}]")
    print(f"  Symmetric: {np.allclose(FC, FC.T)}")

    fc_path = os.path.join(DATA_DIR, 'hcp_fc_68.npy')
    np.save(fc_path, FC)
    print(f"  Saved: {fc_path}")
    return FC


def compute_centroids_from_enigma(sc_labels):
    """
    Compute DK68 region centroids from ENIGMA's fsaverage5 cortical surface.

    Uses ENIGMA Toolbox's bundled fsaverage5 surface meshes (fsa5_lh.gii,
    fsa5_rh.gii) and per-vertex Desikan-Killiany parcellation
    (aparc_fsa5.csv). For each of the 68 cortical regions, the centroid is
    computed as the mean (x, y, z) coordinate of all surface vertices
    belonging to that region in fsaverage5 mm space.

    Mapping from numeric aparc labels to ENIGMA SC region order:
      - LH (ENIGMA SC indices  0-33): aparc labels [1,2,3] + [5..35]
        (label 4  = LH corpus callosum, excluded from cortical analysis)
      - RH (ENIGMA SC indices 34-67): aparc labels [36,37,38] + [40..70]
        (label 39 = RH corpus callosum, excluded)

    Centroids are fully sourced from ENIGMA Toolbox — same package, same
    atlas, same coordinate system as SC and FC matrices.
    """
    print("\n=== Step 2: Compute region centroids from ENIGMA fsaverage5 surface ===")
    import nibabel as nib
    import enigmatoolbox.datasets as ed

    # Locate ENIGMA's bundled data directory dynamically
    enigma_data_dir = os.path.dirname(ed.__file__)
    surf_dir = os.path.join(enigma_data_dir, 'surfaces')
    parc_dir = os.path.join(enigma_data_dir, 'parcellations')

    # Load fsaverage5 cortical surface vertices (LH then RH)
    lh_pts = nib.load(os.path.join(surf_dir, 'fsa5_lh.gii')).darrays[0].data
    rh_pts = nib.load(os.path.join(surf_dir, 'fsa5_rh.gii')).darrays[0].data
    all_pts = np.vstack([lh_pts, rh_pts])
    print(f"  fsaverage5 surface: LH {lh_pts.shape[0]} + RH {rh_pts.shape[0]} = {all_pts.shape[0]} vertices")

    # Load per-vertex DK parcellation labels (0 = background)
    aparc = np.loadtxt(os.path.join(parc_dir, 'aparc_fsa5.csv'),
                       delimiter=',', dtype=int)
    assert aparc.shape[0] == all_pts.shape[0], \
        f"Parcellation length ({aparc.shape[0]}) != vertex count ({all_pts.shape[0]})"

    # Build aparc label list in ENIGMA SC region order
    LH_LABELS = [1, 2, 3] + list(range(5, 36))      # 34 LH labels
    RH_LABELS = [36, 37, 38] + list(range(40, 71))  # 34 RH labels
    ALL_LABELS = LH_LABELS + RH_LABELS              # 68 total

    # Sanity check against ENIGMA SC labels
    assert len(ALL_LABELS) == 68
    n_L = sum(1 for lbl in sc_labels if lbl.startswith('L_'))
    n_R = sum(1 for lbl in sc_labels if lbl.startswith('R_'))
    assert n_L == 34 and n_R == 34, \
        f"Expected 34L/34R in ENIGMA SC labels, got {n_L}L/{n_R}R"

    # Compute centroid per region in ENIGMA SC order
    centroids = np.zeros((68, 3))
    for i, lbl in enumerate(ALL_LABELS):
        mask = aparc == lbl
        if mask.sum() == 0:
            raise RuntimeError(f"aparc label {lbl} ({sc_labels[i]}) has no vertices")
        centroids[i] = all_pts[mask].mean(axis=0)

    print(f"  Computed {centroids.shape[0]} region centroids")
    print(f"  Coordinate range (mm): x=[{centroids[:,0].min():.1f}, {centroids[:,0].max():.1f}], "
          f"y=[{centroids[:,1].min():.1f}, {centroids[:,1].max():.1f}], "
          f"z=[{centroids[:,2].min():.1f}, {centroids[:,2].max():.1f}]")

    path = os.path.join(DATA_DIR, 'hcp_centroids_68.npy')
    np.save(path, centroids)
    print(f"  Saved: {path}")
    return centroids


def compute_distances(centroids):
    """
    Compute Euclidean distance matrix from region centroids.

    Euclidean centroid distances are an approximation of true white-matter
    tract lengths: real fiber paths follow the curvature of white-matter
    architecture, so true tract lengths are systematically longer than
    straight-line distances.

    To approximate true tract lengths in modelling, the experiment scripts
    apply an empirically calibrated whole-brain scale factor (s = 1.25),
    found by sweeping s and selecting the value that maximises Pearson
    correlation between simulated BOLD FC and empirical fMRI BOLD FC. This
    whole-brain scaling captures the average pathway curvature but does
    not preserve per-edge variation.
    """
    print("\n=== Step 3: Compute Euclidean distance matrix ===")
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(centroids, metric='euclidean'))
    print(f"  Distance shape: {D.shape}")
    print(f"  Range (non-zero): [{D[D>0].min():.1f}, {D.max():.1f}] mm")
    print(f"  Mean distance: {D[D>0].mean():.1f} mm")

    # For reference: delays at v=12 m/s
    delays_ms = D / 12.0  # mm / (m/s) = mm / (mm/ms) = ms
    print(f"  Delay range at v=12 m/s: [{delays_ms[delays_ms>0].min():.1f}, {delays_ms.max():.1f}] ms")
    print(f"  NOTE: These are Euclidean (straight-line) distances.")
    print(f"  Note: experiment scripts apply scale factor s=1.25 (BOLD FC calibrated).")

    path = os.path.join(DATA_DIR, 'hcp_dist_68.npy')
    np.save(path, D)
    print(f"  Saved: {path}")
    return D


if __name__ == '__main__':
    print("Preparing connectivity data for Next Gen NMM simulations")
    print("=" * 60)

    SC, labels = download_enigma_sc()
    FC = download_enigma_fc()
    centroids = compute_centroids_from_enigma(labels)
    D = compute_distances(centroids)

    print("\n" + "=" * 60)
    print("Done. Generated files (all sourced from ENIGMA Toolbox):")
    print(f"  data/hcp_sc_68.npy         — SC weights ({SC.shape})")
    print(f"  data/hcp_fc_68.npy         — Empirical FC ({FC.shape})")
    print(f"  data/hcp_labels_68.npy     — Region labels ({len(labels)})")
    print(f"  data/hcp_centroids_68.npy  — fsaverage5 centroids ({centroids.shape})")
    print(f"  data/hcp_dist_68.npy       — Euclidean distances ({D.shape})")
