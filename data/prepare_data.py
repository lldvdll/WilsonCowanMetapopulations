"""
Prepare connectivity data for Next Gen NMM simulations.

Sources:
  - SC weights:    HCP 207-subject avg, MRtrix3 + ACT + CSD + SIFT2
                   (ENIGMA Toolbox; Larivière et al. 2021, Nat Methods)
  - FC empirical:  HCP rs-fMRI BOLD, group-averaged
                   (ENIGMA Toolbox)
  - Region labels: Desikan-Killiany 68 cortical regions (DK68;
                   Desikan et al. 2006)
                   (ENIGMA Toolbox; aparc parcellation)
  - Centroids:     **abagen volumetric DK68 atlas in MNI152 space**
                   (Markello et al. 2021, eLife). Mean (x,y,z) of voxels
                   per parcel. ACTIVE source for the current pipeline.
  - Distances:     Pairwise Euclidean distances (mm) between abagen
                   volumetric centroids. Used in modelling with an
                   empirically calibrated scale factor s (calibrated
                   per experiment by the C3 distance-scale sweep).

Centroid choice — abagen volumetric vs ENIGMA surface
=====================================================
This script provides BOTH centroid implementations.

  ACTIVE:  compute_centroids_from_abagen()  → abagen volumetric atlas
           - Voxels per parcel in MNI152 space → mean coordinate
           - Standard choice in connectomics literature (Hagmann 2008,
             Sporns/TVB, abagen-based pipelines)
           - On this dataset and pipeline, gives substantially higher
             BOLD-FC fit than the surface alternative (≈ +0.09 in r at
             matched parameters, pilot calibration)
           - Better represents the distributed neural mass per parcel
             than surface centroids that can fall into deep sulci

  ALTERNATIVE: compute_centroids_from_enigma()  → ENIGMA fsaverage5 surface
           - Mean (x,y,z) of fsaverage5 surface vertices per parcel
           - Single-source consistency with ENIGMA SC/FC (one toolbox)
           - Anatomically purer (cortex is a 2D folded sheet)
           - Use this if cross-toolbox dependency must be avoided

Both produce 68×3 centroids and 68×68 distance matrices in approximately
MNI-aligned mm units. Region ordering matches ENIGMA SC/FC matrices for
both implementations.

The ENIGMA-surface alternative is preserved here for reference and for
sensitivity comparison.
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


def compute_centroids_from_abagen(sc_labels):
    """
    *** ACTIVE centroid source for the current pipeline. ***

    Compute DK68 region centroids from the abagen-packaged Desikan-Killiany
    volumetric NIFTI atlas in MNI152 space.

    For each of the 68 cortical regions, the centroid is the mean (x,y,z)
    coordinate of all voxels carrying that region's label in the volumetric
    atlas. Output is in MNI152 mm and aligned with the ENIGMA SC/FC region
    ordering.

    For DK68 cortical-ribbon regions, mass-weighted volumetric centroids
    represent the distributed neural population better than surface
    vertices that can sit inside deep sulcal folds (e.g. insula).
   
    Caveat: abagen and ENIGMA derive from the same Desikan & Killiany 2006
    atlas in MNI-aligned space, but voxel-vs-vertex parcel boundary
    assignment can differ by ~1–2 mm at sulcal/gyral boundaries. This is
    small relative to the s-scaled inter-regional distance values.
    """
    print("\n=== Step 2: Compute region centroids from abagen volumetric atlas ===")
    import abagen

    # Fetch DK68 volumetric atlas (NIFTI in MNI152 space) packaged by abagen
    atlas = abagen.fetch_desikan_killiany()
    nii_path = atlas['image']
    info_path = atlas['info']
    print(f"  abagen atlas image: {nii_path}")

    import nibabel as nib
    nii = nib.load(nii_path)
    label_volume = np.asarray(nii.get_fdata(), dtype=int)
    affine = nii.affine
    print(f"  volumetric atlas shape: {label_volume.shape}, "
          f"unique labels: {len(np.unique(label_volume)) - 1} (excluding 0)")

    # Read parcel info table (region IDs + names + hemispheres)
    import pandas as pd
    info = pd.read_csv(info_path)
    cortical = info[info['structure'] == 'cortex'].copy()
    cortical = cortical.sort_values(['hemisphere', 'label']).reset_index(drop=True)
    assert len(cortical) == 68, f"Expected 68 cortical parcels, got {len(cortical)}"

    # Compute centroid in MNI152 mm for each cortical parcel,
    # ordered to match ENIGMA SC labels (LH: 0..33, RH: 34..67)
    centroids = np.zeros((68, 3))
    for i, row in cortical.iterrows():
        parcel_id = int(row['id'])
        mask = label_volume == parcel_id
        if mask.sum() == 0:
            raise RuntimeError(f"Parcel id {parcel_id} ({row['label']}) "
                               f"has no voxels in atlas volume")
        # Voxel coordinates → MNI mm via affine
        voxels_ijk = np.argwhere(mask)
        voxels_xyz = nib.affines.apply_affine(affine, voxels_ijk)
        centroids[i] = voxels_xyz.mean(axis=0)

    print(f"  Computed {centroids.shape[0]} region centroids (volumetric)")
    print(f"  Coord range (MNI152 mm): "
          f"x=[{centroids[:,0].min():.1f}, {centroids[:,0].max():.1f}], "
          f"y=[{centroids[:,1].min():.1f}, {centroids[:,1].max():.1f}], "
          f"z=[{centroids[:,2].min():.1f}, {centroids[:,2].max():.1f}]")

    path = os.path.join(DATA_DIR, 'hcp_centroids_68.npy')
    np.save(path, centroids)
    print(f"  Saved: {path}")
    return centroids


def compute_centroids_from_enigma(sc_labels):
    """
    ALTERNATIVE centroid source — ENIGMA fsaverage5 cortical surface.

    Uses ENIGMA Toolbox's bundled fsaverage5 surface meshes (fsa5_lh.gii,
    fsa5_rh.gii) and per-vertex Desikan-Killiany parcellation
    (aparc_fsa5.csv). For each of the 68 cortical regions, the centroid is
    the mean (x, y, z) of all surface vertices belonging to that region in
    fsaverage5 mm space.

    Mapping from numeric aparc labels to ENIGMA SC region order:
      - LH (ENIGMA SC indices  0-33): aparc labels [1,2,3] + [5..35]
        (label 4  = LH corpus callosum, excluded from cortical analysis)
      - RH (ENIGMA SC indices 34-67): aparc labels [36,37,38] + [40..70]
        (label 39 = RH corpus callosum, excluded)
    """
    print("\n=== [alternative] Compute region centroids from ENIGMA fsaverage5 surface ===")
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

    print(f"  Computed {centroids.shape[0]} region centroids (surface)")
    print(f"  Coordinate range (mm): x=[{centroids[:,0].min():.1f}, {centroids[:,0].max():.1f}], "
          f"y=[{centroids[:,1].min():.1f}, {centroids[:,1].max():.1f}], "
          f"z=[{centroids[:,2].min():.1f}, {centroids[:,2].max():.1f}]")

    # NOTE: this function does NOT save to data/hcp_centroids_68.npy by
    # default, so calling it does not overwrite the active abagen-volumetric
    # centroids. Save explicitly if you want to use surface as the active
    # source: np.save(os.path.join(DATA_DIR, 'hcp_centroids_68.npy'), centroids)
    return centroids


def compute_distances(centroids):
    """
    Compute Euclidean distance matrix from region centroids (in mm).

    Euclidean centroid distances under-estimate true white-matter tract
    lengths because real fibers follow curved paths. Modelling scripts
    therefore apply a unit-less distance scale factor s, with
    delay = D × s / v per inter-regional edge. The value of s is treated
    as a free parameter and calibrated empirically per experiment by
    maximising r(simulated BOLD FC, empirical FC) over a sweep
    (typically s ∈ [1.4, 1.7] for our pipeline).
    """
    print("\n=== Step 3: Compute Euclidean distance matrix ===")
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(centroids, metric='euclidean'))
    print(f"  Distance shape: {D.shape}")
    print(f"  Range (non-zero): [{D[D>0].min():.1f}, {D.max():.1f}] mm")
    print(f"  Mean distance (all pairs):  {D[D>0].mean():.1f} mm")

    # For reference: delay magnitudes at v ∈ {6, 12} m/s, no s applied
    for v_ref in (6.0, 12.0):
        delays_ms = (D / v_ref) * 1000.0  # mm / (m/s) → s → ×1000 = ms
        print(f"  Delay at v={v_ref:.0f} m/s (raw, no s): "
              f"min={delays_ms[delays_ms>0].min():.1f}, "
              f"max={delays_ms.max():.1f} ms, "
              f"mean={delays_ms[delays_ms>0].mean():.1f} ms")
    print(f"  NOTE: pipeline applies an empirically calibrated scale factor s; "
          f"distance used in delay = D × s / v.")

    path = os.path.join(DATA_DIR, 'hcp_dist_68.npy')
    np.save(path, D)
    print(f"  Saved: {path}")
    return D


if __name__ == '__main__':
    print("Preparing connectivity data for Next Gen NMM simulations")
    print("=" * 60)

    SC, labels = download_enigma_sc()
    FC = download_enigma_fc()

    # Default: abagen volumetric centroids (active source for current pipeline)
    centroids = compute_centroids_from_abagen(labels)

    # Alternative: ENIGMA fsaverage5 surface centroids (kept for reference;
    # does not save to active path by default — uncomment + save explicitly
    # to switch the active source to surface).
    # _surface_centroids = compute_centroids_from_enigma(labels)

    D = compute_distances(centroids)

    print("\n" + "=" * 60)
    print("Done. Generated files:")
    print(f"  data/hcp_sc_68.npy         — SC weights ({SC.shape}, ENIGMA)")
    print(f"  data/hcp_fc_68.npy         — Empirical FC ({FC.shape}, ENIGMA)")
    print(f"  data/hcp_labels_68.npy     — Region labels ({len(labels)}, ENIGMA)")
    print(f"  data/hcp_centroids_68.npy  — Centroids ({centroids.shape}, abagen volumetric)")
    print(f"  data/hcp_dist_68.npy       — Euclidean distances ({D.shape}, mm)")
