import numpy as np
import pandas as pd

# -----------------------------
# Inputs (Fiji CSV)
# -----------------------------
BRANCH_INFO_CSV = "Branch information_young37.csv" #fill out your input file name

# -----------------------------
# Outputs
# -----------------------------
VERTICES_OUT = "vertices_3d_young37.out"
EDGES_OUT    = "nodes_to_edges_3d_young37.out"

# -----------------------------
# Settings 
# -----------------------------
ROUND_DECIMALS = 4          # merges near-identical endpoints
THICKNESS_UM   = 2.0        # total slab thickness (z in roughly ±1 um)
SEED           = 0          # makes z reproducible
GRID_SIZE      = 256        # resolution of smooth height field
SMOOTH_SIGMA_PX= 8.0        # smoothness of height field


def gaussian_smooth2d(img, sigma_px: float):
    """Separable Gaussian smoothing (no SciPy required)."""
    if sigma_px <= 0:
        return img
    r = int(max(1, round(3 * sigma_px)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma_px * sigma_px))
    k /= k.sum()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, tmp)
    return out


def vkey(x, y, z):
    return (round(float(x), ROUND_DECIMALS),
            round(float(y), ROUND_DECIMALS),
            round(float(z), ROUND_DECIMALS))


def build_vertices_edges_from_branchinfo(csv_path):
    df = pd.read_csv(csv_path)

    required = ["V1 x", "V1 y", "V1 z", "V2 x", "V2 y", "V2 z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    vmap = {}
    verts = []
    edges = []

    for _, row in df.iterrows():
        k1 = vkey(row["V1 x"], row["V1 y"], row["V1 z"])
        k2 = vkey(row["V2 x"], row["V2 y"], row["V2 z"])

        if k1 not in vmap:
            vmap[k1] = len(verts)
            verts.append(k1)

        if k2 not in vmap:
            vmap[k2] = len(verts)
            verts.append(k2)

        i = vmap[k1]
        j = vmap[k2]
        if i != j:
            edges.append((i, j))

    # Unique undirected edges
    seen = set()
    uniq = []
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        uniq.append((a, b))

    return np.array(verts, dtype=np.float64), uniq


def embed_vertices_in_3d(verts_xyz, thickness_um=2.0, seed=0,
                         grid_size=256, smooth_sigma_px=8.0):
    """
    Assigns z using a smooth random height field over x–y.
    This avoids unrealistic sharp z jumps between neighboring vertices.
    """
    rng = np.random.default_rng(seed)

    xy = verts_xyz[:, :2].copy()
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    span = np.maximum([xmax - xmin, ymax - ymin], 1e-12)
    uv = (xy - [xmin, ymin]) / span  # map into [0,1]^2

    field = rng.normal(0.0, 1.0, size=(grid_size, grid_size))
    field = gaussian_smooth2d(field, smooth_sigma_px)
    field -= field.mean()
    maxabs = np.max(np.abs(field))
    if maxabs > 1e-12:
        field /= maxabs

    # Bilinear sampling
    gx = uv[:, 0] * (grid_size - 1)
    gy = uv[:, 1] * (grid_size - 1)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = np.clip(x0 + 1, 0, grid_size - 1)
    y1 = np.clip(y0 + 1, 0, grid_size - 1)
    tx = gx - x0
    ty = gy - y0

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]
    f0 = f00 * (1 - tx) + f10 * tx
    f1 = f01 * (1 - tx) + f11 * tx
    z_unit = f0 * (1 - ty) + f1 * ty  # ~[-1,1]

    # Center and scale by max absolute value at the vertex samples
    z_unit = z_unit - z_unit.mean()
    maxabs_s = np.max(np.abs(z_unit))
    if maxabs_s > 1e-12:
        z_unit = z_unit / maxabs_s

    z = z_unit * (thickness_um / 2.0)  # target slab thickness
    out = verts_xyz.copy()
    out[:, 2] = z
    return out


def write_vertices(path, verts_xyz):
    with open(path, "w") as f:
        f.write(f"# x y z (from AnalyzeSkeleton Branch information; "
                f"3D embedded; thickness={THICKNESS_UM} um; seed={SEED}; "
                f"round_decimals={ROUND_DECIMALS})\n")
        for x, y, z in verts_xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_edges(path, edges):
    with open(path, "w") as f:
        f.write("# i j (0-based vertex indices)\n")
        for i, j in edges:
            f.write(f"{i} {j}\n")


if __name__ == "__main__":
    verts, edges = build_vertices_edges_from_branchinfo(BRANCH_INFO_CSV)
    verts3d = embed_vertices_in_3d(
        verts,
        thickness_um=THICKNESS_UM,
        seed=SEED,
        grid_size=GRID_SIZE,
        smooth_sigma_px=SMOOTH_SIGMA_PX
    )

    write_vertices(VERTICES_OUT, verts3d)
    write_edges(EDGES_OUT, edges)

    print(f"Wrote {verts3d.shape[0]} vertices -> {VERTICES_OUT}")
    print(f"Wrote {len(edges)} edges -> {EDGES_OUT}")
    print(f"z-range (um): {verts3d[:,2].min():.4f} to {verts3d[:,2].max():.4f}")
