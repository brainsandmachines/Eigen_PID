"""Show bilateral V1/hV4 from below and behind, with posterior up.

The fsaverage coordinates are RAS: +x right, +y anterior, +z superior.
With posterior up, anatomical left appears on image left; --anterior-up reverses both.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nilearn import datasets, surface
import numpy as np

from plot_results import figure_style

COLORS = {"Left V1": "#2878B5", "Left hV4": "#7B4FA3",
          "Right V1": "#D78024", "Right hV4": "#29966F"}


def load_surfaces(data, cache, gap):
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage", data_dir=cache)
    regions = {}
    mapping = np.load(data / "subj01/roi_masks/mapping_prf-visualrois.npy", allow_pickle=True).item()
    for hemi, side in (("lh", "Left"), ("rh", "Right")):
        mesh = surface.load_surf_mesh(fsaverage[f"infl_{side.lower()}"])
        sulc = surface.load_surf_data(fsaverage[f"sulc_{side.lower()}"])
        labels = np.load(data / "subj01/roi_masks" / f"{hemi}.prf-visualrois_fsaverage_space.npy")
        vertices, faces = mesh.coordinates.copy(), mesh.faces
        assert len(vertices) == len(labels) == len(sulc) == 163842
        roi = np.zeros(len(labels), dtype=int)
        for value, names in ((1, {"V1v", "V1d"}), (2, {"hV4"})):
            roi[np.isin(labels, [key for key, name in mapping.items() if name in names])] = value
            assert (roi == value).any()
        # Inflated meshes are independently centered; translate only along RAS x.
        shift = -gap / 2 - vertices[:, 0].max() if hemi == "lh" else gap / 2 - vertices[:, 0].min()
        vertices[:, 0] += shift
        face_roi = np.median(roi[faces], axis=1)
        gray = np.where(np.median(sulc[faces], axis=1) > 0, 0.53, 0.80)
        rgb = np.repeat(gray[:, None], 3, axis=1)
        for value, name in ((1, "V1"), (2, "hV4")):
            rgb[face_roi == value] = to_rgb(COLORS[f"{side} {name}"])
        regions[side] = (vertices, faces, rgb)
    assert regions['Left'][0][:, 0].max() < regions['Right'][0][:, 0].min()
    return regions


def illuminated_colors(vertices, faces, rgb, view):
    # Average incident triangle normals for smooth lighting on the inflated mesh.
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros_like(vertices)
    for column in range(3):
        np.add.at(vertex_normals, faces[:, column], normals)
    normals = vertex_normals[faces].mean(axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    assert np.mean(np.sum(normals * (triangles.mean(axis=1) - vertices.mean(axis=0)), axis=1)) > 0
    light = LightSource(azdeg=315, altdeg=-55)
    intensity = light.shade_normals(normals)
    shaded = rgb * (0.35 + 0.65 * intensity[:, None])
    # A modest Blinn–Phong highlight adds gloss without washing out ROI colors.
    halfway = light.direction + view
    halfway /= np.linalg.norm(halfway)
    highlight = 0.28 * np.maximum(normals @ halfway, 0) ** 28
    return shaded * (1 - highlight[:, None]) + highlight[:, None]


def plot(args):
    assert args.gap > 0 and -90 <= args.elevation <= -45
    figure_style()
    regions = load_surfaces(args.data, args.cache, args.gap)
    fig = plt.figure(figsize=(6.75, 3.4))
    ax = fig.add_axes([0.22, 0.16, 0.56, 0.72], projection="3d", proj_type="ortho")
    angle = np.deg2rad(args.elevation)
    view = np.array([0., -np.cos(angle), np.sin(angle)])
    for side, (vertices, faces, rgb) in regions.items():
        print(f"Rendering {side} hemisphere", flush=True)
        collection = Poly3DCollection(vertices[faces], facecolors=illuminated_colors(vertices, faces, rgb, view),
                                      edgecolors="none", linewidths=0, antialiased=False, rasterized=True)
        collection.set_clip_on(False)
        ax.add_collection3d(collection)
    points = np.concatenate([r[0] for r in regions.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low, zoom=1.35)
    roll = 180 if args.anterior_up else 0
    ax.view_init(elev=args.elevation, azim=-90, roll=roll)
    ax.set_axis_off()
    # Check anatomical directions in the actual projection, not just the labels.
    x, y, _ = proj3d.proj_transform([0, 1, 0], [0, 0, 1], [0, 0, 0], ax.get_proj())
    direction = -1 if args.anterior_up else 1
    assert direction * (x[1] - x[0]) > 0 and direction * (y[2] - y[0]) < 0
    top, bottom = ("Anterior (front)", "Posterior (back)") if args.anterior_up else ("Posterior (back)", "Anterior (front)")
    left, right = ("Right", "Left") if args.anterior_up else ("Left", "Right")
    # Measure the projected silhouette; leave the same 6-point gap on all sides.
    ax.apply_aspect()
    px, py, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    projected = fig.transFigure.inverted().transform(ax.transData.transform(np.column_stack((px, py))))
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    gap_x, gap_y = (6 / 72) / fig.get_size_inches()
    center_x, center_y = (lower + upper) / 2
    fig.text(center_x, upper[1] + gap_y, top, ha="center", va="bottom", fontsize=8)
    fig.text(center_x, lower[1] - gap_y, bottom, ha="center", va="top", fontsize=8)
    fig.text(lower[0] - gap_x, center_y, f"{left}\nhemisphere", ha="right", va="center", fontsize=9)
    fig.text(upper[0] + gap_x, center_y, f"{right}\nhemisphere", ha="left", va="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor=color, label=name) for name, color in COLORS.items()],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output / f"cortex.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    settings = dict(subject=1, mesh="fsaverage", surface="inflated", coordinate_system="RAS",
                    elevation=args.elevation, azimuth=-90, roll=roll, hemisphere_gap=args.gap,
                    screen_left=f"anatomical {left.lower()}", screen_top=top.split()[0].lower(), colors=COLORS,
                    vertices_per_hemisphere=163842, label_gap_points=6,
                    lighting=dict(azimuth=315, altitude=-55, specular=0.28, shininess=28))
    (args.output / "cortex_views.json").write_text(json.dumps(settings, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Algonauts training-data directory")
    parser.add_argument("--cache", type=Path, help="Optional Nilearn cache directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "plots")
    parser.add_argument("--gap", type=float, default=8, help="Separation between inflated hemispheres in mesh units")
    parser.add_argument("--elevation", type=float, default=-45, help="-90 is directly below; -45 tilts toward the occipital cortex")
    parser.add_argument("--anterior-up", action="store_true", help="Rotate the view by 180 degrees")
    args = parser.parse_args()
    assert args.data is not None, "Supply --data"
    plot(args)
