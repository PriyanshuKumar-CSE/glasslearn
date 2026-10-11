import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.decomposition import PCA as SkPCA

import glasslearn as gl
from glasslearn.cluster import KMeans
from glasslearn.cluster.kmeans.events import CentroidMoved
from glasslearn.viz import (
    PCA,
    Arrows,
    FeaturePair,
    Identity,
    Lines,
    Markers,
    Points,
    make_projection,
    make_view,
    sample_per_group,
)
from glasslearn.viz.backends.plotly_backend import (
    animated_figure,
    build_figure,
    update_figure,
)

X2 = gl.datasets.make_blobs(n_samples=300, centers=4, random_state=0, as_frame=True)
X6, _ = gl.datasets.make_blobs(n_samples=400, n_features=6, centers=3, random_state=1)


# -- projections ----------------------------------------------------------------


def test_identity():
    proj = Identity().fit(X2.to_numpy(), ["a", "b"])
    np.testing.assert_array_equal(proj.transform(X2.to_numpy()), X2.to_numpy())
    assert proj.axis_titles == ("a", "b") and proj.note == ""
    one = Identity().fit(X2.to_numpy()[:, :1], ["a"])
    assert one.transform([[3.0]]).tolist() == [[3.0, 0.0]]
    with pytest.raises(ValueError, match="at most 2 features"):
        Identity().fit(X6, list("abcdef"))


def test_feature_pair():
    names = list("abcdef")
    proj = FeaturePair("d", "b").fit(X6, names)
    np.testing.assert_array_equal(proj.transform(X6), X6[:, [3, 1]])
    back = proj.inverse_transform([[1.0, 2.0]])
    assert back[0, 3] == 1.0 and back[0, 1] == 2.0
    assert back[0, 0] == pytest.approx(X6[:, 0].mean())
    assert proj.axis_titles == ("d", "b")
    assert "other 4 features are hidden" in proj.note
    with pytest.raises(ValueError, match="Unknown feature 'z'"):
        FeaturePair("a", "z").fit(X6, names)


def test_pca_matches_sklearn():
    ours = PCA().fit(X6)
    theirs = SkPCA(n_components=2).fit(X6)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, theirs.explained_variance_ratio_, rtol=1e-10
    )
    np.testing.assert_allclose(np.abs(ours.components_), np.abs(theirs.components_))
    np.testing.assert_allclose(np.abs(ours.transform(X6)), np.abs(theirs.transform(X6)))
    assert ours.axis_titles[0].startswith("PC1 (") and ours.axis_titles[0].endswith(
        "%)"
    )
    assert "keep" in ours.note


def test_pca_round_trip_on_planar_data():
    rng = np.random.RandomState(0)
    plane = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 5)) + 3.0
    proj = PCA().fit(plane)
    np.testing.assert_allclose(proj.inverse_transform(proj.transform(plane)), plane)


def test_pca_signs_are_deterministic():
    a = PCA().fit(X6).components_
    b = PCA().fit(X6[::-1]).components_
    np.testing.assert_allclose(a, b)


def test_make_projection():
    assert isinstance(make_projection("auto", X2.to_numpy(), ["a", "b"]), Identity)
    assert isinstance(make_projection("auto", X6, list("abcdef")), PCA)
    assert isinstance(make_projection(("a", "c"), X6, list("abcdef")), FeaturePair)
    fitted = PCA().fit(X6)
    assert make_projection(fitted, X6, list("abcdef")) is fitted
    with pytest.raises(ValueError, match="projection must be"):
        make_projection("tsne", X6, list("abcdef"))


# -- sampling -------------------------------------------------------------------


def test_sample_per_group():
    groups = np.repeat([0, 1, 2], [900, 90, 10])
    rows = sample_per_group(groups, 100)
    assert len(rows) == 100 and np.all(np.diff(rows) > 0)
    counts = np.bincount(groups[rows], minlength=3)
    assert counts.tolist() == [90, 9, 1]
    np.testing.assert_array_equal(rows, sample_per_group(groups, 100))
    np.testing.assert_array_equal(sample_per_group(groups[:50], 100), np.arange(50))


def test_tiny_groups_are_kept():
    groups = np.array([0] * 1000 + [1, 2])
    counts = np.bincount(groups[sample_per_group(groups, 20)])
    assert counts[1] == 1 and counts[2] == 1


# -- the KMeans view --------------------------------------------------------------


@pytest.fixture(scope="module")
def km():
    return KMeans(n_clusters=4, random_state=0).fit(X2)


def scenes_for(km, projection="auto", **options):
    view = make_view(km.trace_, **options)
    proj = make_projection(projection, km.trace_.X, km.trace_.feature_names)
    return view, [view.render(i, proj) for i in range(len(km.trace_))]


def test_view_layers_are_fixed(km):
    _, scenes = scenes_for(km)
    kinds = [[type(layer) for layer in s.layers] for s in scenes]
    assert all(k == [Lines, Points, Arrows, Markers] for k in kinds)
    assert {(s.x_range, s.y_range) for s in scenes} == {
        (scenes[0].x_range, scenes[0].y_range)
    }
    assert scenes[0].x_title == "feature_0" and scenes[0].y_title == "feature_1"


def test_view_content_follows_the_trace(km):
    _, scenes = scenes_for(km)
    trace = km.trace_
    assert set(scenes[0].layers[1].group.tolist()) == {-1}  # nobody assigned at init
    np.testing.assert_array_equal(scenes[-1].layers[1].group, km.labels_)
    np.testing.assert_allclose(
        np.column_stack([scenes[-1].layers[3].x, scenes[-1].layers[3].y]),
        km.cluster_centers_,
    )
    for step, scene in zip(trace, scenes, strict=True):
        arrows = scene.layers[2]
        assert len(arrows.x0) == 4  # always one per centroid, so moves can glide
        lengths = np.hypot(arrows.x1 - arrows.x0, arrows.y1 - arrows.y0)
        if step.phase == "update":  # 2 features: arrows show the recorded moves
            moves = [m.distance for m in step.events_of(CentroidMoved)]
            np.testing.assert_allclose(lengths, moves, atol=1e-12)
        else:
            assert lengths.max() == 0
        assert scene.caption.endswith(step.phase)
    # Trails keep one point per step; the distinct positions grow over time.
    assert {len(p) for s in scenes for p in s.layers[0].paths} == {len(trace)}
    distinct = [len(np.unique(s.layers[0].paths[0], axis=0)) for s in scenes]
    assert distinct == sorted(distinct) and distinct[0] == 1 and distinct[-1] > 1


def test_view_samples_points_per_cluster(km):
    view, scenes = scenes_for(km, max_points=100)
    assert view.n_shown == 100 and view.n_total == 300
    assert len(scenes[-1].layers[1].x) == 100
    assert set(scenes[-1].layers[1].group.tolist()) == {0, 1, 2, 3}


def test_view_with_pca_and_feature_pair():
    km6 = KMeans(n_clusters=3, random_state=0).fit(X6)
    _, scenes = scenes_for(km6)
    assert scenes[0].x_title.startswith("PC1")
    _, scenes = scenes_for(km6, projection=("x2", "x5"))
    assert (scenes[0].x_title, scenes[0].y_title) == ("x2", "x5")


def test_view_needs_a_recorded_trace():
    with pytest.raises(ValueError, match="record=True"):
        make_view(KMeans(n_clusters=2).fit(X2, record=False).trace_)


# -- plotly backend ---------------------------------------------------------------


def test_figure_updates_in_place_with_fixed_trace_count(km):
    _, scenes = scenes_for(km)
    fig = build_figure(scenes[0], widget=True)
    assert isinstance(fig, go.FigureWidget)
    n = len(fig.data)
    assert n == 4 + 4  # a trail per centroid, points, arrows, centroid halo + core
    seen = set()
    for scene in scenes:
        update_figure(fig, scene)
        assert len(fig.data) == n
        seen.add(tuple(fig.data[-1].x))
    assert len(seen) > 1  # the centroids actually moved


def test_update_rejects_a_different_layout(km):
    _, scenes = scenes_for(km)
    fig = build_figure(scenes[0])
    other = KMeans(n_clusters=2, random_state=0).fit(X2)
    _, other_scenes = scenes_for(other)
    with pytest.raises(ValueError, match="same layers"):
        update_figure(fig, other_scenes[0])


def test_glide_uses_an_animated_update(km, monkeypatch):
    _, scenes = scenes_for(km)
    fig = build_figure(scenes[0], widget=True)
    sent = []
    monkeypatch.setattr(fig, "_send_animate_msg", lambda **kw: sent.append(kw))
    update_figure(fig, scenes[1], glide_ms=300)
    assert len(sent) == 1
    assert sent[0]["animation_opts"]["transition"]["duration"] == 300
    update_figure(fig, scenes[2])  # without glide_ms: a plain in-place update
    assert len(sent) == 1


def test_animated_figure(km):
    _, scenes = scenes_for(km)
    captions = [f"step {i}" for i in range(len(scenes))]
    fig = animated_figure(scenes, captions, title="KMeans")
    assert len(fig.frames) == len(scenes)
    assert len(fig.layout.sliders[0].steps) == len(scenes)
    assert fig.frames[3].layout.annotations[0].text == "step 3"
    assert all(len(f.data) == len(fig.data) for f in fig.frames)
