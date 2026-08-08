#!/usr/bin/env python3
"""Unit checks for the paper forward model, metrics, and analytic baselines."""

from __future__ import annotations

import numpy as np

from evaluation.baselines import (
    galosh_form_upsample,
    guided_filter_upsample,
    joint_bilateral,
    korhonen_upsample,
    wada_ejbf,
)
from evaluation.common import (
    degrade_scene,
    downsample_plane,
    make_scene,
    profile_metrics,
    reconstruct_plane,
)
from evaluation.coedge import (
    DEFAULT_MANIFESTS,
    corpus_complete,
    edge_statistics,
    report_markdown as coedge_report,
    selected_manifests,
)


def _from_alpha(scene, alpha):
    u = scene.side_u[0] + (scene.side_u[1] - scene.side_u[0]) * alpha
    v = scene.side_v[0] + (scene.side_v[1] - scene.side_v[0]) * alpha
    return u.astype(np.float32), v.astype(np.float32)


def test_constant_preservation():
    p = np.full((96, 96), 0.173, np.float32)
    for kernel in ("box", "triangle", "bicubic", "lanczos"):
        for siting in ("left", "center", "topleft"):
            low = downsample_plane(p, kernel, siting)
            assert np.max(np.abs(low - 0.173)) < 1e-6, (kernel, siting)
            for up in ("nearest", "bilinear", "bicubic", "lanczos"):
                out = reconstruct_plane(low, p.shape, up, siting)
                assert np.max(np.abs(out - 0.173)) < 2e-6, (kernel, siting, up)

    y = np.full((96, 96), 0.4, np.float32)
    u = np.full((48, 48), -0.12, np.float32)
    v = np.full((48, 48), 0.22, np.float32)
    methods = (
        joint_bilateral(y, u, v, "left"),
        guided_filter_upsample(y, u, v, "left"),
        wada_ejbf(y, u, v, "left"),
        korhonen_upsample(y, u, v, "left"),
        galosh_form_upsample(y, u, v, "left"),
    )
    for out_u, out_v in methods:
        assert np.max(np.abs(out_u + 0.12)) < 2e-5
        assert np.max(np.abs(out_v - 0.22)) < 2e-5


def test_siting_is_a_separate_factor():
    ramp = np.tile(np.arange(96, dtype=np.float32), (96, 1))
    left = downsample_plane(ramp, "box", "left")
    center = downsample_plane(ramp, "box", "center")
    assert abs(float(left[24, 24] - 47.5)) < 1e-6
    assert abs(float(center[24, 24] - 48.5)) < 1e-6
    assert not np.array_equal(left, center)


def test_metric_specificity():
    scene = make_scene(104, "dev")
    h, w = scene.shape
    xx = np.arange(w, dtype=np.float64)[None, :]

    shifted = np.clip(0.5 + (xx - (scene.edge_x[:, None] + 1.0)), 0.0, 1.0)
    su, sv = _from_alpha(scene, shifted)
    shifted_metrics = profile_metrics(scene, su, sv)
    assert 0.8 < shifted_metrics["phase_px"] < 1.2, shifted_metrics
    assert shifted_metrics["alias_px"] < 0.1, shifted_metrics

    alternating_edge = scene.edge_x + 0.8 * np.where(np.arange(h) % 2 == 0, -1.0, 1.0)
    alternating = np.clip(0.5 + (xx - alternating_edge[:, None]), 0.0, 1.0)
    au, av = _from_alpha(scene, alternating)
    alias_metrics = profile_metrics(scene, au, av)
    assert alias_metrics["alias_px"] > 0.6, alias_metrics
    assert alias_metrics["phase_abs_px"] < 0.15, alias_metrics

    padded = np.pad(scene.alpha.astype(np.float64), ((0, 0), (2, 2)), mode="edge")
    blurred = (
        padded[:, 0:w] + 4 * padded[:, 1:w + 1] + 6 * padded[:, 2:w + 2]
        + 4 * padded[:, 3:w + 3] + padded[:, 4:w + 4]
    ) / 16.0
    bu, bv = _from_alpha(scene, blurred)
    blur_metrics = profile_metrics(scene, bu, bv)
    assert blur_metrics["spread_delta_px"] > 0.4, blur_metrics

    distance = xx - scene.edge_x[:, None]
    ringing = scene.alpha.astype(np.float64).copy()
    ringing[(distance < -0.5) & (distance > -2.5)] -= 0.12
    ringing[(distance > 0.5) & (distance < 2.5)] += 0.12
    ru, rv = _from_alpha(scene, ringing)
    ring_metrics = profile_metrics(scene, ru, rv)
    assert ring_metrics["ringing_mass"] > 0.005, ring_metrics


def test_forward_model_shapes():
    scene = make_scene(1001, "test")
    for kind in ("box", "triangle", "bicubic", "lanczos"):
        y, u, v = degrade_scene(scene, kind, "topleft", bits=10)
        assert y.shape == scene.shape
        assert u.shape == (48, 48)
        assert v.shape == (48, 48)
        assert np.isfinite(u).all() and np.isfinite(v).all()


def test_coedge_metric_specificity():
    yy, xx = np.mgrid[0:96, 0:96]
    del yy
    y = (xx >= 48).astype(np.float64)
    u_coedge = 0.3 * (xx >= 48)
    u_shifted = 0.3 * (xx >= 58)
    v = np.zeros_like(y)
    coedge = edge_statistics(y, u_coedge, v)
    shifted = edge_statistics(y, u_shifted, v)
    assert coedge["chroma_near_luma"] > 0.99, coedge
    assert shifted["chroma_near_luma"] < 0.01, shifted


def test_corpus_split_and_manifest_contract():
    custom = selected_manifests([DEFAULT_MANIFESTS[0]])
    assert custom == [DEFAULT_MANIFESTS[0]]
    assert selected_manifests(None) == list(DEFAULT_MANIFESTS)

    rows = []
    for domain, value in (("animation", 0.8), ("natural", 0.2)):
        for shot in range(30):
            rows.append({
                "domain": domain,
                "split": "test",
                "work_id": f"{domain}_{shot % 10}",
                "shot_id": str(shot),
                "luma_edge_density": value,
                "chroma_edge_density": value,
                "chroma_near_luma": value,
                "luma_near_chroma": value,
                "direction_agreement": value,
            })
    rows.append({
        "domain": "animation",
        "split": "pilot",
        "work_id": "pilot",
        "shot_id": "pilot",
        "luma_edge_density": 0.0,
        "chroma_edge_density": 0.0,
        "chroma_near_luma": 0.0,
        "luma_near_chroma": 0.0,
        "direction_agreement": 0.0,
    })
    assert corpus_complete(rows)
    report = coedge_report(rows)
    assert "1 pilot and 60 test" in report
    assert "| animation | 10 | 30 | 0.8000" in report
    assert "| natural | 10 | 30 | 0.2000" in report
    assert "+0.6000 [+0.6000, +0.6000]" in report
    assert not corpus_complete([row for row in rows if row["domain"] == "animation"])


def main():
    test_constant_preservation()
    test_siting_is_a_separate_factor()
    test_metric_specificity()
    test_forward_model_shapes()
    test_coedge_metric_specificity()
    test_corpus_split_and_manifest_contract()
    print("evaluation protocol tests: OK")


if __name__ == "__main__":
    main()
