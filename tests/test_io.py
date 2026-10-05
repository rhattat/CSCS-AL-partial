import csv
import numpy as np
import pytest
import yaml

from cscs_al_partial.io import (
    load_config,
    get_strategy_config,
    load_embeddings_npy,
    load_train_val_ids,
    compute_proxies_from_embeddings,
    write_run_manifest,
    append_round_log,
    append_provenance_log,
)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def test_load_config_merges_all_sources(tmp_path):
    _write_yaml(tmp_path / "default.yaml", {
        "training": {"fold": 0},
        "cscs": {"clip_min": 0.3, "clip_max": 0.7},
        "seeds": [1, 2, 3],
    })
    _write_yaml(tmp_path / "paths.yaml", {
        "sources": {"toy": {"images": "/path/to/images", "labels": "/path/to/labels",
                             "train_file": "/path/train.txt", "val_file": "/path/val.txt"}},
        "embeddings": {"toy": "/path/to/embeddings"},
        "ssl_features": {"toy": "/path/to/ssl_features"},
    })
    _write_yaml(tmp_path / "toy.yaml", {
        "dataset": {"name": "toy", "id_suffixes": ["_0000"]},
        "budgets": {"cold_start_k": 6, "round_k": 3},
        "dcr_precomputed": 0.5,
    })
    _write_yaml(tmp_path / "strategies.yaml", {
        "strategies": {"cscs_curriculum": {"cold_start": "cscs", "round_policy": "curriculum"}},
    })

    config = load_config("toy", config_dir=str(tmp_path))

    assert config["training"]["fold"] == 0
    assert config["cscs"]["clip_min"] == 0.3
    assert config["dataset"]["name"] == "toy"
    assert config["budgets"]["cold_start_k"] == 6
    assert config["dcr_precomputed"] == 0.5
    assert config["source_images"] == "/path/to/images"
    assert config["embeddings_dir"] == "/path/to/embeddings"
    assert "cscs_curriculum" in config["strategies"]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("toy", config_dir=str(tmp_path))


def test_get_strategy_config_unknown_raises():
    config = {"strategies": {"a": {"x": 1}}}
    with pytest.raises(ValueError):
        get_strategy_config(config, "b")
    assert get_strategy_config(config, "a") == {"x": 1}


def test_load_embeddings_npy_preserves_requested_order(tmp_path):
    for i, vid in enumerate(["v1", "v2", "v3"]):
        np.save(tmp_path / f"{vid}.npy", np.full(4, float(i)))
    embeddings, loaded_ids = load_embeddings_npy(str(tmp_path), volume_ids=["v3", "v1"])
    assert loaded_ids == ["v3", "v1"]
    assert embeddings.shape == (2, 4)
    np.testing.assert_allclose(embeddings[0], 2.0)
    np.testing.assert_allclose(embeddings[1], 0.0)


def test_load_embeddings_npy_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embeddings_npy(str(tmp_path / "nope"))


def test_load_train_val_ids_strips_suffix_and_dedups(tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("v1_0000\nv2_0000\n# comment\nv1_0000\n")
    val_file.write_text("v3_0000\n")
    train_ids, val_ids = load_train_val_ids(str(train_file), str(val_file), suffixes=["_0000"])
    assert train_ids == ["v1", "v2"]
    assert val_ids == ["v3"]


def test_compute_proxies_from_embeddings_shapes():
    rng = np.random.RandomState(0)
    embeddings = rng.normal(size=(10, 8))
    U, T = compute_proxies_from_embeddings(embeddings, k=3)
    assert U.shape == (10,)
    assert T.shape == (10,)
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(T))


def test_write_run_manifest_and_append_logs_roundtrip(tmp_path):
    write_run_manifest(
        run_dir=str(tmp_path), dataset="toy", seed_index=1, strategy="cscs_curriculum",
        cscs_al=True, partial_correction=True, correction_budget=0.2,
    )
    manifest_path = tmp_path / "run_manifest.yaml"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["dataset"] == "toy"
    assert manifest["partial_correction"] is True

    append_round_log(str(tmp_path), {"run_id": "r1", "dataset": "toy", "seed": 1, "round": 1, "gamma_r": 0.5})
    append_round_log(str(tmp_path), {"run_id": "r1", "dataset": "toy", "seed": 1, "round": 2, "gamma_r": 0.55})
    with open(tmp_path / "round_log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["round"] == "2"

    append_provenance_log(str(tmp_path), {
        "run_id": "r1", "dataset": "toy", "seed": 1, "round": 1,
        "n_voxels_human_full": 100, "mean_weight": 0.3,
    })
    with open(tmp_path / "provenance_log.csv") as f:
        prov_rows = list(csv.DictReader(f))
    assert prov_rows[0]["mean_weight"] == "0.3"
