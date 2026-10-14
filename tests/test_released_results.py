import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_released_pool_sizes():
    expected = {"scannet": 820, "eth3d": 223, "bevgsreg": 301}
    for dataset, count in expected.items():
        payload = json.loads((ROOT / "data" / dataset / "evidence.json").read_text())
        assert len(payload["records"]) == count


def test_protocol_is_scene_disjoint():
    protocol = json.loads((ROOT / "data" / "protocol.json").read_text())
    calibration = set(protocol["calibration_pairs"])
    test = set(protocol["scannet_test_pairs"])
    assert len(calibration) == len(test) == 41
    assert calibration.isdisjoint(test)


def test_final_calibration_and_transfer_records_are_released():
    calibration = json.loads((ROOT / "data" / "final_calibration.json").read_text())
    assert calibration["failure_ratio_q95"]
    assert 0.0 < calibration["alpha"] < 1.0
    assert 0.0 <= calibration["affinity_floor"] <= 1.0

    gate = json.loads((ROOT / "data" / "gaussian_conformal_gate.json").read_text())
    assert set(gate["fixed_split"]) == {
        "scannet_heldout", "eth3d_transfer", "bevgsreg_transfer"
    }
