from scripts.review_results import audit_cache


def test_lower_score_duplicate_does_not_earn_a_second_match(capsys):
    entry = {"image": "demo__f0001.png", "ground_truth": [[0, .5, .5, .5, .5]],
             "predictions": [[0, score, [.25, .25, .75, .75]] for score in [1., .4]]}
    audit_cache([entry])
    output = capsys.readouterr().out
    assert "1/1 found, 1.00 false boxes/frame" in output
    assert "mean score=1.000, fraction matched=1.000" in output
    assert "mean score=0.400, fraction matched=0.000" in output


def test_checkpoint_verification_checks_contents(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from scripts import review_results

    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"checkpoint_epoch": 4,
                                 "checkpoint_sha256": hashlib.sha256(b"example").hexdigest()}))
    monkeypatch.setitem(review_results.REPORTS, "ThermalScout-Net", report)
    candidate = tmp_path / "weights.pt"
    candidate.write_bytes(b"example")
    review_results.verify_checkpoint(candidate)
    candidate.write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        review_results.verify_checkpoint(candidate)
