from scripts.review_results import audit_cache


def test_lower_score_duplicate_does_not_earn_a_second_match(capsys):
    entry = {"image": "demo__f0001.png", "ground_truth": [[0, .5, .5, .5, .5]],
             "predictions": [[0, score, [.25, .25, .75, .75]] for score in [1., .4]]}
    audit_cache([entry])
    output = capsys.readouterr().out
    assert "1/1 found, 1.00 false boxes/frame" in output
    assert "mean score=1.000, fraction matched=1.000" in output
    assert "mean score=0.400, fraction matched=0.000" in output
