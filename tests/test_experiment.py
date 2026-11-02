"""Regression coverage for relocated run registration and confidence filtering."""
import csv
import hashlib
import project_manager as manager
from thermalscout.evaluate_detector import get_counts


def test_relocated_history_is_not_an_additional_run(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'RECORDS', tmp_path)
    run = tmp_path / 'same_run'
    run.mkdir()
    history = run / 'loss_history.csv'
    history.write_text('epoch,train_loss,val_loss\n1,2,3\n')
    row = dict.fromkeys(manager.RUN_FIELDS, '')
    row.update(run_id='same_run', sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
               source_csv='E:/old_location/same_run/loss_history.csv')
    manager.write_csv(tmp_path / 'training_runs.csv', manager.RUN_FIELDS, [row])
    assert manager.register_loss_csv(history) is False
    assert len(manager.read_csv(tmp_path / 'training_runs.csv')) == 1
    # A genuinely different history under the same ID remains recordable.
    history.write_text('epoch,train_loss,val_loss\n1,2,4\n')
    assert manager.register_loss_csv(history) is True


def test_class_filter_keeps_humans_while_removing_low_score_animals():
    entries = [dict(ground_truth=[(0,.5,.5,.2,.2),(1,.5,.5,.2,.2)],
                    predictions=[(0,.8,[.4,.4,.6,.6]),(1,.3,[.4,.4,.6,.6]),
                                 (0,.995,[0,0,.1,.1])])]
    filtered = [dict(ground_truth=e['ground_truth'], predictions=[p for p in e['predictions']
                if p[1] >= (.99 if p[0] == 0 else .25)]) for e in entries]
    assert get_counts(filtered,1)['tp'] == get_counts(entries,1)['tp'] == 1
    assert get_counts(filtered,0)['tp'] == 0
    assert get_counts(filtered,0)['fp'] == 1
