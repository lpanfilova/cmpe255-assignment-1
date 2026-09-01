import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from model import TASKS, load_task, split_data, train_all


def test_dataset_and_split_contracts():
    for key, spec in TASKS.items():
        frame, loaded = load_task(key)
        train, validation, test = split_data(frame, loaded)
        assert loaded == spec and len(frame) == len(train) + len(validation) + len(test)
        assert frame.isna().sum().sum() == 0
        assert min(len(train), len(validation), len(test)) > 0


def test_dashboard_missing_artifacts(tmp_path):
    client = create_app(tmp_path).test_client()
    assert client.get('/').status_code == 200
    assert client.get('/api/health').get_json()['artifacts_ready'] is False
    assert client.get('/api/dashboard').status_code == 503


@pytest.mark.integration
def test_real_autogluon_training_inference_and_dashboard(tmp_path):
    pytest.importorskip('autogluon.tabular')
    result = train_all(tmp_path, time_limit=6, max_candidates=1)
    assert set(result['tasks']) == set(TASKS)
    assert result['methodology']['holdout_used_for_search'] is False
    assert all(t['leaderboard'] and t['sample_predictions'] for t in result['tasks'].values())
    client = create_app(tmp_path).test_client()
    assert client.get('/api/health').get_json()['artifacts_ready'] is True
    payload = client.get('/api/dashboard').get_json()
    assert payload['tasks']['binary']['test_metrics']['roc_auc'] > .9
    frame, spec = load_task('binary')
    row = frame.drop(columns=spec.label).iloc[0].to_dict()
    response = client.post('/api/predict/binary', json=row)
    assert response.status_code == 200 and len(response.get_json()['predictions']) == 1
