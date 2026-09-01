import csv, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app import create_app
from model import Config, load_baskets, rules_for_config, train

def test_known_rule_metrics():
    counts={('bread',):3,('milk',):2,('bread','milk'):2}
    rules=rules_for_config(counts,4,Config(.1,.1,1,2))
    rule=next(r for r in rules if r['antecedent']==['milk'])
    assert rule['confidence']==1 and round(rule['lift'],3)==1.333

def test_data_contract():
    baskets=load_baskets(max_baskets=100)
    assert len(baskets)==100 and all(len(set(items))==len(items) for _,items in baskets)

def test_pipeline_and_dashboard(tmp_path):
    metrics=train(artifact_dir=tmp_path,max_baskets=700)
    assert metrics['dataset']['baskets']==700
    assert metrics['winner']['rule_count']>0
    assert metrics['research']['evaluated_configurations']>=metrics['research']['audit_space']
    client=create_app(tmp_path).test_client()
    assert client.get('/').status_code==200
    assert client.get('/api/health').get_json()['artifacts_ready'] is True
    payload=client.get('/api/dashboard?min_lift=1.2&item=milk').get_json()
    assert all(r['lift']>=1.2 and 'milk' in (r['antecedent']+r['consequent']).lower() for r in payload['rules'])
