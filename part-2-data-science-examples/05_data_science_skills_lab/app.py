"""Small local dashboard and prediction API for the skills lab."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from lab import OUTPUT, discover_skills, run

app = Flask(__name__)


def ensure_artifacts():
    if not (OUTPUT / "manifest.json").exists() or not (OUTPUT / "titanic_pipeline.joblib").exists():
        run()


@app.get("/")
def index():
    ensure_artifacts()
    import json
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    return render_template_string("""
<!doctype html><html><head><meta charset="utf-8"><title>Data Science Skills Lab</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f5f7fb;color:#172033}h1{margin-bottom:4px}.kpis,.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.card{background:white;padding:18px;border-radius:12px;box-shadow:0 2px 12px #18315314}.value{font-size:2rem;font-weight:700;color:#176b87}.tag{font-size:.8rem;color:#576579}.grid{margin-top:20px}</style></head><body>
<h1>Data Science Skills Lab</h1><p>CRISP-DM demonstration across both vendored skill collections using Kaggle Titanic.</p>
<div class="kpis"><div class="card"><div class="value">{{m.skill_count}}</div>skills demonstrated</div><div class="card"><div class="value">{{m.dataset.rows}}</div>passengers</div><div class="card"><div class="value">{{'%.3f'|format(m.model_metrics.roc_auc)}}</div>held-out ROC AUC</div><div class="card"><div class="value">{{'%.1f%%'|format(m.facts.survival_rate*100)}}</div>survival rate</div></div>
<div class="grid">{% for s in m.skills %}<div class="card"><strong>{{s.name}}</strong><div class="tag">{{s.collection}}</div><p>{{s.description}}</p></div>{% endfor %}</div>
</body></html>""", m=manifest)


@app.get("/api/health")
def health():
    ensure_artifacts()
    return jsonify(status="ok", skills=len(discover_skills()))


@app.post("/api/predict")
def predict():
    ensure_artifacts()
    fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    payload = request.get_json(silent=True) or {}
    missing = [field for field in fields if field not in payload]
    if missing:
        return jsonify(error="missing fields", fields=missing), 400
    probability = float(joblib.load(OUTPUT / "titanic_pipeline.joblib").predict_proba(pd.DataFrame([payload]))[0, 1])
    return jsonify(survival_probability=round(probability, 4), threshold=0.5, prediction=int(probability >= 0.5))


if __name__ == "__main__":
    app.run(port=5004, debug=False)
