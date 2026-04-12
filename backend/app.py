# =============================================================================
# Telco Customer Churn Prediction
# Script : 6_application.py
# Purpose: Flask web application that serves the churn prediction model.
#          Provides a REST API endpoint (/model) for real-time inference and
#          LIME-based explanations, plus static views for the browser UI.
# Run    : python code/6_application.py
#          Open   http://127.0.0.1:5000/
# License: Apache 2.0
# =============================================================================
from flask import Flask, send_from_directory, request, Response
import random
import os
import json
import numpy as np
from collections import ChainMap
import logging
import subprocess


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def jsonify(obj):
    """Return a JSON Flask Response, handling numpy scalar types."""
    return Response(json.dumps(obj, cls=_NumpyEncoder), mimetype="application/json")

# Set base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

import sys
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.churnexplainer import ExplainedModel


# This reduces the the output to the console window
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


# Load the explained model
em = ExplainedModel.load(model_name="telco_linear")

# Creates an explained version of a partiuclar data point. This is almost exactly the same as the data used in the model serving code.
def explainid(N):
    customer_data = dataid(N)[0]
    customer_data.pop("id")
    customer_data.pop("Churn probability")
    data = em.cast_dct(customer_data)
    probability, explanation = em.explain_dct(data)
    return {
        "data": dict(data),
        "probability": probability,
        "explanation": explanation,
        "id": int(N),
    }


# Gets the rest of the row data for a particular customer.
def dataid(N):
    customer_id = em.data.index.dtype.type(N)
    customer_df = em.data.loc[[customer_id]].reset_index()
    return customer_df.to_dict(orient="records")


# Flask configuration with absolute static path
flask_app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "flask"))


@flask_app.route("/")
def home():
    return "<script> window.location.href = '/flask/table_view.html'</script>"


@flask_app.route("/flask/<path:path>")
def send_file(path):
    return send_from_directory("flask", path)


@flask_app.route("/sample_table")
def sample_table():
    # Reduced sample size to 3 for Vercel serverless timeout limit (10s)
    sample_ids = random.sample(range(1, len(em.data)), 3)
    sample_table = []
    for ids in sample_ids:
        sample_table.append(explainid(str(ids)))
    return jsonify(sample_table)


# Shows the names and all the catagories of the categorical variables.
@flask_app.route("/categories")
def categories():
    return jsonify(
        {feat: dict(enumerate(cats)) for feat, cats in em.categories.items()}
    )


# Shows the names and all the statistical variations of the numerica variables.
@flask_app.route("/stats")
def stats():
    return jsonify(em.stats)


@flask_app.route("/debug")
def debug_info():
    import sys
    import os
    info = {
        "cwd": os.getcwd(),
        "sys_path": sys.path,
        "files_in_root": os.listdir("."),
        "files_in_backend": os.listdir("backend") if os.path.exists("backend") else "backend missing",
        "env": {k: v for k, v in os.environ.items() if "PORT" in k or "VERCEL" in k or "PYTHON" in k}
    }
    return jsonify(info)


@flask_app.route("/model", methods=["POST", "GET"])
def model_prediction():
    req_data = request.get_json() or {}
    args = req_data.get("request", {})
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    
    return jsonify({
        "response": {
            "prediction": {
                "data": dict(data),
                "probability": probability,
                "explanation": explanation
            }
        }
    })

print("App running at http://127.0.0.1:5000/")

# Launch Flask server locally
if __name__ == "__main__":
    flask_app.run(host="127.0.0.1", port=5000, debug=False)
