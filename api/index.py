import sys
import os

# Add the project root and code/ directory to the Python path
# so Flask can find churnexplainer.py and the models/ folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "code")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Import the Flask app from code/app.py
from code.app import flask_app as app
