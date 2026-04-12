import sys
import os

# Project root is the parent directory of this file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# API directory is where index.py and app.py live
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Add both to path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

# Import the Flask app from 'app.py' inside this same 'api' directory
from app import flask_app as app
