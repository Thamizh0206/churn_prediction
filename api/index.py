import sys
import os

# Get project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Get backend directory
BACKEND_DIR = os.path.join(ROOT, "backend")

# Add both to sys.path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import the Flask app from 'app.py' inside the 'backend' folder
from backend.app import flask_app as app
