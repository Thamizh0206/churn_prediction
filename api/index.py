import sys
import os

# Get project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Get code directory (previously 6_application was renamed to app.py)
CODE_DIR = os.path.join(ROOT, "code")

# Add both to sys.path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if CODE_DIR not in sys.path:
    # Insert code directory at the front to avoid conflict with built-in 'code' module
    sys.path.insert(0, CODE_DIR)

# Import the Flask app from 'app.py' inside the 'code' folder
# Since CODE_DIR is in sys.path[0], it will find 'app.py' there
from app import flask_app as app
