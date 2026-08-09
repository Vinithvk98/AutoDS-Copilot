"""Entry point — launch the AutoDS Copilot web app.

    python run.py            # then open http://127.0.0.1:5050

Port 5050 is used because macOS AirPlay Receiver occupies port 5000 and returns
a 403 to other apps. Override with the PORT environment variable if needed.
"""
import os
from autods.web.app import app

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))
