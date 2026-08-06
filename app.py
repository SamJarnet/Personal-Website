import os
import sys
from flask import Flask, render_template
from routes.trading_routes import trading_bp
from routes.boids_routes import boids_bp
from routes.motion_routes import motion_bp
from routes.mp3_routes import mp3_bp
from routes.rl_routes import rl_bp
from routes.rocket_routes import rocket_bp

# --- DUAL-PURPOSE PATH ROUTING ---
if getattr(sys, 'frozen', False):
    # 1. Inside the Executable: Use PyInstaller's temporary folder
    base_dir = sys._MEIPASS
    template_folder = os.path.join(base_dir, 'templates')
    static_folder = os.path.join(base_dir, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder, static_url_path='/assets')
else:
    # 2. Normal Website Mode: Keeps your exact original setup
    app = Flask(__name__, static_url_path='/assets')

# --- PRESERVING SECURITY SIGNATURE FOR SESSIONS ---
app.secret_key = os.environ.get("SECRET_KEY")

# Register blueprints
app.register_blueprint(trading_bp)
app.register_blueprint(boids_bp)
app.register_blueprint(motion_bp)
app.register_blueprint(mp3_bp)
app.register_blueprint(rl_bp)
app.register_blueprint(rocket_bp)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Disable debug by default on Pi to save CPU cycles and prevent execution lag
    app.run(host=args.host, port=args.port, debug=args.debug)