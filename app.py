import os
from flask import Flask, render_template
from routes.trading_routes import trading_bp
from routes.boids_routes import boids_bp
from routes.motion_routes import motion_bp
from routes.mp3_routes import mp3_bp
from routes.rl_routes import rl_bp


app = Flask(__name__, static_url_path='/assets')

# --- PRESERVING SECURITY SIGNATURE FOR SESSIONS ---
# This unlocks Flask's internal session configuration so your login manager works
app.secret_key = os.urandom(24) 

# Register blueprints WITHOUT URL prefixes
app.register_blueprint(trading_bp)
app.register_blueprint(boids_bp)
app.register_blueprint(motion_bp)
app.register_blueprint(mp3_bp)
app.register_blueprint(rl_bp)


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
    import sys
    host = "0.0.0.0"
    port = 5000

    # Optional: parse host and port from command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    host = args.host
    port = args.port

    app.run(host=host, port=port, debug=True)