import os
from flask import Flask, render_template
from routes.trading_routes import trading_bp
from routes.boids_routes import boids_bp
from routes.motion_routes import motion_bp
from routes.mp3_routes import mp3_bp
from routes.rl_routes import rl_bp
from routes.rocket_routes import rocket_bp
from routes.space_routes import space_bp

app = Flask(__name__, static_url_path='/assets')
app.secret_key = os.environ.get("SECRET_KEY", "portfolio-secret-key")

# Register Blueprints
app.register_blueprint(trading_bp)
app.register_blueprint(boids_bp)
app.register_blueprint(motion_bp)
app.register_blueprint(mp3_bp)
app.register_blueprint(rl_bp)
app.register_blueprint(rocket_bp)
app.register_blueprint(space_bp)

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
    app.run(host="0.0.0.0", port=5000, debug=True)