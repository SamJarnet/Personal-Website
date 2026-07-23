import os
import torch
import numpy as np
from flask import Blueprint, jsonify, request, render_template

from engines.physics_engine import RocketPhysics
from agent import RocketNetwork

rocket_bp = Blueprint("rocket", __name__)

device = torch.device("cpu")
STATE_DIM, ACTION_DIM = 6, 6
MODEL_PATH = "rocket_dqn.pth"

model = RocketNetwork(STATE_DIM, ACTION_DIM).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

active_physics = RocketPhysics()

def get_observation(physics):
    return np.array([
        physics.pos[0] / 10.0,
        physics.pos[1] / 20.0,
        physics.vel[0] / 5.0,
        physics.vel[1] / 5.0,
        physics.angle / np.pi,
        physics.angular_vel / 2.0
    ], dtype=np.float32)

@rocket_bp.route("/rocket")
def rocket_page():
    return render_template("rocket.html")

@rocket_bp.route("/api/rocket/simulate", methods=["POST"])
def simulate_rocket():
    global active_physics
    try:
        body = request.get_json() or {}
        
        if body.get("reset", False):
            start_y = np.random.uniform(12.0, 15.0)
            start_x = np.random.uniform(3.0, 7.0)
            active_physics.reset(pos=(start_x, start_y))

        mode = body.get("mode", "manual")
        action_name = "IDLE"

        if mode == "ai":
            obs = get_observation(active_physics)
            state_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_t)
                action = torch.argmax(q_values).item()
            crashed, landed = active_physics.step(action)
            action_name = "PYTORCH AI ENGAGED"

        elif mode == "traditional":
            crashed, landed = active_physics.step_traditional()
            action_name = "TRADITIONAL AUTOPILOT"

        else:
            manual_action = int(body.get("action", 0))
            crashed, landed = active_physics.step(manual_action)
            action_name = "HUMAN MANUAL PILOT"

        frame_data = {
            "x": float(active_physics.pos[0]),
            "y": float(active_physics.pos[1]),
            "angle": float(active_physics.angle),
            "vx": float(active_physics.vel[0]),
            "vy": float(active_physics.vel[1]),
            "thrusting": bool(active_physics.thrusting),
            "crashed": crashed,
            "landed": landed,
            "action_name": action_name,
            "landing_pad": [float(active_physics.landing_pad[0]), float(active_physics.landing_pad[1])]
        }

        return jsonify({"status": "success", "frame": frame_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500