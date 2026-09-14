import os
import torch
import numpy as np
from flask import Blueprint, jsonify, request, render_template
from stable_baselines3 import PPO

from engines.physics_engine import RocketPhysics as RocketPhysics2D
from agent import RocketNetwork
from engines.physics_3d import RocketPhysics as RocketPhysics3D

rocket_bp = Blueprint("rocket", __name__)
device = torch.device("cpu")

# Load 2D PyTorch Model (DQN)
STATE_DIM_2D, ACTION_DIM_2D = 6, 6
MODEL_PATH_2D = "rocket_dqn.pth"
model_2d = RocketNetwork(STATE_DIM_2D, ACTION_DIM_2D).to(device)
if os.path.exists(MODEL_PATH_2D):
    model_2d.load_state_dict(torch.load(MODEL_PATH_2D, map_location=device))
    model_2d.eval()

# Load 3D Stable-Baselines3 Model (PPO)
MODEL_PATH_3D = "rocket_ppo_model"
model_3d = None
if os.path.exists(MODEL_PATH_3D) or os.path.exists(f"{MODEL_PATH_3D}.zip"):
    try:
        model_3d = PPO.load(MODEL_PATH_3D, device="cpu")
    except Exception as e:
        print(f"Error loading 3D PPO model: {e}")

# Persistent engine instances
active_physics_2d = RocketPhysics2D()
active_physics_3d = RocketPhysics3D()

def get_obs_2d(p):
    return np.array([
        p.pos[0] / 10.0, p.pos[1] / 20.0,
        p.vel[0] / 5.0, p.vel[1] / 5.0,
        p.angle / np.pi, p.angular_vel / 2.0
    ], dtype=np.float32)

def get_obs_3d(p):
    target = p.landing_pad
    return np.array([
        (p.pos[0] - target[0]) / 10.0, (p.pos[1] - target[1]) / 20.0, (p.pos[2] - target[2]) / 10.0,
        p.vel[0] / 5.0, p.vel[1] / 5.0, p.vel[2] / 5.0,
        p.angle[0] / np.pi, p.angle[1] / np.pi, p.angle[2] / np.pi,
        p.angular_vel[0] / 2.0, p.angular_vel[1] / 2.0, p.angular_vel[2] / 2.0
    ], dtype=np.float32)

@rocket_bp.route("/rocket")
def rocket_page():
    return render_template("rocket.html")

@rocket_bp.route("/api/rocket/simulate", methods=["POST"])
def simulate_rocket():
    try:
        body = request.get_json() or {}
        dim = body.get("dim", "3d")
        mode = body.get("mode", "manual")
        is_reset = body.get("reset", False)

        if dim == "3d":
            if is_reset:
                active_physics_3d.reset(pos=(
                    np.random.uniform(4.0, 8.0),
                    np.random.uniform(10.0, 14.0),
                    np.random.uniform(-2.0, 2.0)
                ))

            if mode == "ai" and model_3d is not None:
                action, _ = model_3d.predict(get_obs_3d(active_physics_3d), deterministic=False)
                crashed, landed = active_physics_3d.step(action)
                action_name = "STABLE-BASELINES3 PPO AI"
            else:
                manual_action = body.get("action", [0.0, 0.0, 0.0])
                crashed, landed = active_physics_3d.step(np.array(manual_action, dtype=np.float32))
                action_name = "HUMAN MANUAL PILOT"

            frame_data = {
                "dim": "3d",
                "pos": active_physics_3d.pos.tolist(),
                "vel": active_physics_3d.vel.tolist(),
                "angle": active_physics_3d.angle.tolist(),
                "thrusting": bool(active_physics_3d.thrusting),
                "crashed": bool(crashed),
                "landed": bool(landed),
                "action_name": action_name
            }

        else: # 2D Mode
            if is_reset:
                active_physics_2d.reset(pos=(
                    np.random.uniform(3.0, 7.0),
                    np.random.uniform(12.0, 15.0)
                ))

            if mode == "ai":
                obs = torch.from_numpy(get_obs_2d(active_physics_2d)).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = torch.argmax(model_2d(obs)).item()
                crashed, landed = active_physics_2d.step(action)
                action_name = "PYTORCH DQN AI"
            elif mode == "traditional":
                crashed, landed = active_physics_2d.step_traditional()
                action_name = "TRADITIONAL AUTOPILOT"
            else:
                crashed, landed = active_physics_2d.step(int(body.get("action", 0)))
                action_name = "HUMAN MANUAL PILOT"

            frame_data = {
                "dim": "2d",
                "x": float(active_physics_2d.pos[0]),
                "y": float(active_physics_2d.pos[1]),
                "angle": float(active_physics_2d.angle),
                "vy": float(active_physics_2d.vel[1]),
                "thrusting": bool(active_physics_2d.thrusting),
                "crashed": bool(crashed),
                "landed": bool(landed),
                "action_name": action_name,
                "landing_pad": active_physics_2d.landing_pad.tolist()
            }

        return jsonify({"status": "success", "frame": frame_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500