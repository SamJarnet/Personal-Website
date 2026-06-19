from flask import Blueprint, jsonify, request, render_template, session
import os
import sqlite3
import bcrypt
from engines.mp3_engines import WebSearchEngine, WebCircularQueue, WebStatsTracker, WebRecommender

mp3_bp = Blueprint("mp3", __name__)

search_engine = WebSearchEngine()
queue_engine = WebCircularQueue()
stats_tracker = WebStatsTracker()
recommender = WebRecommender()

PLAYLIST_DIR = "playlists"
if not os.path.exists(PLAYLIST_DIR):
    os.makedirs(PLAYLIST_DIR)

DB_FILE = "logins.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT NOT NULL,
            password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

def get_current_user():
    return session.get("user", "Guest")

@mp3_bp.route("/music")
def music_page():
    username = session.get("user", "")
    
    # FORCE RESET GUEST MODE: Clear previous guest files when hitting the page fresh
    if not username:
        try:
            guest_master = os.path.join(PLAYLIST_DIR, "playlistsGuest.txt")
            guest_tracks = os.path.join(PLAYLIST_DIR, "Demo PlaylistGuest.txt")
            
            if os.path.exists(guest_master):
                os.remove(guest_master)
            if os.path.exists(guest_tracks):
                os.remove(guest_tracks)
        except Exception as e:
            print(f"Error resetting guest files: {e}")

    return render_template("music.html", username=username)

# --- AUTH SYSTEM ---
@mp3_bp.route("/api/auth/signup", methods=["POST"])
def api_signup():
    body = request.get_json() or {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"status": "error", "msg": "Enter all data."})
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username=?", [username])
    if cursor.fetchone() is not None:
        conn.close()
        return jsonify({"status": "error", "msg": "Username already exists"})
        
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    cursor.execute("INSERT INTO users VALUES (?, ?)", [username, hashed_password])
    conn.commit()
    conn.close()
    
    master_file = os.path.join(PLAYLIST_DIR, f"playlists{username}.txt")
    if not os.path.exists(master_file):
        with open(master_file, "w") as f:
            f.write("")
            
    session["user"] = username
    return jsonify({"status": "success", "msg": "Account created!", "username": username})

@mp3_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    body = request.get_json() or {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", [username])
    result = cursor.fetchone()
    conn.close()
    
    if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
        session["user"] = username
        return jsonify({"status": "success", "msg": "Login successful!", "username": username})
    return jsonify({"status": "error", "msg": "Invalid credentials."})

@mp3_bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.pop("user", None)
    return jsonify({"status": "success"})

# --- PLAYLIST MANAGEMENT ---
@mp3_bp.route("/api/music/playlists", methods=["GET"])
def get_playlists():
    username = get_current_user()
    master_file = os.path.join(PLAYLIST_DIR, f"playlists{username}.txt")
    
    if not os.path.exists(master_file):
        with open(master_file, "w") as f:
            if username == "Guest":
                f.write("Demo Playlist\n")
                demo_tracks = os.path.join(PLAYLIST_DIR, "Demo PlaylistGuest.txt")
                with open(demo_tracks, "w") as dt:
                    dt.write("Starlight Serenade\nMidnight Groove\nCyberpunk Horizon\n")
            else:
                f.write("")
            
    with open(master_file, "r") as f:
        playlists = [line.strip() for line in f if line.strip()]
    return jsonify({"status": "success", "playlists": playlists})

@mp3_bp.route("/api/music/playlists/create", methods=["POST"])
def create_playlist():
    body = request.get_json() or {}
    playlist_name = body.get("playlist_name", "").strip()
    username = get_current_user()
    
    master_file = os.path.join(PLAYLIST_DIR, f"playlists{username}.txt")
    with open(master_file, "a") as f:
        f.write(playlist_name + "\n")
        
    track_file = os.path.join(PLAYLIST_DIR, f"{playlist_name}{username}.txt")
    with open(track_file, "w") as f:
        f.write("")
    return jsonify({"status": "success", "msg": "Playlist created"})

@mp3_bp.route("/api/music/playlists/delete", methods=["POST"])
def delete_playlist():
    body = request.get_json() or {}
    playlist_name = body.get("playlist_name", "").strip()
    username = get_current_user()
    
    master_file = os.path.join(PLAYLIST_DIR, f"playlists{username}.txt")
    if os.path.exists(master_file):
        with open(master_file, "r") as f:
            lines = f.readlines()
        with open(master_file, "w") as f:
            for line in lines:
                if line.strip() != playlist_name:
                    f.write(line)
                    
    track_file = os.path.join(PLAYLIST_DIR, f"{playlist_name}{username}.txt")
    if os.path.exists(track_file):
        os.remove(track_file)
        
    return jsonify({"status": "success", "msg": "Playlist removed"})

@mp3_bp.route("/api/music/playlist-tracks", methods=["POST"])
def get_tracks():
    body = request.get_json() or {}
    name = body.get("playlist_name")
    username = get_current_user()
    
    track_file = os.path.join(PLAYLIST_DIR, f"{name}{username}.txt")
    tracks = []
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            tracks = [line.strip() for line in f if line.strip()]
    return jsonify({"status": "success", "tracks": tracks})

@mp3_bp.route("/api/music/playlist/add-song", methods=["POST"])
def add_song_to_playlist():
    body = request.get_json() or {}
    playlist_name = body.get("playlist_name")
    track_title = body.get("track_title")
    username = get_current_user()
    
    track_file = os.path.join(PLAYLIST_DIR, f"{playlist_name}{username}.txt")
    with open(track_file, "a") as f:
        f.write(track_title + "\n")
    return jsonify({"status": "success"})

@mp3_bp.route("/api/music/playlist/delete-song", methods=["POST"])
def delete_song_from_playlist():
    body = request.get_json() or {}
    playlist_name = body.get("playlist_name")
    track_title = body.get("track_title")
    username = get_current_user()
    
    track_file = os.path.join(PLAYLIST_DIR, f"{playlist_name}{username}.txt")
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            lines = f.readlines()
        with open(track_file, "w") as f:
            for line in lines:
                if line.strip() != track_title:
                    f.write(line)
    return jsonify({"status": "success", "msg": "Track removed"})

@mp3_bp.route("/api/music/search", methods=["POST"])
def api_search():
    body = request.get_json() or {}
    query = body.get("query", "")
    results = search_engine.search_music(query)
    return jsonify({"status": "success", "results": results})

@mp3_bp.route("/api/music/recommend", methods=["POST"])
def api_recommend():
    body = request.get_json() or {}
    playlist_name = body.get("playlist_name")
    username = get_current_user()
    
    track_file = os.path.join(PLAYLIST_DIR, f"{playlist_name}{username}.txt")
    tracks = []
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            tracks = [line.strip() for line in f if line.strip()]
            
    recommendations = recommender.recommend_songs(tracks)
    return jsonify({"status": "success", "recommendations": recommendations})

@mp3_bp.route("/api/music/queue/add", methods=["POST"])
def queue_add():
    body = request.get_json() or {}
    track = body.get("track", "")
    res = queue_engine.enqueue(track)
    return jsonify(res)

@mp3_bp.route("/api/music/queue/remove", methods=["POST"])
def queue_remove():
    res = queue_engine.dequeue()
    return jsonify(res)

@mp3_bp.route("/api/music/stats", methods=["GET", "POST"])
def api_stats():
    if request.method == "POST":
        body = request.get_json() or {}
        action = body.get("action")
        if action == "pause":
            stats_tracker.record_pause()
        elif action == "resume":
            stats_tracker.record_resume()
    return jsonify({"status": "success", "listening_metric": stats_tracker.get_stats_strings()})