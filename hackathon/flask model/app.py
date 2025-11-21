import sys
import ultralytics
print(f"✅ Python Executable: {sys.executable}")
print(f"✅ Ultralytics Version: {ultralytics.__version__}")
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import threading
import time
import os
import requests
from ultralytics import YOLO
import torch

# --- Flask App Setup & Secret Key for Sessions ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_session' # Change this to a random string

# --- Dummy Authority Credentials (Replace with a database in a real application) ---
AUTHORITY_USERS = {
    "authority1": "admin123",
    "authority2":"admin234"
}

# --- Configuration ---
VIDEO_SOURCES = {
    "temple_darshan_zone1": r"flask model\videos\crowd6.mp4", 
    "temple_queue_cam1": r"flask model\videos\crowd3.mp4",      
    "temple_queue_cam2": r"flask model\videos\crowd7.mp4",     
    "temple_outer_queue1": r"flask model\videos\crowd8.mp4",
    "temple_flow_gate2": r"flask model\videos\crowd5.mp4"
}

GATE_CAMERA_MAPPING = {
    "gate1": "temple_queue_cam1",
    "gate2": "temple_queue_cam2",
    "gate3": "temple_outer_queue1" 
}

PROCESSING_INTERVAL_SECONDS = 3.0
CAMERA_CONFIGS = {
    "temple_darshan_zone1": { "HIGH_RISK_THRESHOLD": 0.50, "STAMPEDE_THRESHOLD": 0.7, "NORM_DENSITY": 70.0, },
    "temple_queue_cam1": { "DENSITY_HIGH": 200, "MOTION_LOW_CRUSH": 0.1, "HIGH_RISK_THRESHOLD": 0.1, "STAMPEDE_THRESHOLD": 0.25, "NORM_DENSITY": 250.0, },
    "temple_queue_cam2": { "DENSITY_HIGH": 200, "MOTION_LOW_CRUSH": 0.1, "HIGH_RISK_THRESHOLD": 0.1, "STAMPEDE_THRESHOLD": 0.25, "NORM_DENSITY": 250.0, },
    "temple_outer_queue1": { "DENSITY_HIGH": 50, "MOTION_LOW_CRUSH": 0.8, "HIGH_RISK_THRESHOLD": 0.40, "STAMPEDE_THRESHOLD": 0.75, "NORM_DENSITY": 70.0, },
    "temple_flow_gate2": { "DENSITY_HIGH": 50, "MOTION_LOW_CRUSH": 0.8, "HIGH_RISK_THRESHOLD": 0.40, "STAMPEDE_THRESHOLD": 0.75, "NORM_DENSITY": 70.0, }
}

# --- Thread-safe dictionaries ---
output_frames = {}
status_data = {}
locks = {cam_id: threading.Lock() for cam_id in VIDEO_SOURCES}


# --- Alert Function (Unchanged) ---
def send_alert(alert_type, location="N/A", details=""):
    url = "https://webhook.site/9a01f8af-fcbc-4f5b-befd-f14c99539a5f"
    data = {"alert_type": alert_type, "location": location, "details": details, "timestamp": time.ctime()}
    try:
        requests.post(url, json=data)
        print(f"ALERT SENT: {alert_type} at {location}. Details: {details}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")

# --- Detection Logic (Full Original Code Restored) ---
def run_stampede_detection(camera_id, source_path, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{camera_id}] Using device: {device}")
    
    config = CAMERA_CONFIGS.get(camera_id, CAMERA_CONFIGS["temple_flow_gate2"])

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Cannot open video for {camera_id}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 0.033

    ret, prev_frame = cap.read()
    if not ret: return
    prev_frame = cv2.resize(prev_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    last_person_count = 0
    last_avg_motion = 0.0
    last_situation = "Initializing..."
    last_risk_score = 0.0
    last_processing_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 360))
        
        if (time.time() - last_processing_time) > PROCESSING_INTERVAL_SECONDS:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            valid_flow = mag > 0.2
            if np.any(valid_flow):
                avg_motion, std_motion = np.mean(mag[valid_flow]), np.std(mag[valid_flow])
            else:
                avg_motion, std_motion = 0.0, 0.0

            results = model.predict(frame, conf=0.35, iou=0.5, classes=[0], device=device, verbose=False)
            person_count = len(results[0].boxes) if results[0].boxes is not None else 0

            risk_score = 0.0
            if "darshan" in camera_id:
                motion_risk = min(avg_motion / 1.0, 1.0)
                density_risk = min(person_count / config["NORM_DENSITY"], 1.0)
                risk_score = (motion_risk * 0.70) + (density_risk * 0.30)
            elif "queue" in camera_id or "flow" in camera_id:
                norm_avg_motion = min(avg_motion / 5.0, 1.0)
                norm_std_motion = min(std_motion / 3.0, 1.0)
                norm_person_count = min(person_count / config["NORM_DENSITY"], 1.0)
                risk_score = ((norm_avg_motion * 0.20) + ((1 - norm_std_motion) * -0.10) + (norm_person_count * 0.30))
                if person_count > config["DENSITY_HIGH"] and avg_motion < config["MOTION_LOW_CRUSH"]:
                    risk_score += 0.50

            risk_score = max(0, min(risk_score, 1.0))

            current_situation = "Safe"
            if risk_score >= config.get("STAMPEDE_THRESHOLD", 0.9):
                current_situation = "Stampede in Progress"
            elif risk_score >= config.get("HIGH_RISK_THRESHOLD", 0.5):
                current_situation = "High Risk of Stampede"
            elif person_count > (config["NORM_DENSITY"] * 0.4): # <-- This is the new line
                current_situation = "Crowded"
            if current_situation != "Safe" and current_situation != last_situation:
                alert_details = f"People: {person_count}, Avg Motion: {avg_motion:.2f}, Risk: {risk_score:.2f}"
                send_alert(current_situation, location=camera_id.upper(), details=alert_details)

            last_person_count, last_avg_motion, last_risk_score, last_situation = person_count, avg_motion, risk_score, current_situation
            prev_gray = gray
            last_processing_time = time.time()
        
        overlay = frame.copy()
        alert_color = (0, 255, 0)
        if "Stampede in Progress" in last_situation: alert_color = (0, 0, 255)
        elif "High Risk" in last_situation: alert_color = (0, 165, 255)

        cv2.rectangle(overlay, (0, 0), (520, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        cv2.putText(frame, f'Situation: {last_situation}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
        cv2.putText(frame, f'Risk Score: {last_risk_score:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'People: {last_person_count}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Avg Motion: {last_avg_motion:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        with locks[camera_id]:
            output_frames[camera_id] = frame.copy()
            status_data[camera_id] = {
                "location": camera_id.upper(), "situation": last_situation, "person_count": last_person_count,
                "avg_motion": round(float(last_avg_motion), 2), "risk_score": round(float(last_risk_score), 2)
            }
        time.sleep(frame_delay)

# --- Frame Generator (Full Original Code Restored) ---
def generate_frames(camera_id):
    while True:
        with locks[camera_id]:
            frame = output_frames.get(camera_id)
            if frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Initializing...', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
        flag, encodedImage = cv2.imencode(".jpg", frame)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.02)


# ==========================================================
# === NEW & UPDATED FLASK ROUTES WITH LOGIN LOGIC ===
# ==========================================================

@app.route('/')
def welcome():
    # The main page is now the login page
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Check if user exists and password is correct
    if username in AUTHORITY_USERS and AUTHORITY_USERS[username] == password:
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html', error='Invalid credentials. Please try again.')

@app.route('/dashboard')
def dashboard():
    # Protect this route
    if not session.get('logged_in'):
        return redirect(url_for('welcome'))
    return render_template('index.html', camera_ids=list(VIDEO_SOURCES.keys()))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('welcome'))

@app.route('/pilgrim')
def pilgrim_page():
    # This page is public, no login required
    return render_template('pilgrim.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    # Protect this route
    if not session.get('logged_in'):
        return "Access Denied", 401
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status/all')
def status_all():
    # Protect this route
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(status_data)

# --- Add this new route to your app.py ---

@app.route('/api/public_status')
def public_status():
    """
    A public endpoint to provide status data for the pilgrim page.
    This returns the full status dictionary.
    """
    # Using a lock ensures we don't read the data while it's being written
    with locks[list(VIDEO_SOURCES.keys())[0]]:
        return jsonify(status_data)


@app.route('/api/gate_status')
def gate_status():
    # This API is public for the pilgrim page
    gate_statuses = {}
    with locks[list(VIDEO_SOURCES.keys())[0]]:
        for gate, cam_id in GATE_CAMERA_MAPPING.items():
            situation = status_data.get(cam_id, {}).get("situation", "Initializing...")
            gate_statuses[gate] = situation
    return jsonify(gate_statuses)

@app.route('/register', methods=['POST'])
def register():
    username = request.form['new_username']
    password = request.form['new_password']
    email = request.form['email']

        # TEMP: store in session (for hackathon demo)
    session['logged_in'] = True
    session['username'] = username

    print("New User Registered:", username, email)

    # After registering → show Add Camera form in the same page
    return render_template("login.html", show_add_camera=True)
# --- Main (Full Original Code Restored) ---
if __name__ == '__main__':
    print("Loading shared YOLO model...")
    shared_model = YOLO("yolov8n.onnx")

    print("Model loaded.")

    for cam_id, path in VIDEO_SOURCES.items():
        if not os.path.exists(path):
            print(f"WARNING: Video not found for {cam_id} at {path}")
            continue
        thread = threading.Thread(target=run_stampede_detection, args=(cam_id, path, shared_model), daemon=True)
        thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
