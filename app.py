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
import pathlib
from collections import deque
from datetime import datetime
import smtplib
import ssl
from email.message import EmailMessage

# --- Flask App Setup & Secret Key for Sessions ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_session' # Change this to a random string

# --- Dummy Authority Credentials (Replace with a database in a real application) ---
AUTHORITY_USERS = {
    "authority1": "admin123",
    "authority2":"admin234"
}

import pathlib

# --- Configuration ---
# CCTV → video mapping (already used by dashboard & analytics)
# Map each logical camera to a real video file under the local "videos" folder.
BASE_DIR = pathlib.Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"

VIDEO_SOURCES = {
    # Use your newly uploaded real footage here:
    "temple_darshan_zone1": str(VIDEO_DIR / "crowd6.mp4"),

    # Reuse your crowd clips for the queue / gate cameras:
    "temple_queue_cam1":   str(VIDEO_DIR / "crowd3 (1).mp4"),
    "temple_queue_cam2":   str(VIDEO_DIR / "crowd7.mp4"),
    "temple_outer_queue1": str(VIDEO_DIR / "crowd8.mp4"),
    "temple_flow_gate2":   str(VIDEO_DIR / "crowd5.mp4"),
}

CAMERA_LABELS = {
    "temple_darshan_zone1": "Darshan Hall",
    "temple_queue_cam1": "Gate 1 Queue",
    "temple_queue_cam2": "Gate 2 Queue",
    "temple_outer_queue1": "Outer Queue",
    "temple_flow_gate2": "Gate Flow"
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

# --- Map configuration (generic, can be reused for any venue) ---
# For a new location, you only need to change these coordinates or load them from DB/JSON.
MAP_CONFIG = {
    "center": [12.9718, 77.5949],
    "zoom": 18,
    "gates": [
        {
            "id": "gate1",
            "name": "Gate 1 - Main Entrance",
            "lat": 12.9719,
            "lng": 77.5946,
            "camera_id": "temple_queue_cam1"
        },
        {
            "id": "gate2",
            "name": "Gate 2 - Side Entrance",
            "lat": 12.9722,
            "lng": 77.5950,
            "camera_id": "temple_queue_cam2"
        },
        {
            "id": "gate3",
            "name": "Gate 3 - Outer Queue",
            "lat": 12.9714,
            "lng": 77.5953,
            "camera_id": "temple_outer_queue1"
        }
    ],
    "areas": [
        {
            "id": "darshan",
            "name": "Darshan Hall - Main Temple",
            "lat": 12.9718,
            "lng": 77.5949,
            "radius_m": 35,
            "camera_id": "temple_darshan_zone1"
        }
    ],
    # Each route is attached to one "primary" camera for crowd level;
    # in a full system you can aggregate multiple sources (CCTV, Wi‑Fi, beacons, etc.).
    "routes": [
        {
            "id": "gate1_to_darshan",
            "name": "Gate 1 → Darshan",
            "camera_id": "temple_queue_cam1",
            "points": [
                [12.9719, 77.5946],
                [12.9719, 77.5948],
                [12.9718, 77.5949]
            ]
        },
        {
            "id": "gate2_to_darshan",
            "name": "Gate 2 → Darshan",
            "camera_id": "temple_queue_cam2",
            "points": [
                [12.9722, 77.5950],
                [12.9720, 77.5949],
                [12.9718, 77.5949]
            ]
        },
        {
            "id": "gate3_to_darshan",
            "name": "Gate 3 → Darshan",
            "camera_id": "temple_outer_queue1",
            "points": [
                [12.9714, 77.5953],
                [12.9716, 77.5951],
                [12.9718, 77.5949]
            ]
        }
    ]
}

# --- Thread-safe dictionaries & analytics history ---
output_frames = {}
status_data = {}
frame_locks = {}
status_lock = threading.Lock()
RISK_HISTORY = {}
ALERT_LOG = deque(maxlen=50)
ALERT_CONTACTS = []
stats_tracker = {
    "stampede_alerts": 0,
    "high_risk_events": 0,
    "stampedes_prevented": 0
}
EVENT_LOCATION = {
    "name": "Demo Temple Grounds",
    "address": "Bengaluru, India",
    "lat": 12.9718,
    "lng": 77.5949
}
shared_model = None

SMTP_SETTINGS = {
    "host": os.environ.get("SMTP_HOST"),
    "port": int(os.environ.get("SMTP_PORT", "587")),
    "username": os.environ.get("SMTP_USERNAME"),
    "password": os.environ.get("SMTP_PASSWORD"),
    "sender": os.environ.get("SMTP_FROM", "alerts@indranetra.local"),
    "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() != "false"
}
TEXTBELT_KEY = os.environ.get("TEXTBELT_KEY")


def register_camera_structures(camera_id: str):
    camera_id = camera_id.strip()
    frame_locks[camera_id] = threading.Lock()
    output_frames[camera_id] = None
    with status_lock:
        status_data[camera_id] = {
            "location": camera_id.upper(),
            "situation": "Initializing...",
            "person_count": 0,
            "avg_motion": 0.0,
            "risk_score": 0.0
        }
        RISK_HISTORY[camera_id] = deque(maxlen=120)


def notify_contact(contact, alert_type, location, details):
    channel = (contact.get("channel") or "sms").lower()
    value = contact.get("value", "")
    message = f"[Indra-Netra] {alert_type} at {location}. {details or ''}".strip()

    if not value:
        print("Contact missing value, skipping notification.")
        return False

    try:
        if channel in ("sms", "phone"):
            if not TEXTBELT_KEY:
                print("TEXTBELT_KEY not set, cannot send SMS.")
                return False
            resp = requests.post(
                "https://textbelt.com/text",
                data={
                    "phone": value,
                    "message": message,
                    "key": TEXTBELT_KEY
                },
                timeout=10
            )
            print(f"SMS response for {value}: {resp.text}")
            return resp.ok
        elif channel == "email":
            if not SMTP_SETTINGS["host"]:
                print("SMTP settings missing, cannot send email.")
                return False
            email_msg = EmailMessage()
            email_msg["Subject"] = f"[Indra-Netra] {alert_type}"
            email_msg["From"] = SMTP_SETTINGS["sender"]
            email_msg["To"] = value
            email_msg.set_content(message)

            context = ssl.create_default_context()
            with smtplib.SMTP(SMTP_SETTINGS["host"], SMTP_SETTINGS["port"]) as server:
                if SMTP_SETTINGS["use_tls"]:
                    server.starttls(context=context)
                if SMTP_SETTINGS["username"]:
                    server.login(SMTP_SETTINGS["username"], SMTP_SETTINGS["password"])
                server.send_message(email_msg)
            print(f"E-mail alert sent to {value}")
            return True
        elif channel == "webhook":
            resp = requests.post(value, json={
                "alert_type": alert_type,
                "location": location,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }, timeout=10)
            print(f"Webhook response for {value}: {resp.status_code}")
            return resp.ok
        else:
            print(f"Channel {channel} not supported, fallback to console log.")
            return False
    except Exception as exc:
        print(f"Notification error for {value}: {exc}")
        return False


# --- Alert Function (Unchanged) ---
def send_alert(alert_type, location="N/A", details=""):
    timestamp = datetime.utcnow().isoformat()
    url = "https://webhook.site/9a01f8af-fcbc-4f5b-befd-f14c99539a5f"
    data = {"alert_type": alert_type, "location": location, "details": details, "timestamp": timestamp}
    ALERT_LOG.appendleft(data)
    if "Stampede" in alert_type:
        stats_tracker["stampede_alerts"] += 1
        stats_tracker["stampedes_prevented"] += 1
    elif "High Risk" in alert_type:
        stats_tracker["high_risk_events"] += 1
    for contact in ALERT_CONTACTS:
        notify_contact(contact, alert_type, location, details)
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

    if isinstance(source_path, int) and os.name == 'nt':
        cap = cv2.VideoCapture(source_path, cv2.CAP_DSHOW) # specific fix for Windows local webcams
    else:
        cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Cannot open video for {camera_id}")
        with status_lock:
            if camera_id in status_data:
                status_data[camera_id]["situation"] = "Connection Failed"
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 0.033

    for _ in range(30):
        ret, prev_frame = cap.read()
        if ret and prev_frame is not None:
            break
        time.sleep(0.1)
    
    if not ret or prev_frame is None: 
        print(f"[{camera_id}] Could not read first frame, exiting.")
        with status_lock:
            if camera_id in status_data:
                status_data[camera_id]["situation"] = "Stream Timeout"
        return
    prev_frame = cv2.resize(prev_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    last_person_count = 0
    last_avg_motion = 0.0
    last_situation = "Initializing..."
    last_risk_score = 0.0
    last_processing_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if isinstance(source_path, int) or str(source_path).startswith('http'):
                cap.release()
                time.sleep(0.5)
                if isinstance(source_path, int) and os.name == 'nt':
                    cap = cv2.VideoCapture(source_path, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source_path)
            else:
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
        
        with frame_locks[camera_id]:
            output_frames[camera_id] = frame.copy()
        with status_lock:
            status_data[camera_id] = {
                "location": camera_id.upper(), "situation": last_situation, "person_count": last_person_count,
                "avg_motion": round(float(last_avg_motion), 2), "risk_score": round(float(last_risk_score), 2)
            }
            RISK_HISTORY[camera_id].append({
                "timestamp": datetime.utcnow().isoformat(),
                "risk": round(float(last_risk_score), 3),
                "situation": last_situation
            })
        time.sleep(frame_delay)

# --- Frame Generator (Full Original Code Restored) ---
def generate_frames(camera_id):
    while True:
        with frame_locks[camera_id]:
            frame = output_frames.get(camera_id)
            if frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Initializing...', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
        flag, encodedImage = cv2.imencode(".jpg", frame)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.02)


def start_camera_pipeline(camera_id, source_path, model):
    camera_id = camera_id.strip()
    if not camera_id or camera_id in frame_locks:
        return False
    register_camera_structures(camera_id)
    VIDEO_SOURCES[camera_id] = source_path
    thread = threading.Thread(target=run_stampede_detection, args=(camera_id, source_path, model), daemon=True)
    thread.start()
    return True


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
    return render_template(
        'index.html',
        camera_ids=list(VIDEO_SOURCES.keys()),
        camera_labels=CAMERA_LABELS
    )

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('welcome'))

@app.route('/pilgrim')
def pilgrim_page():
    # This page is public, no login required
    return render_template('pilgrim.html')

@app.route('/map')
def live_map():
    """
    Public, reusable live map page.
    Uses OpenStreetMap + Leaflet and our MAP_CONFIG to render:
      - real gates & areas as markers
      - internal routes as polylines
      - colors based on live crowd level per route
    """
    return render_template(
        'map.html',
        map_center=MAP_CONFIG["center"],
        map_zoom=MAP_CONFIG["zoom"]
    )

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
    with status_lock:
        return jsonify(status_data)

# --- Add this new route to your app.py ---

@app.route('/api/public_status')
def public_status():
    """
    A public endpoint to provide status data for the pilgrim page.
    This returns the full status dictionary.
    """
    # Using a lock ensures we don't read the data while it's being written
    with status_lock:
        return jsonify(status_data)


@app.route('/api/gate_status')
def gate_status():
    # This API is public for the pilgrim page
    gate_statuses = {}
    with status_lock:
        for gate, cam_id in GATE_CAMERA_MAPPING.items():
            situation = status_data.get(cam_id, {}).get("situation", "Initializing...")
            gate_statuses[gate] = situation
    return jsonify(gate_statuses)


@app.route('/api/map_data')
def map_data():
    """
    Aggregated API for the live map.
    Returns gates, areas, and routes with live crowd level
    derived from CCTV analytics (status_data).
    """
    def situation_to_level(situation: str) -> str:
        """
        Map textual situation to a normalized level:
          - low / medium / high / blocked
        """
        if not situation:
            return "unknown"
        s = situation.lower()
        if "stampede in progress" in s:
            return "blocked"
        if "high risk" in s:
            return "high"
        if "crowded" in s:
            return "medium"
        if "safe" in s:
            return "low"
        return "unknown"

    # Build enriched gates
    enriched_gates = []
    for gate in MAP_CONFIG["gates"]:
        cam_id = gate.get("camera_id")
        live = status_data.get(cam_id, {}) if cam_id else {}
        situation = live.get("situation", "Initializing...")
        enriched_gates.append({
            **gate,
            "situation": situation,
            "level": situation_to_level(situation)
        })

    # Build enriched areas
    enriched_areas = []
    for area in MAP_CONFIG["areas"]:
        cam_id = area.get("camera_id")
        live = status_data.get(cam_id, {}) if cam_id else {}
        situation = live.get("situation", "Initializing...")
        enriched_areas.append({
            **area,
            "situation": situation,
            "level": situation_to_level(situation)
        })

    # Build enriched routes
    enriched_routes = []
    for route in MAP_CONFIG["routes"]:
        cam_id = route.get("camera_id")
        live = status_data.get(cam_id, {}) if cam_id else {}
        situation = live.get("situation", "Initializing...")
        enriched_routes.append({
            **route,
            "situation": situation,
            "level": situation_to_level(situation)
        })

    return jsonify({
        "center": MAP_CONFIG["center"],
        "zoom": MAP_CONFIG["zoom"],
        "gates": enriched_gates,
        "areas": enriched_areas,
        "routes": enriched_routes
    })


@app.route('/api/alert_log')
def alert_log():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(list(ALERT_LOG))


@app.route('/api/risk_history')
def risk_history():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    history_payload = {cam_id: list(entries) for cam_id, entries in RISK_HISTORY.items()}
    return jsonify(history_payload)


@app.route('/api/cameras', methods=['GET', 'POST'])
def api_cameras():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'GET':
        data = []
        for cam_id, src in VIDEO_SOURCES.items():
            data.append({
                "id": cam_id,
                "label": CAMERA_LABELS.get(cam_id, cam_id),
                "source": src
            })
        return jsonify(data)

    payload = request.get_json(force=True)
    camera_id = payload.get('camera_id', '').strip().lower().replace(' ', '_')
    label = payload.get('label', camera_id.upper()).strip()
    source = payload.get('source', '').strip()
    if source.isdigit():
        source = int(source)
    elif isinstance(source, str) and source.startswith("http"):
        import urllib.parse
        parsed = urllib.parse.urlparse(source)
        if not parsed.path or parsed.path == "/":
            source = source.rstrip("/") + "/video"
            
    zone_type = payload.get('zone_type', 'open_area').strip().lower()

    if not camera_id or not source:
        return jsonify({"error": "Camera ID and source are required."}), 400
    if camera_id in VIDEO_SOURCES:
        return jsonify({"error": "Camera already exists."}), 400
    if shared_model is None:
        return jsonify({"error": "Model not yet initialized."}), 503

    # Assign appropriate thresholds based on zone type
    ZONE_TYPE_CONFIGS = {
        "darshan": { "HIGH_RISK_THRESHOLD": 0.50, "STAMPEDE_THRESHOLD": 0.7, "NORM_DENSITY": 70.0 },
        "queue": { "DENSITY_HIGH": 200, "MOTION_LOW_CRUSH": 0.1, "HIGH_RISK_THRESHOLD": 0.1, "STAMPEDE_THRESHOLD": 0.25, "NORM_DENSITY": 250.0 },
        "gate": { "DENSITY_HIGH": 50, "MOTION_LOW_CRUSH": 0.8, "HIGH_RISK_THRESHOLD": 0.40, "STAMPEDE_THRESHOLD": 0.75, "NORM_DENSITY": 70.0 },
        "open_area": { "DENSITY_HIGH": 40, "MOTION_LOW_CRUSH": 0.5, "HIGH_RISK_THRESHOLD": 0.45, "STAMPEDE_THRESHOLD": 0.70, "NORM_DENSITY": 60.0 },
    }
    config = ZONE_TYPE_CONFIGS.get(zone_type, ZONE_TYPE_CONFIGS["open_area"])
    CAMERA_CONFIGS[camera_id] = config

    CAMERA_LABELS[camera_id] = label or camera_id.upper()
    register_camera_structures(camera_id)
    VIDEO_SOURCES[camera_id] = source
    thread = threading.Thread(target=run_stampede_detection, args=(camera_id, source, shared_model), daemon=True)
    thread.start()
    return jsonify({"status": "ok", "camera": {"id": camera_id, "label": CAMERA_LABELS[camera_id], "source": source}})


@app.route('/api/test_camera', methods=['POST'])
def api_test_camera():
    """Test if a camera URL/stream can be opened by OpenCV."""
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    payload = request.get_json(force=True)
    source = payload.get('source', '').strip()
    if not source:
        return jsonify({"ok": False, "error": "No source URL provided."}), 400
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str) and source.startswith("http"):
        import urllib.parse
        parsed = urllib.parse.urlparse(source)
        if not parsed.path or parsed.path == "/":
            source = source.rstrip("/") + "/video"
            
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "Could not open the stream. Check the URL and ensure the app is running on your phone."})
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return jsonify({"ok": False, "error": "Stream opened but no frame received. The camera may not be transmitting."})
        h, w = frame.shape[:2]
        return jsonify({"ok": True, "message": f"Connection successful! Resolution: {w}x{h}"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error: {str(e)}"})


@app.route('/api/contacts', methods=['GET', 'POST'])
def api_contacts():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'GET':
        return jsonify(ALERT_CONTACTS)

    payload = request.get_json(force=True)
    name = payload.get('name', 'Responder').strip()
    channel = payload.get('channel', 'sms').strip()
    value = payload.get('value', '').strip()
    if not value:
        return jsonify({"error": "Contact detail is required."}), 400
    ALERT_CONTACTS.append({
        "name": name,
        "channel": channel,
        "value": value
    })
    return jsonify({"status": "ok"})


@app.route('/api/event_location', methods=['GET', 'POST'])
def api_event_location():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'GET':
        return jsonify(EVENT_LOCATION)

    payload = request.get_json(force=True)
    EVENT_LOCATION.update({
        "name": payload.get('name', EVENT_LOCATION["name"]),
        "address": payload.get('address', EVENT_LOCATION.get("address")),
        "lat": float(payload.get('lat', EVENT_LOCATION["lat"])),
        "lng": float(payload.get('lng', EVENT_LOCATION["lng"]))
    })
    return jsonify({"status": "ok", "event": EVENT_LOCATION})


@app.route('/api/system_stats')
def api_system_stats():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    payload = {
        **stats_tracker,
        "active_cameras": len(VIDEO_SOURCES),
        "contacts_count": len(ALERT_CONTACTS),
        "event_location": EVENT_LOCATION
    }
    return jsonify(payload)

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
        register_camera_structures(cam_id)
        if not os.path.exists(path):
            print(f"WARNING: Video not found for {cam_id} at {path}")
            continue
        thread = threading.Thread(target=run_stampede_detection, args=(cam_id, path, shared_model), daemon=True)
        thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
