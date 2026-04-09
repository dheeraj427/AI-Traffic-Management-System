import streamlit as st
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import threading
from gtts import gTTS
from playsound import playsound
import os
from collections import OrderedDict
import datetime
import sqlite3
import requests
import random
import base64

st.set_page_config(page_title="Smart Traffic Management", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🛑 PERFECT TELEGRAM CREDENTIALS 🛑
# ==========================================
BOT_TOKEN = "8632106244:AAGCpuuxuQlbhxSDUMwE2diWCswU97_iiGE".strip()
CHAT_ID = "-1003817415359".strip()

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect('traffic_system.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS traffic_logs_v2
                 (timestamp TEXT, total_vehicles INTEGER, status TEXT, weather TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (timestamp TEXT, type TEXT, details TEXT)''')
    conn.commit()
    return conn

if "db_conn" not in st.session_state: st.session_state.db_conn = init_db()

def log_to_db(table, data):
    c = st.session_state.db_conn.cursor()
    placeholders = ', '.join(['?'] * len(data))
    c.execute(f"INSERT INTO {table} VALUES ({placeholders})", data)
    st.session_state.db_conn.commit()

# ---------------- TRACKER ----------------
class CentroidTracker:
    def __init__(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            inputCentroids[i] = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)): self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            for i, oc in enumerate(objectCentroids):
                for j, ic in enumerate(inputCentroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set(); usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > 50: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            for col in unusedCols: self.register(inputCentroids[col])
        return self.objects

# ---------------- GLOBAL UI & ANIMATIONS ----------------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.glass-card {
    background: rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 15px; padding: 15px;
    border: 1px solid rgba(255,255,255,0.2); box-shadow: 0 8px 32px rgba(0,0,0,0.3); transition: all 0.4s ease-in-out;
}
.glow-box { transition: 0.3s; border-radius: 12px; }
.glow-box:hover { box-shadow: 0 0 20px rgba(0, 200, 255, 0.7); transform: scale(1.02); }
.pulse { animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 5px green; } 50% { box-shadow: 0 0 25px lime; } 100% { box-shadow: 0 0 5px green; } }
.emergency-flash { animation: alert-flash 0.6s infinite; border: 3px solid red; box-shadow: 0 0 30px red; }
@keyframes alert-flash { 0% { background: rgba(255,0,0,0.1); } 50% { background: rgba(255,0,0,0.4); } 100% { background: rgba(255,0,0,0.1); } }
.loader-wrapper { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh; }
.astra-spinner { width: 100px; height: 100px; border-radius: 50%; border: 6px solid rgba(0, 198, 255, 0.1); border-top-color: #00c6ff; border-bottom-color: lime; animation: spin 1.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite; box-shadow: 0 0 25px rgba(0, 198, 255, 0.4); }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.loader-text { color: #00c6ff; margin-top: 25px; font-family: monospace; font-size: 18px; letter-spacing: 2px; text-transform: uppercase; animation: pulse-text 1.5s infinite; }
@keyframes pulse-text { 0% { opacity: 0.5; } 50% { opacity: 1; text-shadow: 0 0 10px #00c6ff; } 100% { opacity: 0.5; } }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1: st.title("🚦 ASTRA Traffic Management System")
with header_col2: 
    st.write("") 
    audio_enabled = st.toggle("🔊 Enable Audio", value=True)

# ---------------- SETTINGS & STATE ----------------
BASE_GREEN = 60
MIN_GREEN = 10
MAX_GREEN = 120
LANES = ["Lane 1","Lane 2","Lane 3","Lane 4"]
LANE_NAMES = {"Lane 1": "North", "Lane 2": "South", "Lane 3": "East", "Lane 4": "West", "PEDESTRIAN": "Pedestrian Phase"}

if "running" not in st.session_state: st.session_state.running = False
if "traffic_history" not in st.session_state: st.session_state.traffic_history = []
if "full_data_log" not in st.session_state: st.session_state.full_data_log = []
if "lane_durations" not in st.session_state: st.session_state.lane_durations = {lane:BASE_GREEN for lane in LANES}
if "cycle_schedule" not in st.session_state: st.session_state.cycle_schedule = []
if "signal_end_time" not in st.session_state: st.session_state.signal_end_time = 0
if "last_status_time" not in st.session_state: st.session_state.last_status_time = 0
if "tracker" not in st.session_state: st.session_state.tracker = CentroidTracker(maxDisappeared=15)
if "track_history" not in st.session_state: st.session_state.track_history = {}
if "last_telegram_time" not in st.session_state: st.session_state.last_telegram_time = 0
if "siren_active" not in st.session_state: st.session_state.siren_active = False
if "rl_q_table" not in st.session_state: st.session_state.rl_q_table = {"LOW": 0, "MODERATE": 0, "HEAVY": 0, "CRITICAL": 0}

if "siren_trigger_event" not in st.session_state: st.session_state.siren_trigger_event = False
if "siren_end_time" not in st.session_state: st.session_state.siren_end_time = 0
if "last_ui_update" not in st.session_state: st.session_state.last_ui_update = 0

# FIX: Initialize memory with the exact columns YOLO produces so it never crashes!
if "cached_detections" not in st.session_state: 
    st.session_state.cached_detections = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

st.session_state.audio_enabled = audio_enabled

# ---------------- AUDIO FUNCTIONS ----------------
def speak_worker(text):
    try:
        filename = f"voice_{int(time.time() * 1000)}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        playsound(filename)
        if os.path.exists(filename): os.remove(filename)
    except: pass

def speak(text):
    if not st.session_state.audio_enabled: return
    threading.Thread(target=speak_worker, args=(text,), daemon=True).start()

def play_beep_worker():
    try: print('\a')
    except Exception: pass

def play_beep():
    if not st.session_state.audio_enabled: return
    threading.Thread(target=play_beep_worker, daemon=True).start()

def announce_signal(lane, duration):
    speak(f"Astra routing. {LANE_NAMES.get(lane, lane)} green for {duration} seconds.")

# ---------------- ASTRA BOOT-UP SEQUENCE ----------------
if "has_welcomed" not in st.session_state:
    speak("Welcome back, sir. I am Astra. All core systems are initialized and ready. Let's get to work.")
    st.session_state.has_welcomed = True

def add_bgm():
    bgm_file = "bgm.mp3"
    if os.path.exists(bgm_file):
        with open(bgm_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<audio autoplay loop style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', 
            unsafe_allow_html=True
        )

if st.session_state.audio_enabled:
    add_bgm()

# ---------------- AUDIO SIREN SIMULATOR THREAD ----------------
def simulate_siren():
    while True:
        if st.session_state.get('run_siren_sim', False) and st.session_state.running:
            if random.random() < 0.05: 
                st.session_state.siren_trigger_event = True
                time.sleep(12) 
        time.sleep(3)

if "siren_thread" not in st.session_state:
    threading.Thread(target=simulate_siren, daemon=True).start()
    st.session_state.siren_thread = True

# ---------------- LOAD YOLO (WITH SPINNER) ----------------
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5","yolov5s",pretrained=True)
    model.eval()
    return model

with st.spinner("🤖 ASTRA AI Core Initializing... Loading YOLOv5 Neural Network..."):
    model = load_model()
target_classes = ["car","truck","bus","motorcycle","person"] 

# ---------------- TELEGRAM ALERT FUNC ----------------
def send_telegram_alert(message):
    current_time = time.time()
    if current_time - st.session_state.last_telegram_time < 15: return 
    st.session_state.last_telegram_time = current_time 
    st.toast(f"📱 Sending Alert...", icon="🚨")
    try: requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": f"🚨 ASTRA:\n{message}"}, timeout=3)
    except: pass 

# ---------------- TRAFFIC MATH & RL ----------------
def traffic_status(total):
    if total <= 10: return "LOW"
    elif total <= 20: return "MODERATE"
    elif total <= 30: return "HEAVY"
    else: return "CRITICAL"

def calculate_lane_times(lane_counts, rl_active, current_state):
    base_cycle = BASE_GREEN * 4
    total_cars = sum(lane_counts.values())
    durations = {}
    
    if total_cars == 0:
        for lane in LANES: durations[lane] = min(base_cycle // 4, MAX_GREEN)
    else:
        weights = {lane:count/total_cars for lane,count in lane_counts.items()}
        for lane in LANES:
            t = int(base_cycle * weights[lane])
            durations[lane] = max(MIN_GREEN, min(MAX_GREEN, t))
            
    if rl_active and total_cars > 0:
        q_score = st.session_state.rl_q_table[current_state]
        modifier = int(q_score * 2) 
        for lane in LANES:
            durations[lane] = max(MIN_GREEN, durations[lane] + modifier)
            
    return durations

def build_cycle_schedule(durations,start_time):
    schedule=[]
    current=start_time
    for lane in LANES:
        end=current+durations[lane]
        schedule.append({"lane":lane,"start":current,"end":end})
        current=end
    return schedule

def predict_traffic(history):
    if len(history)<5: return history[-1] if history else 0
    X=np.arange(len(history)).reshape(-1,1)
    y=np.array(history)
    model_lr=LinearRegression()
    model_lr.fit(X,y)
    future=np.array([[len(history)+1]])
    return int(max(0, model_lr.predict(future)[0]))

# ---------------- UI DASHBOARD COMPONENTS ----------------
def traffic_lights_with_timers(current_time, schedule, override_active_lane=None):
    cols=st.columns(4)
    lane_wait={}
    active_lane = override_active_lane

    for item in schedule:
        lane=item["lane"]
        if item["start"]<=current_time<item["end"]:
            lane_wait[lane] = int(item["end"]-current_time)
            if not override_active_lane: active_lane = lane
        else:
            lane_wait[lane] = int(item["start"]-current_time) if current_time < item["start"] else 0
            
    if override_active_lane:
        for lane in LANES: lane_wait[lane] = "♾️" if lane == override_active_lane else "🛑"

    for lane,col in zip(LANES,cols):
        with col:
            is_active = (lane == active_lane)
            wrapper_class = "pulse" if is_active else ""
            
            percentage = 100
            if is_active and isinstance(lane_wait[lane], int):
                total_duration = st.session_state.lane_durations.get(lane, 60)
                percentage = max(0, min(100, (lane_wait[lane] / total_duration) * 100))

            if is_active:
                st.markdown(f"""
                <div class="glass-card glow-box fade-in {wrapper_class}">
                <h4 style="text-align:center">{LANE_NAMES.get(lane, lane)}</h4>
                <div style="width:40px;height:40px;background:red;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:lime;border-radius:50%;margin:auto; box-shadow: 0 0 15px lime;"></div>
                <p style="color:lime;text-align:center;font-size:24px; font-weight:bold;">{lane_wait[lane]}s</p>
                <div style="width: 100%; background-color: rgba(255,255,255,0.1); border-radius: 5px; height: 6px;">
                    <div style="width: {percentage}%; height: 6px; background-color: lime; border-radius: 5px; transition: width 0.3s linear;"></div>
                </div>
                </div>
                """,unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card glow-box fade-in shine">
                <h4 style="text-align:center; color: #888;">{LANE_NAMES.get(lane, lane)}</h4>
                <div style="width:40px;height:40px;background:red;border-radius:50%;margin:auto; box-shadow: 0 0 10px red;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <p style="color:#ff4444;text-align:center;font-size:20px;">Wait: {lane_wait[lane]}s</p>
                </div>
                """,unsafe_allow_html=True)

def lane_breakdown_ui(detailed_counts, active_lane=None):
    st.subheader("Lane-wise Vehicle Breakdown")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    def render_card(title, data, is_active=False):
        status = "🟢 ACTIVE" if is_active else "🔴 STOPPED"
        wrapper_class = "pulse" if is_active else ""
        border_color = "rgba(0, 255, 0, 0.4)" if is_active else "rgba(255, 255, 255, 0.1)"

        st.markdown(f"""
        <div class="glass-card glow-box fade-in {wrapper_class}" style="background:#0f172a; margin:10px; border:1px solid {border_color};">
            <h4 style="color:white; margin-bottom: 10px;">{title} <span style="float:right; font-size: 14px;">{status}</span></h4>
            <div style="display: flex; justify-content: space-between; color: #aaa; font-size: 18px; padding: 10px 0;">
                <span style="transition: all 0.3s;">🚗 <b style="color:white">{data['car']}</b></span>
                <span style="transition: all 0.3s;">🏍️ <b style="color:white">{data['motorcycle']}</b></span>
                <span style="transition: all 0.3s;">🚌 <b style="color:white">{data['bus']}</b></span>
                <span style="transition: all 0.3s;">🚚 <b style="color:white">{data['truck']}</b></span>
            </div>
        </div>
        """,unsafe_allow_html=True)

    with col1: render_card("NORTH", detailed_counts["Lane 1"], is_active=("Lane 1" == active_lane))
    with col2: render_card("SOUTH", detailed_counts["Lane 2"], is_active=("Lane 2" == active_lane))
    with col3: render_card("EAST", detailed_counts["Lane 3"], is_active=("Lane 3" == active_lane))
    with col4: render_card("WEST", detailed_counts["Lane 4"], is_active=("Lane 4" == active_lane))

# ==========================================
# 🛑 SIDEBAR NAVIGATION 🛑
# ==========================================
with st.sidebar:
    app_mode = st.radio("🛰️ ASTRA Navigation", ["Live AI Feed", "Data Analytics"])
    st.markdown("---")
    
    if app_mode == "Live AI Feed":
        st.header("⚙️ AI Core Controls")
        st.caption("⚠️ Enable toggles BEFORE starting camera.")
        
        st.subheader("🧠 Q-Learning Mode")
        rl_enabled = st.toggle("Enable Auto-Learning", value=False)
        if rl_enabled: st.write(f"Q-Scores: {st.session_state.rl_q_table}")
        
        st.subheader("🚑 Audio AI")
        st.session_state.run_siren_sim = st.toggle("Listen for Sirens", value=False)
        
        st.subheader("📱 Alerts")
        if st.button("🧪 Send Test Alert"):
            st.toast("📱 Sending Test Alert...", icon="🚨")
            try:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": "🚨 ASTRA ONLINE: Connection Verified."}, timeout=5)
                st.success("Test signal sent!")
            except Exception as e: st.error("Network error.")
            
    elif app_mode == "Data Analytics":
        st.header("📊 Database Export")
        df_export = pd.DataFrame(st.session_state.full_data_log)
        if not df_export.empty:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download CSV", data=csv, file_name=f"traffic_report_{int(time.time())}.csv", mime="text/csv")

# ==========================================
# 🛑 PAGE 1: LIVE AI FEED 🛑
# ==========================================
if app_mode == "Live AI Feed":
    
    # 🛑 THE IP ADDRESS HAS BEEN UPDATED RIGHT HERE:
    camera_url = st.text_input("Camera URL (DroidCam) or 0 for webcam", "http://10.154.221.57:4747/video")
    
    col1,col2 = st.columns(2)

    if col1.button("Start Camera"): st.session_state.running = True
    if col2.button("Stop Camera"):
        st.session_state.running = False
        if "cap" in st.session_state:
            st.session_state.cap.release()
            del st.session_state["cap"]

    alert_placeholder = st.empty()
    env_indicator = st.empty()
    video_placeholder = st.empty()
    graph_placeholder = st.empty()
    lights_placeholder = st.empty()
    status_panel = st.empty()
    lane_placeholder = st.empty()
    history_placeholder = st.empty()
    metric_placeholder = st.empty()

    frame_counter=0
    lane_counts={lane:0 for lane in LANES}

    if not st.session_state.running:
        video_placeholder.markdown("""
            <div class="loader-wrapper">
                <div class="astra-spinner"></div>
                <div class="loader-text">SYSTEM STAND BY<br><span style="font-size: 12px; color: #aaa; text-transform: none;">Waiting for Video Feed...</span></div>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.running:

        if "cap" not in st.session_state:
            # FIX: Properly parse digit inputs in case you switch to standard webcam indexes (0, 1, 2)
            cam_src = int(camera_url) if camera_url.isdigit() else camera_url
            st.session_state.cap = cv2.VideoCapture(cam_src)

        while st.session_state.running:
            ret,frame = st.session_state.cap.read()

            if not ret:
                st.warning("Camera disconnected, reconnecting...")
                st.session_state.cap.release()
                time.sleep(1)
                
                # Re-check the URL in case it was modified
                cam_src = int(camera_url) if camera_url.isdigit() else camera_url
                st.session_state.cap = cv2.VideoCapture(cam_src)
                continue

            frame=cv2.resize(frame,(320,320))
            frame_counter+=1
            current_time = time.time()

            # ---------------- SAFE SIREN TRIGGER HANDLING ----------------
            if st.session_state.siren_trigger_event:
                st.session_state.siren_active = True
                st.session_state.siren_end_time = current_time + 12
                st.session_state.siren_trigger_event = False
                send_telegram_alert("🚑 AMBULANCE SIREN DETECTED! Initiating Emergency Preemption.")
                speak("Emergency vehicle detected. Clearing intersection.")
                
            if st.session_state.siren_active and current_time > st.session_state.siren_end_time:
                st.session_state.siren_active = False

            # ---------------- WEATHER & FOG AI ----------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_foggy = laplacian_var < 50
            
            yolo_conf = 0.35
            weather_status = "Clear"
            
            if is_foggy:
                weather_status = "Fog/Rain"
                yolo_conf = 0.25 
                env_indicator.warning("🌫️ Bad Weather Detected: Lowering YOLO confidence threshold & engaging CLAHE.")
                if frame_counter % 100 == 0: st.snow() 
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                cl = clahe.apply(l_channel)
                frame = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
            else:
                env_indicator.empty()

            # ---------------- AI FRAME CACHING (SPEED FIX) ----------------
            if frame_counter % 3 == 1:
                with torch.no_grad(): results=model(frame)
                df=results.pandas().xyxy[0]
                df=df[df["confidence"]>yolo_conf]
                st.session_state.cached_detections = df[df["name"].isin(target_classes)]
                
            detections = st.session_state.cached_detections

            # ---------------- SCHEDULING & EMERGENCY ----------------
            active_lane = None
            forced_lane = None
            
            if st.session_state.siren_active:
                forced_lane = "EMERGENCY"
                active_lane = "EMERGENCY"
                
            elif len(detections[detections['name'] == 'person']) >= 5:
                forced_lane = "PEDESTRIAN" 
                active_lane = "PEDESTRIAN"

            if not forced_lane:
                for item in st.session_state.cycle_schedule:
                    if item["start"] <= current_time < item["end"]:
                        active_lane = item["lane"]
                        break

                if current_time>=st.session_state.signal_end_time:
                    current_state = traffic_status(sum(lane_counts.values()))
                    
                    if rl_enabled and sum(lane_counts.values()) > 0:
                        reward = 1 if current_state in ["LOW", "MODERATE"] else -1
                        st.session_state.rl_q_table[current_state] = round(st.session_state.rl_q_table[current_state] + 0.1 * reward, 2)

                    st.session_state.lane_durations = calculate_lane_times(lane_counts, rl_enabled, current_state)
                    st.session_state.cycle_schedule = build_cycle_schedule(st.session_state.lane_durations, current_time)
                    st.session_state.signal_end_time = st.session_state.cycle_schedule[-1]["end"]
                    announce_signal(st.session_state.cycle_schedule[0]["lane"], st.session_state.lane_durations[st.session_state.cycle_schedule[0]["lane"]])
            else:
                time_shift = current_time - st.session_state.cycle_schedule[0]["start"] if st.session_state.cycle_schedule else 0
                if time_shift > 0:
                    for item in st.session_state.cycle_schedule:
                        item["start"] += 0.1 ; item["end"] += 0.1
                    st.session_state.signal_end_time += 0.1

            # ---------------- TRACKER & VIOLATIONS ----------------
            rects = []
            for _, det in detections[detections['name'] != 'person'].iterrows():
                rects.append((int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)))
            
            objects = st.session_state.tracker.update(rects)
            annotated = frame.copy()

            cv2.rectangle(annotated, (100, 100), (220, 220), (0, 255, 255), 2)
            cv2.putText(annotated, "INTERSECTION ZONE", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            for (objectID, centroid) in objects.items():
                cx, cy = centroid[0], centroid[1]
                
                if objectID not in st.session_state.track_history:
                    st.session_state.track_history[objectID] = []
                st.session_state.track_history[objectID].append((cx, cy, current_time))
                if len(st.session_state.track_history[objectID]) > 20:
                    st.session_state.track_history[objectID].pop(0)

                history = st.session_state.track_history[objectID]
                is_anomaly = False
                if len(history) == 20:
                    sx, sy, stime = history[0]
                    dist = np.hypot(cx - sx, cy - sy)
                    if dist < 10 and (current_time - stime) > 3.0:
                        if 100 < cx < 220 and 100 < cy < 220: 
                            is_anomaly = True
                
                is_violation = False
                if 100 < cx < 220 and 100 < cy < 220 and active_lane != "None":
                    if len(history) > 5:
                        prev_x, prev_y, _ = history[-5]
                        origin_lane = None
                        if prev_y < 100 and 100 < prev_x < 160: origin_lane = "Lane 1"
                        elif prev_y > 220 and 160 < prev_x < 220: origin_lane = "Lane 2"
                        elif prev_x < 100 and 160 < prev_y < 220: origin_lane = "Lane 3"
                        elif prev_x > 220 and 100 < prev_y < 160: origin_lane = "Lane 4"

                        if origin_lane and origin_lane != active_lane and active_lane != "PEDESTRIAN":
                            is_violation = True

                color = (0, 255, 0)
                if is_anomaly: 
                    color = (0, 0, 255)
                    cv2.putText(annotated, "WARNING: STATIONARY", (cx-30, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    if frame_counter % 30 == 0:
                        log_to_db('violations', (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Anomaly', f'Obj {objectID} stopped'))
                        send_telegram_alert(f"Accident/Breakdown suspected! Vehicle ID: {objectID} stopped in intersection.")
                        
                elif is_violation:
                    color = (255, 0, 0)
                    cv2.putText(annotated, "VIOLATION!", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    if frame_counter % 30 == 0:
                        log_to_db('violations', (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Red Light', f'Obj {objectID} ran red light'))
                        cv2.imwrite(f"violation_obj_{objectID}.jpg", annotated) 
                        send_telegram_alert(f"🚨 RED LIGHT VIOLATION! Vehicle ID {objectID} ran the intersection.")

                text = f"ID {objectID}"
                cv2.putText(annotated, text, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                
            for _, det in detections.iterrows():
                cv2.rectangle(annotated, (int(det.xmin), int(det.ymin)), (int(det.xmax), int(det.ymax)), (0, 255, 255), 2)
                
            # ---------------- HIGH SPEED OPENCV HARDWARE SCANNER ----------------
            scan_y = (frame_counter * 12) % 320
            cv2.line(annotated, (0, scan_y), (320, scan_y), (0, 255, 255), 2)
            cv2.line(annotated, (0, scan_y-1), (320, scan_y-1), (0, 150, 150), 1)
            
            video_placeholder.image(annotated, channels="BGR", use_container_width=True)

            # ---------------- REGION COUNTING ----------------
            h,w,_=frame.shape
            lanes_region = {"Lane 1":(0,0,w//2,h//2), "Lane 2":(w//2,0,w,h//2), "Lane 3":(0,h//2,w//2,h), "Lane 4":(w//2,h//2,w,h)}
            for k in lane_counts: lane_counts[k]=0
            detailed_counts = {lane: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0} for lane in LANES}
            
            for _,det in detections.iterrows():
                cx=(int(det.xmin)+int(det.xmax))//2
                cy=(int(det.ymin)+int(det.ymax))//2
                v_class = det["name"]
                for lane,(lx1,ly1,lx2,ly2) in lanes_region.items():
                    if lx1<=cx<=lx2 and ly1<=cy<=ly2: 
                        lane_counts[lane]+=1
                        if v_class in detailed_counts[lane]:
                            detailed_counts[lane][v_class] += 1
            
            # ---------------- UI THROTTLING (SPEED FIX) ----------------
            if current_time - st.session_state.last_ui_update >= 1.0:
                st.session_state.last_ui_update = current_time
                
                total=sum(lane_counts.values())
                level = traffic_status(total)

                # 1. Bar Chart
                fig,ax=plt.subplots()
                ax.bar(list(lane_counts.keys()), list(lane_counts.values()), color='#00c6ff')
                ax.set_title("Traffic Density", color='white')
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors='white')
                graph_placeholder.pyplot(fig, clear_figure=True)
                plt.close(fig) 

                # 2. Lights
                with lights_placeholder.container():
                    traffic_lights_with_timers(current_time, st.session_state.cycle_schedule, override_active_lane=forced_lane)
                
                # 3. Status Panel
                with status_panel.container():
                    flash_class = "emergency-flash" if forced_lane == "EMERGENCY" else ""
                    alert_color = "#ff4b4b" if level in ["CRITICAL", "HEAVY"] else "#00c6ff"
                    st.markdown(f'<div class="glass-card {flash_class}" style="border-left: 5px solid {alert_color};">', unsafe_allow_html=True)
                    cols = st.columns(4)
                    cols[0].metric("Total Vehicles", total)
                    cols[1].metric("Congestion Level", level)
                    cols[2].metric("Weather", weather_status)
                    cols[3].metric("RL AI Mode", "ON" if rl_enabled else "OFF")
                    
                    if forced_lane == "EMERGENCY": st.error("🚨 SIREN DETECTED: ALL LANES STOPPED. CLEARING INTERSECTION.")
                    elif forced_lane == "PEDESTRIAN": st.warning("🚶‍♂️ PEDESTRIAN CROSSING ACTIVE.")
                    st.markdown('</div>', unsafe_allow_html=True)

                # 4. Lane Cards
                with lane_placeholder.container():
                    lane_breakdown_ui(detailed_counts, active_lane)

                # ---------------- DATABASE LOGGING & HISTORY ----------------
                current_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = {"Timestamp": current_dt, "Total_Vehicles": total, "Status": level, "Weather": weather_status}
                for l in LANES:
                    log_entry[f"{l}_Total"] = lane_counts[l]
                    for v in ["car", "motorcycle", "bus", "truck"]:
                        log_entry[f"{l}_{v}"] = detailed_counts[l][v]
                
                st.session_state.full_data_log.append(log_entry)

                if current_time - st.session_state.last_status_time >= 30 and not forced_lane:
                    if level == "CRITICAL": send_telegram_alert(f"Critical Traffic Detected. Current Volume: {total} vehicles.")
                    st.session_state.last_status_time = current_time
                    log_to_db('traffic_logs_v2', (current_dt, total, level, weather_status))

                st.session_state.traffic_history.append(total)
                if len(st.session_state.traffic_history)>50: st.session_state.traffic_history.pop(0)

                # 5. History Chart
                df_hist=pd.DataFrame({"Frame":range(len(st.session_state.traffic_history)), "Vehicles":st.session_state.traffic_history})
                history_placeholder.subheader("Traffic Flow History")
                history_placeholder.line_chart(df_hist.set_index("Frame"), color="#00c6ff")

                # 6. Predictions & Export
                predicted=predict_traffic(st.session_state.traffic_history)
                metric_placeholder.metric("Predicted Vehicles Next Interval", predicted)

            time.sleep(0.08)

# ==========================================
# 🛑 PAGE 2: DATA ANALYTICS DASHBOARD 🛑
# ==========================================
elif app_mode == "Data Analytics":
    st.title("📈 Astra Data Analytics Dashboard")
    st.caption("Historical data pulled directly from SQLite.")
    
    conn = st.session_state.db_conn
    try:
        df_traffic = pd.read_sql_query("SELECT * FROM traffic_logs_v2", conn)
        df_violations = pd.read_sql_query("SELECT * FROM violations", conn)
    except Exception as e:
        st.error("Database loading error.")
        df_traffic = pd.DataFrame()
        df_violations = pd.DataFrame()
        
    if df_traffic.empty:
        st.info("No traffic data logged yet. Please run the Live AI Feed to collect data.")
    else:
        df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Entries", len(df_traffic))
        col2.metric("Peak Congestion Volume", df_traffic['total_vehicles'].max())
        col3.metric("Total Recorded Violations", len(df_violations))
        
        st.markdown("---")
        
        st.subheader("📊 Traffic Volume Timeline")
        chart_data = df_traffic.set_index('timestamp')['total_vehicles']
        st.line_chart(chart_data, color="#00ffcc")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("⚠️ Violations Breakdown")
            if not df_violations.empty:
                violation_counts = df_violations['type'].value_counts()
                st.bar_chart(violation_counts, color="#ff4b4b")
            else:
                st.success("No violations recorded in the database yet.")
                
        with col_b:
            st.subheader("🌦️ Weather Conditions Logged")
            weather_counts = df_traffic['weather'].value_counts()
            st.bar_chart(weather_counts, color="#00c6ff")
            
        st.markdown("---")
        st.subheader("🗄️ Raw SQLite Logs (Last 100)")
        st.dataframe(df_traffic.tail(100), use_container_width=True)