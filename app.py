import base64
from collections import OrderedDict
import datetime
import gc
import os
import random
import sqlite3
import threading
import time

import cv2
from gtts import gTTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
import streamlit as st
import torch

st.set_page_config(
    page_title="ASTRA Smart Traffic Management",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 🛑 PERFECT TELEGRAM CREDENTIALS 🛑
# ==========================================
try:
  BOT_TOKEN = st.secrets["BOT_TOKEN"]
  CHAT_ID = st.secrets["CHAT_ID"]
except:
  BOT_TOKEN = ""
  CHAT_ID = ""


# ---------------- DATABASE INIT ----------------
def init_db():
  conn = sqlite3.connect("traffic_system.db", check_same_thread=False)
  c = conn.cursor()
  c.execute("""CREATE TABLE IF NOT EXISTS traffic_logs_v2 
                 (timestamp TEXT, total_vehicles INTEGER, status TEXT, weather TEXT)"""
  )
  c.execute("""CREATE TABLE IF NOT EXISTS violations 
                 (timestamp TEXT, type TEXT, details TEXT)""")
  conn.commit()
  return conn


if "db_conn" not in st.session_state:
  st.session_state.db_conn = init_db()


def log_to_db(table, data):
  c = st.session_state.db_conn.cursor()
  placeholders = ", ".join(["?"] * len(data))
  c.execute(f"INSERT INTO {table} VALUES ({placeholders})", data)
  st.session_state.db_conn.commit()


# ---------------- TRACKER ----------------
class CentroidTracker:

  def __init__(self, maxDisappeared=50):
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
        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)
      return self.objects

    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    for i, (startX, startY, endX, endY) in enumerate(rects):
      inputCentroids[i] = (
          int((startX + endX) / 2.0),
          int((startY + endY) / 2.0),
      )

    if len(self.objects) == 0:
      for i in range(0, len(inputCentroids)):
        self.register(inputCentroids[i])
    else:
      objectIDs = list(self.objects.keys())
      objectCentroids = list(self.objects.values())
      D = distance.cdist(np.array(objectCentroids), inputCentroids)
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]

      usedRows, usedCols = set(), set()
      for row, col in zip(rows, cols):
        if row in usedRows or col in usedCols:
          continue
        objectID = objectIDs[row]
        self.objects[objectID] = inputCentroids[col]
        self.disappeared[objectID] = 0
        usedRows.add(row)
        usedCols.add(col)

      unusedRows = set(range(0, D.shape[0])).difference(usedRows)
      unusedCols = set(range(0, D.shape[1])).difference(usedCols)

      for row in unusedRows:
        objectID = objectIDs[row]
        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)

      for col in unusedCols:
        self.register(inputCentroids[col])

    return self.objects


tracker = CentroidTracker(maxDisappeared=10)


# ---------------- GLOBAL UI & ANIMATIONS ----------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
  st.title("🚦 ASTRA Traffic Management System")
with header_col2:
  st.write("")
  audio_enabled = st.toggle("🔊 Enable Audio", value=True)

# ---------------- SETTINGS & STATE ----------------
BASE_GREEN = 60
MIN_GREEN = 10
MAX_GREEN = 120
LANES = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
LANE_NAMES = {
    "Lane 1": "North",
    "Lane 2": "South",
    "Lane 3": "East",
    "Lane 4": "West",
    "PEDESTRIAN": "Pedestrian Phase",
}

if "running" not in st.session_state:
  st.session_state.running = False
if "traffic_history" not in st.session_state:
  st.session_state.traffic_history = []
if "full_data_log" not in st.session_state:
  st.session_state.full_data_log = []
if "lane_durations" not in st.session_state:
  st.session_state.lane_durations = {lane: BASE_GREEN for lane in LANES}
if "cycle_schedule" not in st.session_state:
  st.session_state.cycle_schedule = []
if "signal_end_time" not in st.session_state:
  st.session_state.signal_end_time = 0
if "last_status_time" not in st.session_state:
  st.session_state.last_status_time = 0
if "track_history" not in st.session_state:
  st.session_state.track_history = {}
if "last_telegram_time" not in st.session_state:
  st.session_state.last_telegram_time = 0
if "siren_active" not in st.session_state:
  st.session_state.siren_active = False
if "rl_q_table" not in st.session_state:
  st.session_state.rl_q_table = {
      "LOW": 0,
      "MODERATE": 0,
      "HEAVY": 0,
      "CRITICAL": 0,
  }

if "siren_trigger_event" not in st.session_state:
  st.session_state.siren_trigger_event = False
if "siren_end_time" not in st.session_state:
  st.session_state.siren_end_time = 0
if "last_ui_update" not in st.session_state:
  st.session_state.last_ui_update = 0.0
if "siren_playing_now" not in st.session_state:
  st.session_state.siren_playing_now = False

if "cached_detections" not in st.session_state:
  st.session_state.cached_detections = pd.DataFrame(
      columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
  )
if "draw_commands" not in st.session_state:
  st.session_state.draw_commands = []
if "lane_counts" not in st.session_state:
  st.session_state.lane_counts = {lane: 0 for lane in LANES}
if "detailed_counts" not in st.session_state:
  st.session_state.detailed_counts = {
      lane: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0} for lane in LANES
  }
if "active_lane" not in st.session_state:
  st.session_state.active_lane = None
if "forced_lane" not in st.session_state:
  st.session_state.forced_lane = None
if "weather_status" not in st.session_state:
  st.session_state.weather_status = "Clear"
if "smoothed_lane_counts" not in st.session_state:
  st.session_state.smoothed_lane_counts = {lane: 0.0 for lane in LANES}

st.session_state.audio_enabled = audio_enabled


# ---------------- CLOUD-SAFE AUDIO FUNCTIONS ----------------
def speak(text):
  if "audio_enabled" in st.session_state and not st.session_state.audio_enabled:
    return
  try:
    tts = gTTS(text=text, lang="en", slow=False)
    audio_file = f"voice_{random.randint(1000, 9999)}.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
      data = f.read()
    b64 = base64.b64encode(data).decode()
    if os.path.exists(audio_file):
      os.remove(audio_file)
    audio_html = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(audio_html, unsafe_allow_html=True)
  except Exception as e:
    print(f"Audio playback error: {e}")


def announce_signal(lane, duration):
  speak(
      f"Astra routing. {LANE_NAMES.get(lane, lane)} green for {duration}"
      " seconds."
  )


# ---------------- ASTRA BOOT-UP SEQUENCE ----------------
if "has_welcomed" not in st.session_state:
  speak(
      "Welcome back, sir. I am Astra. All core systems are initialized and"
      " ready. Let's get to work."
  )
  st.session_state.has_welcomed = True


def add_bgm():
  bgm_file = "bgm.mp3"
  if os.path.exists(bgm_file):
    with open(bgm_file, "rb") as f:
      b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<audio autoplay loop style="display:none;"><source'
        f' src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )


if st.session_state.audio_enabled:
  add_bgm()


# ---------------- SIREN SIMULATION THREAD ----------------
def simulate_siren():
  while st.session_state.get("siren_simulation", False):
    time.sleep(random.randint(15, 30))
    if random.random() < 0.05:
      st.session_state.siren_trigger_event = True


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.hub.load(
      "ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True
  ).to(device)
  if torch.cuda.is_available():
    model.half()
  model.eval()
  model.iou = 0.45
  return model


with st.spinner("🤖 ASTRA AI Core Initializing... Loading Network..."):
  model = load_model()
target_classes = ["car", "truck", "bus", "motorcycle", "person"]


# ---------------- TELEGRAM ALERT FUNC ----------------
def send_telegram_alert(message, force=False):
  if not st.session_state.get("telegram_alerts", False) and not force:
    return
  if not BOT_TOKEN or not CHAT_ID:
    return
  current_time = time.time()
  if not force and (
      current_time - st.session_state.last_telegram_time < 15
  ):
    return
  st.session_state.last_telegram_time = current_time
  st.toast("Sending Alert...", icon="🚨")
  try:
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": f"🚨 ASTRA:\n{message}"},
        timeout=3,
    )
  except:
    pass


# ---------------- TRAFFIC MATH & RL ----------------
def traffic_status(total):
  if total <= 10:
    return "LOW"
  elif total <= 20:
    return "MODERATE"
  elif total <= 30:
    return "HEAVY"
  else:
    return "CRITICAL"


def calculate_lane_times(lane_counts, rl_active, current_state):
  base_cycle = BASE_GREEN * 4
  total_cars = sum(lane_counts.values())
  durations = {}
  if total_cars == 0:
    for lane in LANES:
      durations[lane] = min(base_cycle // 4, MAX_GREEN)
  else:
    weights = {lane: count / total_cars for lane, count in lane_counts.items()}
    for lane in LANES:
      t = int(base_cycle * weights[lane])
      durations[lane] = max(MIN_GREEN, min(MAX_GREEN, t))
  if rl_active and total_cars > 0:
    q_score = st.session_state.rl_q_table[current_state]
    modifier = int(q_score * 2)
    for lane in LANES:
      durations[lane] = max(MIN_GREEN, durations[lane] + modifier)
  return durations


def build_cycle_schedule(durations, start_time):
  schedule = []
  current = start_time
  for lane in LANES:
    end = current + durations[lane]
    schedule.append({"lane": lane, "start": current, "end": end})
    current = end
  return schedule


def predict_traffic(history):
  if len(history) < 5:
    return history[-1] if history else 0
  X = np.arange(len(history)).reshape(-1, 1)
  y = np.array(history)
  model_lr = LinearRegression()
  model_lr.fit(X, y)
  future = np.array([[len(history) + 1]])
  return int(max(0, model_lr.predict(future)[0]))


# ---------------- UI DASHBOARD COMPONENTS ----------------
def traffic_lights_with_timers(
    current_time, schedule, override_active_lane=None
):
  cols = st.columns(4)
  lane_wait = {lane: 0 for lane in LANES}
  active_lane = override_active_lane

  for item in schedule:
    lane = item["lane"]
    if item["start"] <= current_time < item["end"]:
      lane_wait[lane] = int(item["end"] - current_time)
      if not override_active_lane:
        active_lane = lane
    else:
      lane_wait[lane] = (
          int(item["start"] - current_time)
          if current_time < item["start"]
          else 0
      )

  if override_active_lane:
    for lane in LANES:
      lane_wait[lane] = "♾️" if lane == override_active_lane else "🛑"

  for lane, col in zip(LANES, cols):
    with col:
      is_active = lane == active_lane
      wrapper_class = "pulse" if is_active else ""
      percentage = 100
      if is_active and isinstance(lane_wait[lane], int):
        total_duration = st.session_state.lane_durations.get(lane, 60)
        if total_duration > 0:
          percentage = max(
              0, min(100, (lane_wait[lane] / total_duration) * 100)
          )

      if is_active:
        st.markdown(
            f"""
                <div class="glass-card glow-box fade-in {wrapper_class}">
                <h4 style="text-align:center">{LANE_NAMES.get(lane, lane)}</h4>
                <div style="width:40px;height:40px;background:red;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:lime;border-radius:50%;margin:auto; box-shadow: 0 0 15px lime;"></div>
                <p style="color:lime;text-align:center;font-size:24px; font-weight:bold;">{lane_wait[lane]}s</p>
                <div style="width: 100%; background-color: rgba(255,255,255,0.1); border-radius: 5px; height: 6px;">
                    <div style="width: {percentage}%; height: 6px; background-color: lime; border-radius: 5px;"></div>
                </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            f"""
                <div class="glass-card glow-box fade-in shine">
                <h4 style="text-align:center; color: #888;">{LANE_NAMES.get(lane, lane)}</h4>
                <div style="width:40px;height:40px;background:red;border-radius:50%;margin:auto; box-shadow: 0 0 10px red;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <div style="width:40px;height:40px;background:gray;border-radius:50%;margin:auto; opacity: 0.2;"></div>
                <p style="color:#ff4444;text-align:center;font-size:20px;">Wait: {lane_wait[lane]}s</p>
                </div>
                """,
            unsafe_allow_html=True,
        )


def lane_breakdown_ui(detailed_counts, active_lane=None):
  st.subheader("Lane-wise Vehicle Breakdown")
  col1, col2 = st.columns(2)
  col3, col4 = st.columns(2)

  def render_card(title, data, is_active=False):
    status = "🟢 ACTIVE" if is_active else "🔴 STOPPED"
    wrapper_class = "pulse" if is_active else ""
    border_color = (
        "rgba(0, 255, 0, 0.4)" if is_active else "rgba(255, 255, 255, 0.1)"
    )
    st.markdown(
        f"""
        <div class="glass-card glow-box fade-in {wrapper_class}" style="background:#0f172a; margin:10px; border:1px solid {border_color};">
            <h4 style="color:white; margin-bottom: 10px;">{title} <span style="float:right; font-size: 14px;">{status}</span></h4>
            <div style="display: flex; justify-content: space-between; color: #aaa; font-size: 18px; padding: 10px 0;">
                <span>🚗 <b style="color:white">{data['car']}</b></span>
                <span>🏍️ <b style="color:white">{data['motorcycle']}</b></span>
                <span>🚌 <b style="color:white">{data['bus']}</b></span>
                <span>🚚 <b style="color:white">{data['truck']}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with col1:
    render_card(
        "NORTH", detailed_counts["Lane 1"], is_active=("Lane 1" == active_lane)
    )
  with col2:
    render_card(
        "SOUTH", detailed_counts["Lane 2"], is_active=("Lane 2" == active_lane)
    )
  with col3:
    render_card(
        "EAST", detailed_counts["Lane 3"], is_active=("Lane 3" == active_lane)
    )
  with col4:
    render_card(
        "WEST", detailed_counts["Lane 4"], is_active=("Lane 4" == active_lane)
    )


# ==========================================
# 🛑 SIDEBAR NAVIGATION & TOGGLES 🛑
# ==========================================
st.sidebar.title("🚦 ASTRA Control Panel")
app_mode = st.sidebar.selectbox(
    "Navigate", ["Dashboard", "Live AI Feed", "Data Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Modules")
ambulance_demo = st.sidebar.toggle("🚑 Ambulance Auto-Routing", value=False)
telegram_alerts = st.sidebar.toggle("🔕 Enable Telegram Alerts", value=False)
q_learning = st.sidebar.toggle("🧠 Enable Auto-Learning", value=True)
siren_simulation = st.sidebar.toggle("🚨 Listen for Sirens", value=False)

st.session_state.ambulance_demo = ambulance_demo
st.session_state.telegram_alerts = telegram_alerts
st.session_state.siren_simulation = siren_simulation

if siren_simulation and not st.session_state.get("siren_thread_started"):
  st.session_state.siren_thread_started = True
  threading.Thread(target=simulate_siren, daemon=True).start()

# ==========================================
# 🛑 PAGE 1: HOME / DASHBOARD 🛑
# ==========================================
if app_mode == "Dashboard":
  st.title("🚦 ASTRA Control Center")
  st.markdown("### Advanced Smart Traffic Routing Architecture")

  col1, col2, col3, col4 = st.columns(4)
  total_vehicles = sum(st.session_state.smoothed_lane_counts.values())
  congestion = "HIGH" if total_vehicles > 40 else "LOW"

  col1.metric("Total Vehicles", int(total_vehicles))
  col2.metric("Congestion Level", congestion)
  col3.metric("Weather Status", st.session_state.weather_status)
  col4.metric("Engine Mode", "UNCAPPED FPS TENSOR")

  st.markdown("---")
  st.subheader("Lane Density (EMA Smoothed)")

  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor("#0e1117")
  ax.set_facecolor("#0e1117")
  ax.bar(
      LANES,
      list(st.session_state.smoothed_lane_counts.values()),
      color=[
          "#00ff00" if l == st.session_state.active_lane else "#1f77b4"
          for l in LANES
      ],
  )
  ax.set_ylim(
      0, max(20, max(st.session_state.smoothed_lane_counts.values()) + 10)
  )
  ax.tick_params(colors="white")
  st.pyplot(fig)

  st.info(
      "System Online. Navigate to 'Live AI Feed' to initialize inference"
      " engine."
  )

# ==========================================
# 🛑 PAGE 2: LIVE AI FEED 🛑
# ==========================================
elif app_mode == "Live AI Feed":
  st.title("🎥 Live Intersection AI")
  camera_url = st.text_input("Video Source", "traffic.mp4")
  col1, col2 = st.columns(2)

  if col1.button("Start Camera"):
    st.session_state.running = True
  if col2.button("Stop Camera"):
    st.session_state.running = False
    if "cap" in st.session_state:
      st.session_state.cap.release()
      del st.session_state["cap"]

  env_indicator = st.empty()
  siren_placeholder = st.empty()
  video_placeholder = st.empty()
  graph_placeholder = st.empty()
  lights_placeholder = st.empty()
  status_panel = st.empty()
  lane_placeholder = st.empty()

  frame_counter = 0

  if not st.session_state.running:
    video_placeholder.markdown(
        """
            <div class="loader-wrapper">
                <div class="astra-spinner"></div>
                <div class="loader-text">SYSTEM STANDBY<br><span style="font-size: 12px; color: #aaa; text-transform: none;">Waiting for Video Feed...</span></div>
            </div>
        """,
        unsafe_allow_html=True,
    )

  if st.session_state.running:
    if "cap" not in st.session_state:
      cam_src = int(camera_url) if camera_url.isdigit() else camera_url
      st.session_state.cap = cv2.VideoCapture(cam_src)
      st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while st.session_state.running:
      ret, frame = st.session_state.cap.read()

      for _ in range(2):
        st.session_state.cap.grab()

      if not ret:
        st.session_state.cap.release()
        time.sleep(0.5)
        cam_src = int(camera_url) if camera_url.isdigit() else camera_url
        st.session_state.cap = cv2.VideoCapture(cam_src)
        continue

      frame_counter += 1
      current_time = time.time()
      frame = cv2.resize(frame, (1024, 768))

      # ---------------- AI FRAME INFERENCE ----------------
      if frame_counter % 3 == 0:
        if st.session_state.siren_trigger_event:
          st.session_state.siren_active = True
          st.session_state.siren_end_time = current_time + 12
          st.session_state.siren_trigger_event = False
          send_telegram_alert(
              "🚑 AMBULANCE SIREN DETECTED! Initiating Emergency Preemption.",
              force=True,
          )
          speak("Emergency vehicle detected. Clearing intersection.")

        if (
            st.session_state.siren_active
            and current_time > st.session_state.siren_end_time
        ):
          st.session_state.siren_active = False

        if frame_counter % 15 == 0:
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
          st.session_state.weather_status = (
              "Fog/Rain" if laplacian_var < 50 else "Clear"
          )
          st.session_state.current_yolo_conf = (
              0.15 if laplacian_var < 50 else 0.20
          )

        conf = st.session_state.get("current_yolo_conf", 0.20)
        with torch.no_grad():
          results = model(frame, size=640)

        df = results.pandas().xyxy[0]
        df = df[df["confidence"] > conf]
        detections = df[df["name"].isin(target_classes)]
        st.session_state.cached_detections = detections

        detected_ambulance_lane = None
        if ambulance_demo:
          h, w, _ = frame.shape
          lanes_region_temp = {
              "Lane 1": (0, 0, w // 2, h // 2),
              "Lane 2": (w // 2, 0, w, h // 2),
              "Lane 3": (0, h // 2, w // 2, h),
              "Lane 4": (w // 2, h // 2, w, h),
          }
          for _, det in detections.iterrows():
            if det["name"] in ["truck", "bus"]:
              cx = (int(det.xmin) + int(det.xmax)) // 2
              cy = (int(det.ymin) + int(det.ymax)) // 2
              for lane, (lx1, ly1, lx2, ly2) in lanes_region_temp.items():
                if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                  detected_ambulance_lane = lane
                  break

        col_w = 1024 // 4
        lanes_region = {
            "Lane 1": (0, 0, col_w, 768),
            "Lane 2": (col_w, 0, col_w * 2, 768),
            "Lane 3": (col_w * 2, 0, col_w * 3, 768),
            "Lane 4": (col_w * 3, 0, 1024, 768),
        }

        l_counts = {lane: 0 for lane in LANES}
        d_counts = {
            lane: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for lane in LANES
        }
        rects = []

        for _, det in detections.iterrows():
          x1, y1, x2, y2 = (
              int(det.xmin),
              int(det.ymin),
              int(det.xmax),
              int(det.ymax),
          )
          cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
          v_class = det["name"]
          rects.append((x1, y1, x2, y2))

          for lane, (lx1, ly1, lx2, ly2) in lanes_region.items():
            if lx1 <= cx < lx2 and ly1 <= cy < ly2:
              l_counts[lane] += 1
              if v_class in d_counts[lane]:
                d_counts[lane][v_class] += 1

        objects = tracker.update(rects)

        alpha = 0.20
        for l in LANES:
          st.session_state.smoothed_lane_counts[l] = (alpha * l_counts[l]) + (
              (1 - alpha) * st.session_state.smoothed_lane_counts[l]
          )

        st.session_state.lane_counts = l_counts
        st.session_state.detailed_counts = d_counts

        active_lane = None
        forced_lane = None

        def trigger_siren_audio():
          audio_file = (
              "siren.mp3"
              if os.path.exists("siren.mp3")
              else ("siren.m4a" if os.path.exists("siren.m4a") else None)
          )
          if audio_file:
            with open(audio_file, "rb") as f:
              b64_siren = base64.b64encode(f.read()).decode()

            with siren_placeholder.container():
              st.error("🚨 EMERGENCY SIREN BROADCAST ACTIVE")
              if st.session_state.audio_enabled:
                st.markdown(
                    f'<iframe src="data:audio/mpeg;base64,{b64_siren}"'
                    ' allow="autoplay" style="display:none"'
                    ' id="siren-frame"></iframe>',
                    unsafe_allow_html=True,
                )
          else:
            with siren_placeholder.container():
              st.error(
                  "🚨 EMERGENCY ACTIVE (Missing siren.mp3 or siren.m4a file!)"
              )

        s_sim_on = st.session_state.get("siren_simulation", False)
        s_active_bool = st.session_state.siren_active

        if detected_ambulance_lane:
          forced_lane = detected_ambulance_lane
          active_lane = detected_ambulance_lane
          if not st.session_state.siren_playing_now:
            st.session_state.siren_playing_now = True
            send_telegram_alert(
                f"🚑 VISUAL AMBULANCE DETECTED IN"
                f" {LANE_NAMES[active_lane].upper()}!",
                force=True,
            )
          trigger_siren_audio()

        elif s_sim_on and s_active_bool:
          forced_lane = "EMERGENCY"
          active_lane = "EMERGENCY"
          if not st.session_state.siren_playing_now:
            st.session_state.siren_playing_now = True
          trigger_siren_audio()

        else:
          if st.session_state.siren_playing_now:
            st.session_state.siren_playing_now = False
            siren_placeholder.empty()

        if not forced_lane and len(detections[detections["name"] == "person"]) >= 5:
          forced_lane = "PEDESTRIAN"
          active_lane = "PEDESTRIAN"

        if not forced_lane:
          for item in st.session_state.cycle_schedule:
            if item["start"] <= current_time < item["end"]:
              active_lane = item["lane"]
              break

        if current_time >= st.session_state.signal_end_time:
          current_state = traffic_status(
              sum(st.session_state.lane_counts.values())
          )
          st.session_state.lane_durations = calculate_lane_times(
              st.session_state.lane_counts, q_learning, current_state
          )
          st.session_state.cycle_schedule = build_cycle_schedule(
              st.session_state.lane_durations, current_time
          )
          st.session_state.signal_end_time = st.session_state.cycle_schedule[
              -1
          ]["end"]
          announce_signal(
              st.session_state.cycle_schedule[0]["lane"],
              st.session_state.lane_durations[
                  st.session_state.cycle_schedule[0]["lane"]
              ],
          )

        st.session_state.active_lane = active_lane
        st.session_state.forced_lane = forced_lane

        # Full Trajectory Tracking & Anomaly Checking Logic
        active_ids = list(objects.keys())
        st.session_state.track_history = {
            k: v
            for k, v in st.session_state.track_history.items()
            if k in active_ids
        }

        draw_cmds = []
        for objectID, centroid in objects.items():
          cx, cy = centroid[0], centroid[1]
          if objectID not in st.session_state.track_history:
            st.session_state.track_history[objectID] = []

          if frame_counter % 6 == 0:
            st.session_state.track_history[objectID].append(
                (cx, cy, current_time)
            )
          if len(st.session_state.track_history[objectID]) > 20:
            st.session_state.track_history[objectID].pop(0)

          history = st.session_state.track_history[objectID]
          is_anomaly = False
          int_x1, int_x2 = 350, 674
          int_y1, int_y2 = 250, 518

          if len(history) == 20:
            sx, sy, stime = history[0]
            dist = np.hypot(cx - sx, cy - sy)
            if dist < 10 and (current_time - stime) > 3.0:
              if int_x1 < cx < int_x2 and int_y1 < cy < int_y2:
                is_anomaly = True

          is_violation = False
          if (
              int_x1 < cx < int_x2
              and int_y1 < cy < int_y2
              and active_lane != "None"
              and active_lane is not None
          ):
            if len(history) > 5:
              prev_x, prev_y, _ = history[-5]
              origin_lane = None
              if prev_y < int_y1 and int_x1 < prev_x < 500:
                origin_lane = "Lane 1"
              elif prev_y > int_y2 and 500 < prev_x < int_x2:
                origin_lane = "Lane 2"
              elif prev_x < int_x1 and int_y1 < prev_y < int_y2:
                origin_lane = "Lane 4"
              elif prev_x > int_x2 and int_y1 < prev_y < int_y2:
                origin_lane = "Lane 3"

              if (
                  origin_lane
                  and origin_lane != active_lane
                  and active_lane != "PEDESTRIAN"
              ):
                is_violation = True

          color = (0, 255, 0)
          alert_text = ""
          if is_anomaly:
            color = (0, 0, 255)
            alert_text = "WARNING: STATIONARY"
            if frame_counter % 30 == 0:
              log_to_db(
                  "violations",
                  (
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "Anomaly",
                      f"Obj {objectID} stopped",
                  ),
              )
              send_telegram_alert(
                  "Accident/Breakdown suspected! Vehicle ID:"
                  f" {objectID} stopped in intersection."
              )
          elif is_violation:
            color = (255, 0, 0)
            alert_text = "VIOLATION!"
            if frame_counter % 30 == 0:
              log_to_db(
                  "violations",
                  (
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "Red Light",
                      f"Obj {objectID} ran red light",
                  ),
              )
              send_telegram_alert(
                  f"🚨 RED LIGHT VIOLATION! Vehicle ID {objectID} ran the"
                  " intersection."
              )

          draw_cmds.append({
              "id": objectID,
              "cx": cx,
              "cy": cy,
              "color": color,
              "alert": alert_text,
          })
        st.session_state.draw_commands = draw_cmds

      # --- RENDER TICK ---
      if frame_counter % 5 == 0:
        annotated = frame.copy()
        int_x1, int_x2 = 350, 674
        int_y1, int_y2 = 250, 518
        cv2.rectangle(
            annotated, (int_x1, int_y1), (int_x2, int_y2), (0, 255, 255), 2
        )
        cv2.putText(
            annotated,
            "INTERSECTION ZONE",
            (int_x1, int_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        for cmd in st.session_state.draw_commands:
          cx, cy, color, alert = (
              cmd["cx"],
              cmd["cy"],
              cmd["color"],
              cmd["alert"],
          )
          cv2.putText(
              annotated,
              f"ID {cmd['id']}",
              (cx - 10, cy - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              color,
              2,
          )
          cv2.circle(annotated, (cx, cy), 4, color, -1)
          if alert:
            cv2.putText(
                annotated,
                alert,
                (cx - 30, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        for _, det in st.session_state.cached_detections.iterrows():
          cv2.rectangle(
              annotated,
              (int(det.xmin), int(det.ymin)),
              (int(det.xmax), int(det.ymax)),
              (0, 255, 255),
              2,
          )

        _, buffer = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        video_placeholder.image(buffer.tobytes(), use_container_width=True)

        del annotated, buffer
        gc.collect()

      # ---------------- UI DASHBOARD UPDATES ----------------
      if current_time - st.session_state.last_ui_update >= 1.5:
        st.session_state.last_ui_update = current_time

        display_counts = {
            k: int(round(v))
            for k, v in st.session_state.smoothed_lane_counts.items()
        }
        total = sum(display_counts.values())
        level = traffic_status(total)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(
            list(display_counts.keys()),
            list(display_counts.values()),
            color="#00c6ff",
        )
        ax.set_title("Traffic Density", color="white")
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.set_ylim(
            0,
            max(15, max(display_counts.values()) if display_counts else 15),
        )

        graph_placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)

        with lights_placeholder.container():
          traffic_lights_with_timers(
              current_time,
              st.session_state.cycle_schedule,
              override_active_lane=st.session_state.forced_lane,
          )

        with status_panel.container():
          flash_class = (
              "emergency-flash" if st.session_state.forced_lane else ""
          )
          alert_color = (
              "#ff4b4b" if level in ["CRITICAL", "HEAVY"] else "#00c6ff"
          )
          st.markdown(
              f'<div class="glass-card {flash_class}" style="border-left: 5px'
              f' solid {alert_color};">',
              unsafe_allow_html=True,
          )
          cols = st.columns(4)
          cols[0].metric("Total Vehicles", total)
          cols[1].metric("Congestion Level", level)
          cols[2].metric("Weather", st.session_state.weather_status)
          cols[3].metric("Engine Mode", "UNCAPPED FPS TENSOR OPTIMIZED")
          st.markdown("</div>", unsafe_allow_html=True)

        with lane_placeholder.container():
          lane_breakdown_ui(
              st.session_state.detailed_counts, st.session_state.active_lane
          )

      # ---------------- MEMORY NUKE (Every 30 frames) ----------------
      if frame_counter % 30 == 0:
        gc.collect()
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

# ==========================================
# 🛑 PAGE 3: DATA ANALYTICS 🛑
# ==========================================
elif app_mode == "Data Analytics":
  st.title("📊 Data Analytics & System Logs")
  conn = st.session_state.db_conn

  st.subheader("🚨 Intersection Violations Log")
  try:
    df_violations = pd.read_sql_query(
        "SELECT * FROM violations ORDER BY timestamp DESC LIMIT 100", conn
    )
    if not df_violations.empty:
      st.dataframe(df_violations, use_container_width=True)
    else:
      st.info("No traffic violations recorded yet.")
  except:
    st.warning("Database table 'violations' not yet initialized.")

  st.subheader("📈 Traffic Density Log")
  try:
    df_traffic = pd.read_sql_query(
        "SELECT * FROM traffic_logs_v2 ORDER BY timestamp DESC LIMIT 100", conn
    )
    if not df_traffic.empty:
      st.dataframe(df_traffic, use_container_width=True)
    else:
      st.info("No traffic density logs recorded yet.")
  except:
    st.warning("Database table 'traffic_logs_v2' not yet initialized.")