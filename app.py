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
from sklearn.linear_model import LinearRegression
import streamlit as st
import torch

st.set_page_config(
    page_title="Smart Traffic Management - Master Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 🛑 SECURE TELEGRAM CREDENTIALS 🛑
# ==========================================
BOT_TOKEN = st.secrets["BOT_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]


# ---------------- THREAD-SAFE FRAME BUFFER (HIGH FPS) ----------------
class FrameBuffer:

  def __init__(self):
    self.frame = None
    self.lock = threading.Lock()

  def update(self, frame):
    with self.lock:
      self.frame = frame

  def read(self):
    with self.lock:
      return self.frame


global_buffer = FrameBuffer()


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

  def __init__(self, maxDisappeared=15):
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
      D = np.zeros((len(objectCentroids), len(inputCentroids)))
      for i, oc in enumerate(objectCentroids):
        for j, ic in enumerate(inputCentroids):
          D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]
      usedRows = set()
      usedCols = set()
      for row, col in zip(rows, cols):
        if row in usedRows or col in usedCols:
          continue
        if D[row, col] > 50:
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
if "tracker" not in st.session_state:
  st.session_state.tracker = CentroidTracker(maxDisappeared=15)
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
  while True:
    if not st.session_state.get("run_siren_sim", False):
      st.session_state.siren_active = False
    if (
        st.session_state.get("run_siren_sim", False)
        and st.session_state.running
    ):
      if random.random() < 0.05:
        st.session_state.siren_trigger_event = True
        time.sleep(12)
    time.sleep(3)


if (
    st.session_state.get("running", False)
    and "siren_thread" not in st.session_state
):
  threading.Thread(target=simulate_siren, daemon=True).start()
  st.session_state.siren_thread = True


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
  if not st.session_state.get("alerts_enabled", False):
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


# ---------------- HIGH PERFORMANCE INFERENCE WORKER THREAD ----------------
def video_inference_worker(camera_url):
  cam_src = int(camera_url) if camera_url.isdigit() else camera_url
  cap = cv2.VideoCapture(cam_src)
  frame_counter = 0

  while st.session_state.get("running", False):
    ret, frame = cap.read()
    if not ret:
      cap.release()
      time.sleep(0.5)
      cap = cv2.VideoCapture(cam_src)
      continue

    frame_counter += 1
    current_time = time.time()
    frame = cv2.resize(frame, (1024, 768))

    # AI Detection Cycle
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

      for _, det in detections.iterrows():
        cx = (int(det.xmin) + int(det.xmax)) // 2
        cy = (int(det.ymin) + int(det.ymax)) // 2
        v_class = det["name"]
        for lane, (lx1, ly1, lx2, ly2) in lanes_region.items():
          if lx1 <= cx < lx2 and ly1 <= cy < ly2:
            l_counts[lane] += 1
            if v_class in d_counts[lane]:
              d_counts[lane][v_class] += 1

      alpha = 0.20
      for l in LANES:
        st.session_state.smoothed_lane_counts[l] = (alpha * l_counts[l]) + (
            (1 - alpha) * st.session_state.smoothed_lane_counts[l]
        )

      st.session_state.lane_counts = l_counts
      st.session_state.detailed_counts = d_counts

      # Signal timing logic sync
      if current_time >= st.session_state.signal_end_time:
        current_state = traffic_status(
            sum(st.session_state.lane_counts.values())
        )
        st.session_state.lane_durations = calculate_lane_times(
            st.session_state.lane_counts, False, current_state
        )
        st.session_state.cycle_schedule = build_cycle_schedule(
            st.session_state.lane_durations, current_time
        )
        st.session_state.signal_end_time = st.session_state.cycle_schedule[-1][
            "end"
        ]

      # Tracking & Annotations
      rects = []
      for _, det in detections[detections["name"] != "person"].iterrows():
        rects.append(
            (int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax))
        )
      objects = st.session_state.tracker.update(rects)

      draw_cmds = []
      for objectID, centroid in objects.items():
        cx, cy = centroid[0], centroid[1]
        draw_cmds.append({"id": objectID, "cx": cx, "cy": cy})
      st.session_state.draw_commands = draw_cmds

    # Render Visual Overlay on Frame
    annotated = frame.copy()
    for cmd in st.session_state.get("draw_commands", []):
      cv2.circle(annotated, (cmd["cx"], cmd["cy"]), 4, (0, 255, 0), -1)
      cv2.putText(
          annotated,
          f"ID {cmd['id']}",
          (cmd["cx"] - 10, cmd["cy"] - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
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

    global_buffer.update(annotated)
    time.sleep(0.01)

  cap.release()


# ==========================================
# 🛑 SIDEBAR NAVIGATION 🛑
# ==========================================
with st.sidebar:
  app_mode = st.radio("🛰️ ASTRA Navigation", ["Live AI Feed", "Data Analytics"])
  st.markdown("---")

  if app_mode == "Live AI Feed":
    st.header("⚙️ AI Core Controls")
    ambulance_demo = st.toggle("🚑 Ambulance Auto-Routing", value=False)
    st.session_state.alerts_enabled = st.toggle(
        "🔕 Enable Telegram Alerts", value=False
    )
    rl_enabled = st.toggle("Enable Auto-Learning", value=False)
    st.session_state.run_siren_sim = st.toggle(
        "Listen for Sirens", value=False
    )
  elif app_mode == "Data Analytics":
    st.header("📊 Database Export")
    df_export = pd.DataFrame(st.session_state.full_data_log)
    if not df_export.empty:
      csv = df_export.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download CSV",
          data=csv,
          file_name=f"traffic_report_{int(time.time())}.csv",
          mime="text/csv",
      )

# ==========================================
# 🛑 PAGE 1: LIVE AI FEED 🛑
# ==========================================
if app_mode == "Live AI Feed":
  camera_url = st.text_input("Video Source", "traffic.mp4")
  col1, col2 = st.columns(2)

  if col1.button("Start Camera"):
    if not st.session_state.running:
      st.session_state.running = True
      threading.Thread(
          target=video_inference_worker, args=(camera_url,), daemon=True
      ).start()
  if col2.button("Stop Camera"):
    st.session_state.running = False

  video_placeholder = st.empty()
  lights_placeholder = st.empty()
  status_panel = st.empty()
  lane_placeholder = st.empty()
  graph_placeholder = st.empty()

  if not st.session_state.running:
    video_placeholder.markdown(
        """
            <div class="loader-wrapper">
                <div class="astra-spinner"></div>
                <div class="loader-text">SYSTEM STANDBY<br><span style="font-size: 12px; color: #aaa; text-transform: none;">Click 'Start Camera' to launch high-speed stream.</span></div>
            </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    while st.session_state.running:
      frame = global_buffer.read()
      if frame is not None:
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        video_placeholder.image(buffer.tobytes(), use_container_width=True)

      # UI Status Dashboard Updates
      current_time = time.time()
      if current_time - st.session_state.last_ui_update >= 1.0:
        st.session_state.last_ui_update = current_time

        display_counts = {
            k: int(round(v))
            for k, v in st.session_state.smoothed_lane_counts.items()
        }
        total = sum(display_counts.values())
        level = traffic_status(total)

        with status_panel.container():
          cols = st.columns(4)
          cols[0].metric("Total Vehicles", total)
          cols[1].metric("Congestion Level", level)
          cols[2].metric("Weather", st.session_state.weather_status)
          cols[3].metric("Engine Mode", "HIGH-FPS THREADED ENGINE")

      time.sleep(0.02)

# ==========================================
# 🛑 PAGE 2: DATA ANALYTICS 🛑
# ==========================================
elif app_mode == "Data Analytics":
  st.title("📊 Data Analytics & System Logs")
  conn = st.session_state.db_conn

  st.subheader("🚨 Intersection Violations Log")
  df_violations = pd.read_sql_query(
      "SELECT * FROM violations ORDER BY timestamp DESC LIMIT 100", conn
  )
  if not df_violations.empty:
    st.dataframe(df_violations, use_container_width=True)
  else:
    st.info("No traffic violations recorded yet.")

  st.subheader("📈 Traffic Density Log")
  df_traffic = pd.read_sql_query(
      "SELECT * FROM traffic_logs_v2 ORDER BY timestamp DESC LIMIT 100", conn
  )
  if not df_traffic.empty:
    st.dataframe(df_traffic, use_container_width=True)
  else:
    st.info("No traffic density logs recorded yet.")