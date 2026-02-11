import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
from flask import Flask
from threading import Thread
import logging
import sys
import datetime
import pyotp
import json
from NorenRestApiPy.NorenApi import NorenApi

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ProjectP")

# ==========================================
# 1. FLASK KEEP-ALIVE
# ==========================================
app = Flask('')

@app.route('/')
def home():
    return "PROJECT P: SYSTEMS ONLINE"

def run_http():
    port = int(os.environ.get("PORT", 8080))
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_http)
    t.start()

# ==========================================
# 2. CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Shoonya Credentials
SHOONYA_USER_ID = os.environ.get("SHOONYA_USER_ID", "")
SHOONYA_PASSWORD = os.environ.get("SHOONYA_PASSWORD", "")
SHOONYA_TOTP_SECRET = os.environ.get("SHOONYA_TOTP_SECRET", "")
SHOONYA_VC = os.environ.get("SHOONYA_VC", "")
SHOONYA_API_KEY = os.environ.get("SHOONYA_API_KEY", "")

TOKEN_NIFTY = '26000' 
EXCHANGE = 'NSE'

# Hyperparameters
INPUT_SEQ_LEN = 10     
INPUT_FEATURES = 5     
HIDDEN_SIZE = 64       
LEARNING_RATE = 0.001
DELAY_SECONDS = 1800   # 30 Minutes
MIN_TRADES_STATS = 50  

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass

def parse_shoonya_time(time_str):
    """Robust timestamp parser for mixed formats"""
    try:
        # Try converting direct epoch string (e.g. '167...')
        return float(time_str)
    except ValueError:
        try:
            # Try parsing date string (e.g. '10-02-2026 15:15:00')
            # Adjust format if Shoonya sends something different
            dt_obj = datetime.datetime.strptime(time_str, '%d-%m-%Y %H:%M:%S')
            return dt_obj.timestamp()
        except Exception as e:
            logger.warning(f"Time Parse Error for '{time_str}': {e}")
            return time.time() # Fallback to current time

# ==========================================
# 4. DATA MANAGEMENT (FIXED)
# ==========================================
class DataManager:
    def __init__(self):
        self.history = deque(maxlen=INPUT_SEQ_LEN)
        self.api = NorenApi(host='https://api.shoonya.com/NorenWClientTP/', 
                           websocket='wss://api.shoonya.com/NorenWSTP/')
        self.last_login_time = 0
        self.last_price = 0.0 # To check for stale data
        self.login()

    def login(self):
        try:
            if not SHOONYA_TOTP_SECRET or not SHOONYA_USER_ID:
                return False

            totp = pyotp.TOTP(SHOONYA_TOTP_SECRET).now()
            ret = self.api.login(
                userid=SHOONYA_USER_ID, password=SHOONYA_PASSWORD, 
                twoFA=totp, vendor_code=SHOONYA_VC, 
                api_secret=SHOONYA_API_KEY, imei='12345'
            )
            
            if ret and 'stat' in ret and ret['stat'] == 'Ok':
                logger.info("✅ Shoonya Login Successful")
                self.last_login_time = time.time()
                return True
            else:
                return False
        except Exception:
            return False

    def fetch_market_data(self):
        """
        Fetches data with robust error handling.
        Returns: (packet, current_price, is_active)
        """
        # Session Refresh
        if time.time() - self.last_login_time > 72000: self.login()
        
        packet = []
        current_price = 0.0
        current_ts = time.time()

        try:
            # --- A. Get LIVE Price ---
            quote = self.api.get_quotes(exchange=EXCHANGE, token=TOKEN_NIFTY)
            
            if quote and 'lp' in quote:
                current_price = float(quote['lp'])
            else:
                # If Live fails, try fallback but mark as possibly stale
                return None, None, False

            # --- CRITICAL: STALE DATA CHECK ---
            # If price hasn't moved since last loop, Market is likely closed/frozen.
            # We allow small float differences, but exact match usually means frozen.
            if current_price == self.last_price:
                 # Logic: If exact same price, assume no activity.
                 # This prevents the "Loss loop" on weekends.
                 return None, current_price, False
            
            self.last_price = current_price

            # --- B. Get History ---
            timeframes = [
                {'iv': 'd',  'dur': 0},      
                {'iv': '60', 'dur': 3600},   
                {'iv': '30', 'dur': 1800},   
                {'iv': '5',  'dur': 300}     
            ]
            
            start_ts = (datetime.datetime.now() - datetime.timedelta(days=5)).timestamp()

            for tf in timeframes:
                iv = tf['iv']
                duration = tf['dur']
                
                # Try/Except block specifically for History API calls
                try:
                    hist = self.api.get_time_price_series(exchange=EXCHANGE, token=TOKEN_NIFTY, starttime=start_ts, interval=iv)
                except Exception:
                    hist = None
                
                selected_val = current_price 

                if hist:
                    # Sort using the robust parser
                    sorted_hist = sorted(hist, key=lambda x: parse_shoonya_time(x['time']))
                    
                    for candle in reversed(sorted_hist):
                        # Use the robust parser here too
                        c_time = parse_shoonya_time(candle['time'])
                        
                        if iv == 'd':
                            c_date = datetime.datetime.fromtimestamp(c_time).date()
                            today_date = datetime.datetime.now().date()
                            if c_date < today_date:
                                selected_val = float(candle['intc'])
                                break
                        else:
                            candle_close_time = c_time + duration
                            if candle_close_time <= current_ts:
                                selected_val = float(candle['intc'])
                                break
                
                packet.append(selected_val)

            # --- C. Final Assembly ---
            packet.append(current_price)
            
            # Normalize
            normalized_packet = [(p - current_price) / current_price for p in packet]
            
            return normalized_packet, current_price, True

        except Exception as e:
            # Catch JSON errors or connection drops silently
            if "Expecting value" in str(e):
                logger.warning("Shoonya API Maintenance (JSON Error). Sleeping.")
            else:
                logger.error(f"Fetch Error: {e}")
            self.login()
            return None, None, False

    def get_input_tensor(self):
        data_list = list(self.history)
        if len(data_list) < INPUT_SEQ_LEN:
            missing = INPUT_SEQ_LEN - len(data_list)
            padding = [[0.0] * INPUT_FEATURES for _ in range(missing)]
            data_list = padding + data_list
        return torch.tensor([data_list], dtype=torch.float32)

# ==========================================
# 5. THE BRAIN
# ==========================================
class ProjectP_Net(nn.Module):
    def __init__(self):
        super(ProjectP_Net, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    keep_alive()
    bot = ProjectP_Net()
    optimizer = optim.Adam(bot.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    data_manager = DataManager()
    win_loss_queue = deque(maxlen=1000) 
    pending_trades = []                 

    logger.info("--- PROJECT P: REBOOTED ---")
    send_telegram("🚀 PROJECT P: Online.\nStatus: Monitoring Market...")

    while True:
        loop_start = time.time()
        
        # 1. Sense
        packet, current_price, is_active = data_manager.fetch_market_data()
        
        if is_active and packet:
            # MARKET IS OPEN AND MOVING
            data_manager.history.append(packet)
            
            # 2. Learn
            for trade in pending_trades[:]:
                if loop_start - trade['timestamp'] >= DELAY_SECONDS:
                    diff = current_price - trade['entry_price']
                    
                    # Ignore trades with 0.0 difference (prevents fake losses on flat markets)
                    if abs(diff) < 0.5: 
                        pending_trades.remove(trade)
                        continue

                    did_win = (trade['action'] == 1 and diff > 0) or (trade['action'] == 0 and diff < 0)
                    win_loss_queue.append(1 if did_win else 0)
                    
                    optimizer.zero_grad()
                    pred = bot(trade['input_tensor'])
                    target = pred.clone().detach()
                    target[0][trade['action']] = 1.0 if did_win else -1.0
                    loss = loss_fn(pred, target)
                    loss.backward()
                    optimizer.step()
                    pending_trades.remove(trade)
                    logger.info(f" > Training: {'WIN' if did_win else 'LOSS'} (Diff: {diff:.2f})")

            # 3. Predict
            if len(data_manager.history) > 0:
                with torch.no_grad():
                    input_tensor = data_manager.get_input_tensor()
                    action = torch.argmax(bot(input_tensor)).item()
                
                direction = "UP 🟢" if action == 1 else "DOWN 🔴"
                win_rate = f"{(sum(win_loss_queue)/len(win_loss_queue))*100:.1f}%" if len(win_loss_queue) >= MIN_TRADES_STATS else f"Calibrating ({len(win_loss_queue)})"

                msg = (f"📡 P-REPORT\nPred: {direction}\nPrice: {current_price}\nWinRate: {win_rate}")
                logger.info(f"Pred: {direction} | Price: {current_price} | WR: {win_rate}")
                send_telegram(msg)
                
                pending_trades.append({
                    'timestamp': loop_start,
                    'input_tensor': input_tensor,
                    'action': action,
                    'entry_price': current_price
                })
        else:
            # MARKET IS CLOSED / STAGNANT
            logger.info(f"💤 Market Idle. Last Price: {current_price if current_price else 'Unknown'}")
            pass

        # 4. Wait
        elapsed = time.time() - loop_start
        time.sleep(max(0, 300 - elapsed))