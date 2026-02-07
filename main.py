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
# 1. FLASK KEEP-ALIVE (For Render)
# ==========================================
app = Flask('')

@app.route('/')
def home():
    return "PROJECT P: SHOONYA LINK ACTIVE"

def run_http():
    port = int(os.environ.get("PORT", 8080))
    # Silence Flask logs to keep console clean
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    """Starts the fake web server to satisfy Render."""
    t = Thread(target=run_http)
    t.start()

# ==========================================
# 2. CONFIGURATION & CREDENTIALS
# ==========================================
# Telegram Keys
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Shoonya (Finvasia) Keys
SHOONYA_USER_ID = os.environ.get("SHOONYA_USER_ID", "YOUR_USER_ID")
SHOONYA_PASSWORD = os.environ.get("SHOONYA_PASSWORD", "YOUR_PWD")
SHOONYA_TOTP_SECRET = os.environ.get("SHOONYA_TOTP_SECRET", "YOUR_TOTP")
SHOONYA_VC = os.environ.get("SHOONYA_VC", "YOUR_VC_CODE")
SHOONYA_API_KEY = os.environ.get("SHOONYA_API_KEY", "YOUR_API_KEY")

# Market Data Config
# Token 26000 is the hardcoded ID for Nifty 50 Index on NSE
TOKEN_NIFTY = '26000' 
EXCHANGE = 'NSE'

# Project P Hyperparameters
INPUT_SEQ_LEN = 10     # Look back 10 steps
INPUT_FEATURES = 5     # Daily, 60m, 30m, 5m, Current Price
HIDDEN_SIZE = 64       
LEARNING_RATE = 0.001
DELAY_SECONDS = 1800   # 30 Minutes
MIN_TRADES_STATS = 50  

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def send_telegram(message):
    """Sends a notification to your Telegram App."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(f"Telegram not configured. Log: {message}")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Telegram Error: {e}")

# ==========================================
# 4. DATA MANAGEMENT (SHOONYA API)
# ==========================================
class DataManager:
    def __init__(self):
        self.history = deque(maxlen=INPUT_SEQ_LEN)
        # Initialize NorenApi
        self.api = NorenApi(host='https://api.shoonya.com/NorenWClientTP/', 
                           websocket='wss://api.shoonya.com/NorenWSTP/')
        self.last_login_time = 0
        
        # Initial Login
        self.login()

    def login(self):
        """Handles the TOTP generation and login handshake."""
        try:
            logger.info("Generating TOTP and logging in to Shoonya...")
            
            if not SHOONYA_TOTP_SECRET or not SHOONYA_USER_ID:
                logger.error("Shoonya Credentials missing in Environment Variables!")
                return False

            # Generate TOTP using the secret
            totp = pyotp.TOTP(SHOONYA_TOTP_SECRET).now()
            
            # Login call
            ret = self.api.login(
                userid=SHOONYA_USER_ID, 
                password=SHOONYA_PASSWORD, 
                twoFA=totp, 
                vendor_code=SHOONYA_VC, 
                api_secret=SHOONYA_API_KEY, 
                imei='12345'
            )
            
            if ret and 'stat' in ret and ret['stat'] == 'Ok':
                logger.info("✅ Shoonya Login Successful")
                self.last_login_time = time.time()
                return True
            else:
                logger.error(f"❌ Login Failed. Response: {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Login Exception: {e}")
            return False

    def fetch_market_data(self):
        """
        Fetches: [Daily, 60m, 30m, 5m, Current_Price]
        Returns normalized packet and raw current price.
        Includes Weekend Fallback Logic.
        """
        # Session Refresh: If login is older than 20 hours, re-login
        if time.time() - self.last_login_time > 72000: 
            logger.info("Session expired. Re-logging in...")
            self.login()

        packet = []
        current_price = 0.0
        
        try:
            # --- 1. FETCH CURRENT PRICE (With Fallback) ---
            # Attempt A: Live Quote
            quote = self.api.get_quotes(exchange=EXCHANGE, token=TOKEN_NIFTY)
            
            if quote and 'stat' in quote and quote['stat'] == 'Ok':
                if 'lp' in quote:
                    current_price = float(quote['lp'])
                else:
                    current_price = float(quote.get('c', 0.0)) # Close price if LP missing
            
            # Attempt B: Fallback to History (If Live Quote failed)
            if current_price == 0.0:
                logger.warning("Live Quote failed (Weekend/Closed?). Using History Fallback...")
                
                # Look back 5 days to find the last valid trading minute
                end_time = datetime.datetime.now()
                start_time = end_time - datetime.timedelta(days=5) 
                
                # Fetch 1-minute candles
                fallback_hist = self.api.get_time_price_series(
                    exchange=EXCHANGE, 
                    token=TOKEN_NIFTY, 
                    starttime=start_time.timestamp(), 
                    interval='1'
                )
                
                if fallback_hist:
                    # Sort by time and take the absolute last candle
                    sorted_fb = sorted(fallback_hist, key=lambda x: x['time'])
                    last_candle = sorted_fb[-1]
                    current_price = float(last_candle['intc'])
                    logger.info(f"✅ Fallback Successful. Last Known Price: {current_price}")
                else:
                    logger.error("❌ Critical: Both Quote and History Fallback failed.")
                    return None, None

            # --- 2. FETCH HISTORY CANDLES (Daily, 60m, 30m, 5m) ---
            # We fetch 5 days back to ensure we find data
            start_ts = (datetime.datetime.now() - datetime.timedelta(days=5)).timestamp()
            
            # Shoonya Interval Codes
            intervals = ['d', '60', '30', '5']
            
            for interval in intervals:
                hist = self.api.get_time_price_series(
                    exchange=EXCHANGE, 
                    token=TOKEN_NIFTY, 
                    starttime=start_ts, 
                    interval=interval
                )
                
                if hist:
                    # Sort and pick latest
                    sorted_hist = sorted(hist, key=lambda x: x['time'])
                    latest_candle = sorted_hist[-1]
                    close_val = float(latest_candle['intc'])
                    packet.append(close_val)
                else:
                    # Fallback if specific timeframe fails
                    packet.append(current_price)
            
            # 3. Add Current Price as the 5th Input
            packet.append(current_price)

            # 4. Normalization: [(Value - Current) / Current]
            normalized_packet = [(p - current_price) / current_price for p in packet]
            
            return normalized_packet, current_price

        except Exception as e:
            logger.error(f"Fetch Exception: {e}")
            # Try immediate re-login if fetch crashed
            self.login()
            return None, None

    def get_input_tensor(self):
        """Converts history deque into a PyTorch Tensor."""
        data_list = list(self.history)
        
        # Zero Padding if history is not full yet
        if len(data_list) < INPUT_SEQ_LEN:
            missing = INPUT_SEQ_LEN - len(data_list)
            padding = [[0.0] * INPUT_FEATURES for _ in range(missing)]
            data_list = padding + data_list
            
        return torch.tensor([data_list], dtype=torch.float32)

# ==========================================
# 5. THE BRAIN (LSTM-RL HYBRID)
# ==========================================
class ProjectP_Net(nn.Module):
    def __init__(self):
        super(ProjectP_Net, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=2, 
            batch_first=True
        )
        
        # Decision Layer
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: [Score_Down, Score_Up]
        )

    def forward(self, x):
        # x shape: (1, 10, 5)
        lstm_out, _ = self.lstm(x)
        # Take the output from the LAST time step only
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

# ==========================================
# 6. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Start Web Server (Render Requirement)
    keep_alive()
    
    # 2. Init AI
    bot = ProjectP_Net()
    optimizer = optim.Adam(bot.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    # 3. Init Data Manager (Logs in to Shoonya)
    data_manager = DataManager()
    win_loss_queue = deque(maxlen=1000) 
    pending_trades = []                 

    # 4. Startup Notification
    logger.info("--- PROJECT P: SHOONYA CONNECTED ---")
    send_telegram("🚀 PROJECT P: Online.\nSource: Shoonya API (Real-Time)\nTarget: Nifty 50 Index (Token 26000)")

    # 5. Infinite Loop
    while True:
        loop_start = time.time()
        
        # --- PHASE 1: SENSE ---
        packet, current_price = data_manager.fetch_market_data()
        
        if packet:
            data_manager.history.append(packet)
            
            # --- PHASE 2: LEARN (Delayed Reward) ---
            # Check trades made > 30 mins ago
            for trade in pending_trades[:]:
                if loop_start - trade['timestamp'] >= DELAY_SECONDS:
                    
                    diff = current_price - trade['entry_price']
                    
                    # Win Condition: (UP & Diff>0) OR (DOWN & Diff<0)
                    did_win = (trade['action'] == 1 and diff > 0) or \
                              (trade['action'] == 0 and diff < 0)
                    
                    win_loss_queue.append(1 if did_win else 0)
                    
                    # Backpropagation (Training)
                    optimizer.zero_grad()
                    pred = bot(trade['input_tensor'])
                    target = pred.clone().detach()
                    
                    reward = 1.0 if did_win else -1.0
                    target[0][trade['action']] = reward
                    
                    loss = loss_fn(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    pending_trades.remove(trade)
                    logger.info(f" > Verifying T-30m: {'WIN' if did_win else 'LOSS'} (Reward: {reward})")

            # --- PHASE 3: PREDICT ---
            if len(data_manager.history) > 0:
                with torch.no_grad():
                    input_tensor = data_manager.get_input_tensor()
                    q_values = bot(input_tensor)
                    action = torch.argmax(q_values).item()
                
                direction = "UP 🟢" if action == 1 else "DOWN 🔴"
                
                # Stats
                if len(win_loss_queue) >= MIN_TRADES_STATS:
                    win_rate = f"{(sum(win_loss_queue)/len(win_loss_queue))*100:.1f}%"
                else:
                    win_rate = f"Calibrating ({len(win_loss_queue)})"

                # Notification
                msg = (
                    f"📡 P-REPORT (Live)\n"
                    f"Pred: {direction}\n"
                    f"Price: {current_price}\n"
                    f"WinRate: {win_rate}"
                )
                logger.info(f"Pred: {direction} | Price: {current_price} | WR: {win_rate}")
                send_telegram(msg)
                
                # Store Prediction
                pending_trades.append({
                    'timestamp': loop_start,
                    'input_tensor': input_tensor,
                    'action': action,
                    'entry_price': current_price
                })
        else:
            logger.warning("Data Fetch Error (Shoonya). Retrying...")

        # --- PHASE 4: WAIT ---
        elapsed = time.time() - loop_start
        # Sleep remainder of 5 minutes (300s)
        time.sleep(max(0, 300 - elapsed))