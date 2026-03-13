import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import random
from flask import Flask
from threading import Thread
import logging
import sys
import datetime
import pyotp
import json
from NorenRestApiPy.NorenApi import NorenApi
import pymongo
import io
from bson.binary import Binary

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
    return "PROJECT P: SUPERVISED LEARNING ONLINE"

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
MONGO_URI = os.environ.get("MONGO_URI", "")  # Add your MongoDB URI here

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
LEARNING_RATE = 0.005  
DELAY_SECONDS = 1800    
MIN_TRADES_STATS = 50  
BATCH_SIZE = 4          

# ==========================================
# 3. HELPER FUNCTIONS (Including MongoDB DB Logic)
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
    try:
        return float(time_str)
    except ValueError:
        try:
            dt_obj = datetime.datetime.strptime(time_str, '%d-%m-%Y %H:%M:%S')
            return dt_obj.timestamp()
        except Exception:
            return time.time()

# MongoDB Setup
db_client = None
weights_collection = None
if MONGO_URI:
    try:
        db_client = pymongo.MongoClient(MONGO_URI)
        db = db_client["ProjectP_DB"]
        weights_collection = db["model_weights"]
        logger.info("MongoDB Connected Successfully.")
    except Exception as e:
        logger.error(f"MongoDB Connection Failed: {e}")

def load_weights_from_mongo(model):
    """Pulls the binary weights from MongoDB and loads them into the model."""
    if weights_collection is None:
        return False
    try:
        doc = weights_collection.find_one({"_id": "latest_weights"})
        if doc and "weights" in doc:
            buffer = io.BytesIO(doc["weights"])
            state_dict = torch.load(buffer, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Model weights successfully loaded from MongoDB.")
            return True
        else:
            logger.info("No existing weights found in MongoDB. Starting fresh.")
            return False
    except Exception as e:
        logger.error(f"Failed to load weights from MongoDB: {e}")
        return False

def save_weights_to_mongo(model):
    """Saves the current model weights to MongoDB as a binary stream."""
    if weights_collection is None:
        return
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        binary_weights = Binary(buffer.read())
        
        # Upsert ensures we overwrite the old weights instead of creating new documents
        weights_collection.update_one(
            {"_id": "latest_weights"},
            {"$set": {"weights": binary_weights, "updated_at": time.time()}},
            upsert=True
        )
        logger.info("Model weights backed up to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to save weights to MongoDB: {e}")

# ==========================================
# 4. DATA MANAGEMENT
# ==========================================
class DataManager:
    def __init__(self):
        self.history = deque(maxlen=INPUT_SEQ_LEN)
        self.api = NorenApi(host='https://api.shoonya.com/NorenWClientTP/', 
                            websocket='wss://api.shoonya.com/NorenWSTP/')
        self.last_login_time = 0
        self.last_price = 0.0 
        self.login()

    def login(self):
        try:
            if not SHOONYA_TOTP_SECRET or not SHOONYA_USER_ID:
                return False
            totp = pyotp.TOTP(SHOONYA_TOTP_SECRET).now()
            ret = self.api.login(
                userid=SHOONYA_USER_ID, password=SHOONYA_PASSWORD, 
                twoFA=totp, vendor_code=SHOONYA_VC, api_secret=SHOONYA_API_KEY, imei='12345'
            )
            if ret and 'stat' in ret and ret['stat'] == 'Ok':
                self.last_login_time = time.time()
                return True
            return False
        except Exception:
            return False

    def fetch_market_data(self):
        if time.time() - self.last_login_time > 72000: self.login()
        packet = []
        current_price = 0.0
        current_ts = time.time()

        try:
            quote = self.api.get_quotes(exchange=EXCHANGE, token=TOKEN_NIFTY)
            if quote and 'lp' in quote:
                current_price = float(quote['lp'])
            else:
                return None, None, False

            if current_price == self.last_price:
                 return None, current_price, False
            self.last_price = current_price

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
                try:
                    hist = self.api.get_time_price_series(exchange=EXCHANGE, token=TOKEN_NIFTY, starttime=start_ts, interval=iv)
                except Exception:
                    hist = None
                
                selected_val = current_price 
                if hist:
                    sorted_hist = sorted(hist, key=lambda x: parse_shoonya_time(x['time']))
                    for candle in reversed(sorted_hist):
                        c_time = parse_shoonya_time(candle['time'])
                        if iv == 'd':
                            if datetime.datetime.fromtimestamp(c_time).date() < datetime.datetime.now().date():
                                selected_val = float(candle['intc'])
                                break
                        else:
                            if c_time + duration <= current_ts:
                                selected_val = float(candle['intc'])
                                break
                packet.append(selected_val)

            packet.append(current_price)
            normalized_packet = [(p - current_price) / current_price for p in packet]
            return normalized_packet, current_price, True

        except Exception as e:
            if "Expecting value" not in str(e):
                logger.error(f"Fetch Error: {e}")
            self.login()
            return None, None, False

    def get_IT_tensor(self):
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
    loss_fn = nn.CrossEntropyLoss() 
    
    # Check Mongo for existing weights and load them before starting
    load_weights_from_mongo(bot)
    
    data_manager = DataManager()
    queue_E = deque(maxlen=4)                 
    win_loss_queue = deque(maxlen=1000)       
    pending_IT_objects = []                   

    logger.info("--- PROJECT P: SUPERVISED LEARNING ONLINE ---")
    send_telegram("🚀 PROJECT P: Core Architecture Upgraded.\nStatus: Monitoring Market...")

    while True:
        loop_start = time.time()
        
        packet, current_price, is_active = data_manager.fetch_market_data()
        
        if is_active and packet:
            data_manager.history.append(packet)
            
            for item in pending_IT_objects[:]:
                if loop_start - item['timestamp'] >= DELAY_SECONDS:
                    diff = current_price - item['entry_price']
                    
                    if abs(diff) < 0.5: 
                        pending_IT_objects.remove(item)
                        continue

                    label = 1 if diff > 0 else 0
                    Q = (item['it_tensor'], label)
                    queue_E.append(Q)
                    
                    did_win = (item['pred_action'] == label)
                    win_loss_queue.append(1 if did_win else 0)
                    
                    pending_IT_objects.remove(item)
                    logger.info(f" > Object Q created. Label: {label}. Prediction was: {'WIN' if did_win else 'LOSS'}")

            # Supervised Batch Training (1 time per cycle, 2 Q objects per batch)
            if len(queue_E) >= BATCH_SIZE:
                batch = random.sample(list(queue_E), 2)
                
                inputs = torch.cat([b[0] for b in batch], dim=0) 
                labels = torch.tensor([b[1] for b in batch], dtype=torch.long) 
                
                optimizer.zero_grad()
                preds = bot(inputs) 
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
            
                logger.info(f" > Batch Training Executed. Loss: {loss.item():.4f}")
                
                # Automatically save the newly updated weights to MongoDB
                save_weights_to_mongo(bot)

            if len(data_manager.history) > 0:
                with torch.no_grad():
                    it_tensor = data_manager.get_IT_tensor()
                    action = torch.argmax(bot(it_tensor)).item()
                
                direction = "UP 🟢" if action == 1 else "DOWN 🔴"
                win_rate = f"{(sum(win_loss_queue)/len(win_loss_queue))*100:.1f}%" if len(win_loss_queue) >= MIN_TRADES_STATS else f"Calibrating ({len(win_loss_queue)})"

                msg = (f"📡 P-REPORT\nPred: {direction}\nPrice: {current_price}\nWinRate: {win_rate}")
                logger.info(f"Pred: {direction} | Price: {current_price} | WR: {win_rate}")
                send_telegram(msg) 
                
                pending_IT_objects.append({
                    'timestamp': loop_start,
                    'it_tensor': it_tensor,
                    'pred_action': action,
                    'entry_price': current_price
                })
        else:
            logger.info(f"💤 Market Idle. Last Price: {current_price if current_price else 'Unknown'}")

        elapsed = time.time() - loop_start
        time.sleep(max(0, 300 - elapsed))