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

# ==========================================
# 1. FLASK KEEP-ALIVE (For Render)
# ==========================================
app = Flask('')

@app.route('/')
def home():
    return "PROJECT P: SYSTEMS NOMINAL"

def run_http():
    # Render automatically sets the 'PORT' environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    """Starts the lightweight web server in a background thread."""
    t = Thread(target=run_http)
    t.start()

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Retrieve secrets from Render Environment Variables
API_KEY = os.environ.get("TWELVEDATA_API_KEY", "YOUR_API_KEY_HERE")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

SYMBOL = "NSE/NIFTY"
TIMEFRAMES = ["1day", "4h", "1h", "5min"] 

# Hyperparameters
INPUT_SEQ_LEN = 10     # Look back 10 steps (50 mins)
INPUT_FEATURES = 5     # 5 Data points per step
HIDDEN_SIZE = 64       # LSTM Memory Size
LEARNING_RATE = 0.001
DELAY_SECONDS = 1800   # 30 Minutes (Learning Delay)
MIN_TRADES_STATS = 50  # Trades needed before showing Win Rate

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def send_telegram(message):
    """Sends a notification to your Telegram App."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"Telegram not configured. Log: {message}")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ==========================================
# 4. DATA MANAGEMENT (The Senses)
# ==========================================
class DataManager:
    def __init__(self):
        # Rolling buffer for the last 10 normalized market snapshots
        self.history = deque(maxlen=INPUT_SEQ_LEN)

    def fetch_market_data(self):
        """
        Fetches current price + candles for all timeframes.
        Normalizes data to % change relative to current price.
        """
        packet = []
        current_price = 0.0
        
        try:
            # A. Fetch Current Price
            url_price = f"https://api.twelvedata.com/price?symbol={SYMBOL}&apikey={API_KEY}"
            resp_price = requests.get(url_price).json()
            
            if 'price' not in resp_price:
                print(f"[Error] API Fetch Failed: {resp_price}")
                return None, None
                
            current_price = float(resp_price['price'])

            # B. Fetch Candles (Sequentially to ensure data integrity)
            for interval in TIMEFRAMES:
                url_ts = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={interval}&outputsize=1&apikey={API_KEY}"
                resp_ts = requests.get(url_ts).json()
                
                if 'values' in resp_ts:
                    close_val = float(resp_ts['values'][0]['close'])
                    packet.append(close_val)
                else:
                    # Fallback to current price if specific timeframe fails
                    packet.append(current_price)
            
            # Add Current Price as the 5th feature
            packet.append(current_price)

            # C. Normalization (Critical for LSTM)
            # Converts raw prices (21000) into small relative values (0.001)
            normalized_packet = [(p - current_price) / current_price for p in packet]
            
            return normalized_packet, current_price

        except Exception as e:
            print(f"[Error] Exception in fetch: {e}")
            return None, None

    def get_input_tensor(self):
        """Prepares the 2D tensor (Batch=1, Seq=10, Feat=5) for the AI."""
        data_list = list(self.history)
        
        # Zero Padding if we don't have 10 data points yet
        if len(data_list) < INPUT_SEQ_LEN:
            missing = INPUT_SEQ_LEN - len(data_list)
            padding = [[0.0] * INPUT_FEATURES for _ in range(missing)]
            data_list = padding + data_list
            
        return torch.tensor([data_list], dtype=torch.float32)

# ==========================================
# 5. THE BRAIN (LSTM-RL Hybrid)
# ==========================================
class ProjectP_Net(nn.Module):
    def __init__(self):
        super(ProjectP_Net, self).__init__()
        
        # LSTM Layer: Extracts temporal patterns from the sequence
        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=2, 
            batch_first=True
        )
        
        # Fully Connected Layer: Makes the final UP/DOWN decision
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: [Value_Down, Value_Up]
        )

    def forward(self, x):
        # x shape: (1, 10, 5)
        lstm_out, _ = self.lstm(x)
        # Take the output from the LAST time step only
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

# ==========================================
# 6. MAIN SYSTEM LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Ignite Web Server (Required for Render)
    keep_alive()
    
    # 2. Initialize System
    bot = ProjectP_Net()
    optimizer = optim.Adam(bot.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    data_manager = DataManager()
    win_loss_queue = deque(maxlen=1000) # Performance Database
    pending_trades = []                 # Delayed Reward Buffer

    # 3. Branding Startup
    print("--- PROJECT P: ONLINE ---")
    send_telegram("🚀 PROJECT P: Online.\nSystem: LSTM-RL Hybrid\nTarget: Nifty 50\nState: Initializing...")

    # 4. Infinite Loop
    while True:
        loop_start = time.time()
        
        # --- PHASE 1: SENSE (Ingest Data) ---
        packet, current_price = data_manager.fetch_market_data()
        
        if packet:
            data_manager.history.append(packet)
            
            # --- PHASE 2: LEARN (Check Past Predictions) ---
            # We iterate a copy [:] to safely modify the list while looping
            for trade in pending_trades[:]:
                if loop_start - trade['timestamp'] >= DELAY_SECONDS:
                    
                    # A. Calculate Reality
                    diff = current_price - trade['entry_price']
                    
                    # B. Determine Win/Loss
                    # Action 1 (UP) wins if diff > 0. Action 0 (DOWN) wins if diff < 0.
                    did_win = (trade['action'] == 1 and diff > 0) or \
                              (trade['action'] == 0 and diff < 0)
                    
                    # C. Update Database
                    win_loss_queue.append(1 if did_win else 0)
                    
                    # D. Train the Brain (Backpropagation)
                    optimizer.zero_grad()
                    
                    # Re-run the specific input that caused this prediction
                    pred_q = bot(trade['input_tensor'])
                    target_q = pred_q.clone().detach()
                    
                    # Reward = +1.0 for Win, -1.0 for Loss
                    reward = 1.0 if did_win else -1.0
                    target_q[0][trade['action']] = reward
                    
                    loss = loss_fn(pred_q, target_q)
                    loss.backward()
                    optimizer.step()
                    
                    # E. Cleanup
                    pending_trades.remove(trade)
                    print(f" > Verifying T-30m: {'WIN' if did_win else 'LOSS'} (Reward: {reward})")

            # --- PHASE 3: PREDICT (Future Probability) ---
            # Only predict if we have gathered enough initial data
            if len(data_manager.history) > 0:
                with torch.no_grad():
                    input_tensor = data_manager.get_input_tensor()
                    q_values = bot(input_tensor)
                    action = torch.argmax(q_values).item()
                
                # Formatting the Report
                direction = "UP 🟢" if action == 1 else "DOWN 🔴"
                
                if len(win_loss_queue) >= MIN_TRADES_STATS:
                    win_rate = f"{(sum(win_loss_queue)/len(win_loss_queue))*100:.1f}%"
                else:
                    win_rate = f"Calibrating ({len(win_loss_queue)}/{MIN_TRADES_STATS})"

                # Telegram Report
                msg = (
                    f"📡 PROJECT P REPORT\n"
                    f"-------------------\n"
                    f"Prediction (T+30): {direction}\n"
                    f"Current Price: {current_price}\n"
                    f"Win Rate: {win_rate}\n"
                    f"Active Memory: {len(data_manager.history)}/10"
                )
                
                print(msg)
                send_telegram(msg)
                
                # Store for future validation
                pending_trades.append({
                    'timestamp': loop_start,
                    'input_tensor': input_tensor,
                    'action': action,
                    'entry_price': current_price
                })

        else:
            print("Data Fetch Error. Retrying next cycle.")

        # --- PHASE 4: WAIT ---
        # Sleep for remainder of 5 minutes (300 seconds)
        elapsed = time.time() - loop_start
        sleep_duration = max(0, 300 - elapsed)
        print(f"Sleeping for {int(sleep_duration)}s...")
        time.sleep(sleep_duration)