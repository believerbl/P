# Project P

**Project P** stands for *Probability* — specifically **P(t+30)**.

It is an autonomous, self-correcting algorithmic trading system built around a single binary classification problem:

> **Will the market be higher or lower 30 minutes from now?**

Unlike a traditional "train once, deploy once" ML pipeline, Project P is an **online learning system** — it makes a live prediction, waits to see what actually happened, learns from the outcome, and updates its own weights continuously while running. There is no separate offline training phase; the model learns in production, on live market data, forever.

---

## How It Works

Project P runs a continuous loop, once every 5 minutes, against NIFTY (NSE index):

1. **Fetch live market data** — pulls the current quote and recent historical candles across four timeframes (daily, 60-min, 30-min, 5-min) from the Shoonya broker API.
2. **Build a feature vector** — normalizes each timeframe's closing price relative to the current price, producing a 5-feature snapshot of "where price stands" across multiple horizons.
3. **Maintain a rolling sequence** — the last 10 snapshots are kept in a sliding window (`deque`), forming the input sequence for the LSTM.
4. **Predict** — the LSTM model outputs a binary prediction: price will be **UP** or **DOWN** in 30 minutes.
5. **Wait and label** — each prediction is stored as a *pending object*. After 30 minutes (`DELAY_SECONDS`), the system checks the actual price movement and assigns a ground-truth label (1 = up, 0 = down).
6. **Learn online** — labeled examples are added to a replay-style queue. Once enough examples accumulate, the model trains on a small batch (2 samples) using cross-entropy loss and backpropagation — live, without stopping the system.
7. **Persist weights** — after every training step, the updated model weights are serialized and saved to MongoDB, so the model survives restarts and redeployments with its learned state intact.
8. **Report** — every cycle, a status message (prediction, current price, and rolling win rate) is sent to a Telegram chat for real-time monitoring.

This creates a closed feedback loop: **predict → wait → grade itself → learn → predict again**, entirely unattended.

---

## Model Architecture

The core model (`ProjectP_Net`) is a lightweight 2-layer LSTM classifier:

```
Input (10 timesteps × 5 features)
        │
   2-layer LSTM (hidden size = 64)
        │
  Linear(64 → 32) → ReLU
        │
     Linear(32 → 2)
        │
  [P(down), P(up)]
```

- **Input:** a sequence of 10 normalized price snapshots, each with 5 features (daily / 60-min / 30-min / 5-min / current price, all normalized relative to current price).
- **Output:** a 2-class prediction (down / up), trained with `CrossEntropyLoss`.
- **Optimizer:** Adam, learning rate `0.005`.

---

## Key Features

- **Online / continual learning** — the model updates itself on every new labeled outcome instead of relying on a static, pre-trained checkpoint.
- **Self-labeling** — no manually labeled dataset is needed; the market itself provides the ground truth 30 minutes after each prediction.
- **Persistent memory across restarts** — model weights are stored in MongoDB (as binary blobs) and reloaded automatically on startup, so the bot resumes learning from where it left off.
- **Live broker integration** — connects directly to the Shoonya (Finvasia) trading API via `NorenRestApiPy`, including automated TOTP-based two-factor login.
- **Multi-timeframe feature engineering** — blends daily, hourly, 30-min, and 5-min price context into a single feature vector per timestep.
- **Real-time Telegram reporting** — sends live predictions, current price, and rolling win-rate statistics to a Telegram chat every cycle.
- **Cloud keep-alive** — runs a minimal Flask server alongside the main loop so the bot can be kept alive on always-on hosting platforms (e.g. Render) that ping an HTTP endpoint.
- **Resilient session handling** — automatically re-authenticates with the broker if the session expires or a data fetch fails.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Framework | PyTorch (`torch.nn`, LSTM) |
| Broker API | Shoonya / Finvasia (`NorenRestApiPy`) |
| Authentication | TOTP-based 2FA (`pyotp`) |
| Persistence | MongoDB (`pymongo`) — binary weight storage |
| Notifications | Telegram Bot API |
| Keep-alive server | Flask |
| Deployment target | Any always-on Python host (e.g. Render) |

---

## Project Structure

```
P-main/
├── main.py            # Full system: data pipeline, model, training loop, Telegram + Mongo integration
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Project P is fully configured via environment variables — no secrets are hardcoded.

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Telegram bot token for sending status updates |
| `TELEGRAM_CHAT_ID` | Telegram chat ID to send updates to |
| `MONGO_URI` | MongoDB connection string, used to persist/restore model weights |
| `SHOONYA_USER_ID` | Shoonya broker user ID |
| `SHOONYA_PASSWORD` | Shoonya broker password |
| `SHOONYA_TOTP_SECRET` | TOTP secret used to generate 2FA codes automatically |
| `SHOONYA_VC` | Shoonya vendor code |
| `SHOONYA_API_KEY` | Shoonya API secret/key |
| `PORT` | *(optional)* Port for the Flask keep-alive server (default: `8080`) |

### 3. Run
```bash
python main.py
```

The bot will:
- Start a background Flask server (for host keep-alive pings)
- Attempt to restore previously learned weights from MongoDB
- Log in to the Shoonya broker API
- Begin the continuous predict → wait → learn loop, reporting to Telegram every cycle

---

## Notes & Disclaimer

- Project P is a **research/experimental system**. It trades on live NIFTY price data and makes real directional predictions, but this repository does not include actual order placement logic in `main.py` — it currently focuses on the prediction and self-learning loop rather than executing trades.
- Live financial markets are noisy and non-stationary; a small, continuously-updating LSTM trained on 2-sample batches is a lightweight, experimental approach and should not be treated as a production-grade trading strategy without significant further validation, backtesting, and risk controls.
- Do not commit real credentials (`SHOONYA_*`, `TELEGRAM_*`, `MONGO_URI`) to version control — always supply them as environment variables or secrets in your deployment platform.

## Author

Built by [Parimarjan Shukla](https://github.com/believerbl).
