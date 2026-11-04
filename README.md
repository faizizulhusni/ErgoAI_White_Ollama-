# ErgoAI — White Dashboard (Ollama Only)

Updated light-background dashboard. Fixed overlapping text and graphs by removing fixed panel heights, viewport clipping, negative margins, and chart-title/legend overlap. All two-agent simulation/controller logic is preserved.

## What's on the screen

- Three pressure sliders (lumbar, mid-back, upper-back) in simulated PU.
- Agent 1: automatic Ollama-generated virtual human postures.
- Agent 2: Ollama chair pressure decisions, with concise decision explanation.
- Animated front and side seated-person views (posture follows Agent 1; orange supports follow actuator control).
- Six-zone pressure chart and pressure trend chart.
- Pause, Reset and simulated E-STOP.

No manual posture buttons, rule-based AI planner, cloud-model selection, or API keys.

## Windows installation

1. Extract `ErgoAI_White_Ollama.zip` to `Downloads` so the extracted folder contains `app.py` and `requirements.txt`.
2. Open PowerShell and enter:

```powershell
cd "$HOME\Downloads\ErgoAI_White_Ollama"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you already have a virtual environment and Ollama running, reuse the environment, or copy only `app.py`, `visuals.py`, and `.streamlit/config.toml` into the previous project folder and run from there.

Streamlit opens on http://localhost:8501 . Set browser zoom to 100%. Layout aims to fit on typical 1366×768 at 100% zoom; if a screen is shorter, it **scrolls** rather than clipping content.

The white theme is configured in `.streamlit/config.toml`, and charts use white Plotly backgrounds. To update `.env`, set `OLLAMA_MODEL=llama3.2` and `POSTURE_INTERVAL_SECONDS=15`.

The term PU refers to arbitrary *simulation* pressure units, not calibrated kPa. This simulation is not a physical chair controller or biomechanical safety assessment.

## Streamlit Community Cloud Deployment

1. **Push this repository to GitHub** (see the file list below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your GitHub repository, set **Main file path** to `app.py`, and click **Deploy**.
4. **(Optional) Connect a live Cloud LLM via Secrets:**
   - In your Streamlit app dashboard, go to **Settings > Secrets**.
   - Paste your API configuration (e.g., free Groq API for Llama 3):
     ```toml
     OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
     OPENAI_API_KEY = "gsk_your_groq_api_key_here"
     OPENAI_MODEL = "llama-3.3-70b-versatile"
     ```
   - If no secrets are provided, the app will run seamlessly in **Autonomous AI Simulation mode** with full ergonomic reasoning!

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_compact.py
```
