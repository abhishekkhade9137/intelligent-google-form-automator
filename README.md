# 🤖 Intelligent AI Google Form Automator & Psychometric Survey Suite

A production-grade, advanced AI survey data synthesis and Google Form automation platform built with clean **Domain-Driven Design (DDD)**. Powered by **Ollama Cloud Models**, **Google Gemini**, or **OpenAI**, this platform integrates formal mathematical psychometrics, Gaussian Copulas, empirical benchmark rebalancing, and human diurnal temporal scheduling to generate datasets that are indistinguishably authentic and statistically verified.

---

## ✨ Key Architectural Capabilities

### 1. 🏛️ Domain-Driven Design (DDD) & Modular Engineering
- **`src/domain/`**: Immutable type-safe data models and schema entities (`FormSchema`, `Question`, `Persona`, `ScheduledSubmissionTask`).
- **`src/automation/`**: Playwright browser engines, live form DOM extractors, temporal scheduling queues, and multithreaded worker swarms.
- **`src/synthesis/`**: Generative engines, multi-pattern email handle generators, and Bayesian Latent Sentiment Oracles.
- **`src/statistical/`**: Real-world demographic preset profiles, Hare-Niemann deterministic quota rebalancing, human anomaly noise injection, and Gaussian Copulas.
- **`src/persistence/`**: SQLite audit ledgers and persistent job queues with zero data loss across restarts.

### 2. 🧠 Multi-Dimensional Psychometrics & Gaussian Copula Correlation
- **Beyond Flat Randomness**: Employs **Cholesky matrix decomposition** ($\Sigma = L \cdot L^T$) to sample multi-dimensional behavioral Z-scores (`tech_enthusiasm`, `risk_skepticism`, `pragmatic_workflow`).
- **Psychological Consistency**: Guarantees organic correlations—such as ensuring a respondent who scores high on privacy risk skepticism systematically expresses caution regarding AI code licensing and organizational mandatory adoption without brittle rule-based if/else statements.

### 3. 🕒 Believable All-Day Temporal Scheduling & Concurrency Swarms
- **Human Diurnal Activity Timers**: Avoids unnatural traffic clusters by generating timestamp delivery schedules weighted across natural human daily cycles (nocturnal rest between 01:00–06:00, strong engagement surges during lunch 13:00–14:30 and evening peak hours 18:00–22:00).
- **Multithreaded Worker Swarms**: Launches bounded pools of Playwright browser instances (`max_workers=3`) to execute simultaneous traffic bursts during peak hours without IP blocking or browser memory leaks.

### 4. 📧 Authentic Multi-Strategy Email Synthesis Engine
Prevents repetitive name-based email patterns by utilizing four weighted real-world generation strategies:
- **Standard Name-Based Format (45%)**: Traditional combinations (`rohit.sharma24@gmail.com`).
- **Abstract Tech & Gamer Handles (25%)**: Anonymous internet personas (`pixel_coder99@gmail.com`, `dark_nebula42@yahoo.com`).
- **Alphanumeric Privacy Tags (15%)**: Disposable identifiers (`anon6411733@gmail.com`).
- **Institutional & Campus Domains (15%)**: Academic & enterprise domains (`pranav.b2024@vitstudent.ac.in`, `hritik.dyal@cognizant.com`).

### 5. 🎛️ Interactive Pre-Flight Dashboard & Percentage Customizer
Includes an interactive **Streamlit Web UI** (`app.py`) featuring a specialized **Pre-Flight Customizer & All-Day Scheduler** tab:
1. **Inspect & Load Questions**: Connects to your Google Form link via headless Chrome to extract all questions and categorical choices instantly.
2. **Override Percentage Distributions**: Use intuitive sliders to override target distribution percentages for any Multiple Choice, Checkbox, or Dropdown control before submission.
3. **Deploy & Schedule**: Queue up to 500 respondents over customized time horizons (e.g. 12 to 48 hours) directly into a persistent SQLite queue and run worker swarm cycles on demand.

---

## 🚀 Quick Setup & Installation

### 1. Install Python Dependencies
Ensure you have Python 3.10+ installed, then install packages and browser binaries:
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 2. Configure AI Providers (Ollama / Gemini / OpenAI)
- **Ollama Cloud & Local Models**: Natively supports cloud-routed Ollama models such as `glm-5.2:cloud`, `nemotron-3-super:cloud`, and `kimi-k2.7-code:cloud` or entirely offline local LLMs like `llama3.1`, `gemma2`, and `mistral`.
- **Google Gemini**: Set your environment variable:
  ```powershell
  $env:GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
  ```

---

## 🖥️ Usage Guide

### Method A: Streamlit Web Dashboard (Recommended)
Launch the rich interactive UI with built-in analytics, empirical distribution verification, and schedule managers:
```bash
streamlit run app.py
```
- Navigate to **"🚀 Campaign Runner & Live Telemetry"** for immediate real-time batch filling.
- Navigate to **"🎛️ Pre-Flight Customizer & All-Day Scheduler"** to discover form questions, edit option percentage targets via sliders, and queue all-day automated submissions.
- Navigate to **"📊 Analytics & Mathematical Distributions"** to visualize dataset histograms, examine email diversity metrics, and download complete SQLite audit trails as `.csv` spreadsheets.

### Method B: Standalone Terminal Scripts & CLI
For headless batch runs directly from your command line:
```bash
# Execute pre-configured VIT Vellore Survey Campaign Script
python scripts/run_vit_campaign.py

# Run via modular CLI runner
python -m src.cli run --url "https://docs.google.com/forms/d/e/YOUR_FORM_ID/viewform" --count 50 --provider ollama --model glm-5.2:cloud
```

---

## 🧪 Comprehensive Unit Testing Suite
The modular codebase includes **21 automated unit tests** spanning domain serialization, email strategies, latent probability oracles, psychometrics, diurnal scheduling, and database persistence ledgers:
```bash
python -m pytest -v tests/
```

---

## 🔒 Handling Forms Requiring Google Account Sign-In
If your target Google Form restricts access to signed-in accounts or institutional organizations:
1. Ensure headless mode is unchecked in the dashboard (or set `--headless false` in CLI) so Chrome opens visibly.
2. The automation engine will detect the Google OAuth sign-in wall and pause cleanly with a helpful prompt.
3. Sign into your Google Account once in the browser. Your session profile and cookies are safely persisted inside `./chrome_session_profile/`, ensuring all subsequent automated swarms run instantly without re-authentication!
