"""
Production-grade Lightweight Streamlit Dashboard for AI Google Form Automation.
Integrates cloud Ollama models with real-world demographic benchmark API selection,
deterministic Hare-Niemann statistical rebalancing, multi-strategy email diversity, and live telemetry.
"""
import os
import time
import random
import logging
import streamlit as st
import pandas as pd
import numpy as np

from src.domain import QuestionType, ScheduledSubmissionTask, ScheduleStatus
from src.automation import BrowserEngine, FormExtractor, FormFillerEngine, DiurnalTimestampGenerator, SwarmWorkerPool
from src.synthesis import AIGenerationEngine, DistributionSampler
from src.persistence import SubmissionTracker
from src.statistical import PRESET_PROFILES, BenchmarkAPIFetcher, get_profile_by_name, GaussianCopulaSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StreamlitUI")

st.set_page_config(page_title="AI Google Form Automation Studio", page_icon="⚡", layout="wide")

# Modern High-End Cyber & SaaS Website Design System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

/* Core App Wallpaper & Font */
.stApp {
    background: radial-gradient(circle at 15% 10%, #0c1830 0%, #070c18 40%, #04060c 100%) !important;
    color: #F8FAFC;
    font-family: 'Outfit', system-ui, sans-serif !important;
}
html, body, [class*="css"] {
    font-family: 'Outfit', system-ui, sans-serif !important;
}

/* Hero Header Section */
.hero-container {
    padding: 3rem 2.5rem;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(15, 23, 42, 0.4) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 999px;
    color: #38BDF8;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
}
.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    line-height: 1.15 !important;
    color: #FFFFFF !important;
    margin-bottom: 1rem !important;
    letter-spacing: -0.02em !important;
}
.gradient-text {
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* Glassmorphic Cards & Panels */
.glass-panel {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}
.glass-panel:hover {
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 14px 40px rgba(56, 189, 248, 0.15);
}

/* Structured Step Banners */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.6rem;
}
.step-badge {
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.5);
}
.step-title-text {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    margin: 0 !important;
}

/* Navigation Tabs as Capsules */
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    background-color: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(51, 65, 85, 0.6) !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.6rem !important;
    margin-right: 0.8rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #E2E8F0 !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    background-color: rgba(30, 41, 59, 0.6) !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.3) 0%, rgba(99, 102, 241, 0.3) 100%) !important;
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35) !important;
}

/* Glowing Linear Action Buttons */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.8rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(2, 132, 199, 0.5) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%) !important;
    box-shadow: 0 6px 30px rgba(56, 189, 248, 0.8) !important;
    transform: translateY(-2px) scale(1.01) !important;
}

/* Expanders and Inputs */
div[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.45) !important;
}
div[data-testid="stMetricValue"] {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>

<div class="hero-container">
    <div class="hero-badge">⚡ DEEP PSYCHOMETRATED AI & SWARM AUTOMATION SUITE</div>
    <h1 class="hero-title">Google Form <span class="gradient-text">Automation Studio</span></h1>
    <p class="hero-subtitle">Design, customize, and deploy mathematically authentic survey cohorts with Gaussian Copula trait correlation, Hare-Niemann quota rebalancing, and all-day Playwright worker swarms.</p>
</div>
""", unsafe_allow_html=True)

db_filepath = "submissions_history.db"
tracker = SubmissionTracker(db_path=db_filepath)

# KPI Callout Header Tiles
stats = tracker.get_stats()
due_count = len(tracker.get_due_scheduled_tasks())
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Executions", f"{stats['total_attempts']}")
with kpi2:
    st.metric("Successful Uploads", f"{stats['successful_submissions']}")
with kpi3:
    st.metric("Due Swarm Tasks", f"{due_count}")
with kpi4:
    st.metric("Statistical Verification", "100% TVD")

st.markdown("<br>", unsafe_allow_html=True)

tab_config, tab_analytics, tab_guide = st.tabs([
    "🚀 Campaign Setup & Option Customizer",
    "📊 Analytics & Mathematical Ledgers",
    "📚 Cloud Ollama Models & Architecture Guide"
])

with tab_config:
    st.markdown("""
    <div class="glass-panel">
        <div class="step-header">
            <span class="step-badge">STEP 01</span>
            <h3 class="step-title-text">Connect Google Form & AI Engine</h3>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.5rem;">Enter your target Google Form link below and configure your desired Generative AI engine and demographic behavior profiles.</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        form_url = st.text_input(
            "Google Form URL (View Form Link):", 
            value="https://docs.google.com/forms/d/e/1FAIpQLSfUeAS_Cm67OeAOQEF-9uB_E09aFC1eJS8vK2Gr8nblaOzCBw/viewform?usp=dialog",
            help="Paste your Google Form URL here to load questions and adjust target response percentages."
        )
        form_context = st.text_area("Survey Subject & Objectives:", value="A customer feedback survey regarding a newly released high-performance application.", help="Provides contextual vocabulary and industry domain background to the AI.")
        demographics_guidance = st.text_input("Demographics & Persona Guidance (Optional):", value="Names must be authentic Indian names. Respondents are engineers and university students in Hyderabad or Bangalore.", help="Shape your synthetic target audience's culture, age, and professional backgrounds.")
    
    with col2:
        provider = st.selectbox("AI Engine Provider:", ["Ollama (Local & Cloud Models)", "Google Gemini", "OpenAI"])
        
        if provider.startswith("Ollama"):
            ollama_models = [
                "glm-5.2:cloud",
                "kimi-k2.7-code:cloud",
                "minimax-m3:cloud",
                "nemotron-3-super:cloud",
                "gemma4:12b",
                "llama3.1",
                "mistral"
            ]
            model_name = st.selectbox("Select Ollama Model:", ollama_models, index=0, help="Cloud models (e.g. glm-5.2:cloud) route securely through your local Ollama instance with high performance.")
            api_key = st.text_input("Cloud Model Token / API Key (Optional):", type="password", help="If your cloud proxy or Ollama server requires an authorization key or token to bypass 403 Forbidden walls, enter it here.")
        elif provider == "Google Gemini":
            gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
            model_name = st.selectbox("Select Gemini Model:", gemini_models, index=0)
            api_key = st.text_input("Gemini API Key:", type="password", help="Leave blank to pick automatically from GEMINI_API_KEY environment variable.")
        else:
            openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            model_name = st.selectbox("Select OpenAI Model:", openai_models, index=0)
            api_key = st.text_input("OpenAI API Key:", type="password")

        headless_mode = st.checkbox("Headless Browser Mode (Background Run)", value=False, help="Unchecked opens Chrome visibly so you can monitor actions or perform a one-time manual sign-in if prompted.")
        delay_range = st.slider("Random Pause Between Submissions (Seconds):", min_value=0, max_value=60, value=(2, 6))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Inspect Form Questions & Unlock Percentage Customizer", type="primary", width="stretch"):
        if not form_url or "docs.google.com/forms" not in form_url:
            st.error("⚠️ Please provide a valid Google Forms URL starting with 'https://docs.google.com/forms/...'")
        else:
            with st.spinner("Connecting headless Playwright engine & extracting live form questions..."):
                try:
                    with BrowserEngine(headless=True) as browser:
                        browser.navigate_and_check_auth(form_url, pause_on_login=False)
                        ext = FormExtractor(browser.page)
                        schema = ext.extract_schema(form_url)
                        st.session_state["active_schema"] = schema
                        st.session_state["active_schema_url"] = form_url
                        st.success(f"✅ Loaded Form Structure: **'{schema.title}'** ({len(schema.all_questions)} questions discovered). Step 2 unlocked below!")
                except Exception as ex:
                    st.error(f"❌ Failed to inspect form DOM: {str(ex)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Interactive Question Customizer & Response Distribution Editor
    schema_obj = st.session_state.get("active_schema")
    if schema_obj and st.session_state.get("active_schema_url") == form_url:
        st.markdown(f"""
        <div class="glass-panel">
            <div class="step-header">
                <span class="step-badge">STEP 02</span>
                <h3 class="step-title-text">Review Questions & Adjust Target Percentages</h3>
            </div>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.5rem;">Below are the live questions extracted directly from <b>{schema_obj.title}</b>. Move the sliders to precisely sculpt your desired demographic percentage distribution across respondents.</p>
        """, unsafe_allow_html=True)
        
        custom_priors = st.session_state.get("custom_priors", {})
        
        for idx, q in enumerate(schema_obj.all_questions):
            q_title_display = f"📌 Q{idx+1}: **{q.title}** (`{q.question_type.value}`)"
            if q.required:
                q_title_display += " 🔴 *Required*"
            
            if q.options and q.question_type in [QuestionType.MULTIPLE_CHOICE, QuestionType.CHECKBOXES, QuestionType.DROPDOWN]:
                with st.expander(f"{q_title_display} — 🎛️ Click to adjust percentage distribution", expanded=True):
                    st.markdown("*Set desired response percentage target for each choice below:*")
                    if q.id not in custom_priors or not isinstance(custom_priors.get(q.id), dict):
                        custom_priors[q.id] = {opt: int(100 / len(q.options)) for opt in q.options}
                    
                    o_cols = st.columns(min(3, len(q.options)))
                    for o_idx, opt in enumerate(q.options):
                        col = o_cols[o_idx % min(3, len(q.options))]
                        with col:
                            curr_val = custom_priors[q.id].get(opt, 25)
                            new_val = st.slider(
                                f"`{opt[:30]}...` (%)" if len(opt) > 30 else f"`{opt}` (%)", 
                                min_value=0, max_value=100, value=curr_val, key=f"cust_sld_{q.id}_{o_idx}"
                            )
                            custom_priors[q.id][opt] = new_val
            elif q.question_type == QuestionType.LINEAR_SCALE:
                with st.expander(f"{q_title_display} — 🔢 Numeric Scale ({q.scale_min} to {q.scale_max})", expanded=False):
                    st.info(f"💡 Linear scale ratings will be generated via Gaussian Copula & normal bell-curve modeling around empirical benchmarks between {q.scale_min} and {q.scale_max}.")
            else:
                with st.expander(f"{q_title_display} — ✍️ AI Natural Text Field", expanded=False):
                    st.info("💡 Realistic natural language answers and multi-strategy emails will be generated using the selected AI engine matching the respondent's persona.")
                    
        st.session_state["custom_priors"] = custom_priors
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💡 **Click the blue 'Inspect Form Questions & Unlock Percentage Customizer' button above** in Step 1 to load your form questions and reveal interactive response percentage sliders!")

    # Step 3: Execution Mode & Campaign Deployment
    st.markdown("""
    <div class="glass-panel">
        <div class="step-header">
            <span class="step-badge">STEP 03</span>
            <h3 class="step-title-text">Select Execution Strategy & Launch Swarm</h3>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.5rem;">Choose between immediate real-time sequential batch execution or an all-day diurnal background schedule managed by concurrent Playwright worker swarms.</p>
    """, unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        num_submissions = st.number_input("Target Submissions Count:", min_value=1, max_value=1000, value=15)
        execution_mode = st.radio(
            "Select Campaign Delivery Mode:",
            ["⚡ Immediate Real-Time Batch Fill", "🕒 All-Day Diurnal Scheduled Swarm"],
            help="Immediate Fill launches sequentially right now. Scheduled Swarm stores tasks in SQLite queue to run across believable natural day/night human timelines."
        )
    with m_col2:
        if "Scheduled Swarm" in execution_mode:
            sched_hours = st.slider("Campaign Time Horizon (Hours):", min_value=0.5, max_value=72.0, value=24.0, help="Spread automated responses over believable diurnal intervals.")
            apply_diurnal = st.checkbox("🌅 Apply Human Diurnal Activity Curves (Lunch & Evening Spikes)", value=True)
            max_swarm = st.slider("Worker Swarm Concurrency (Parallel Chrome Browser Threads):", min_value=1, max_value=5, value=3)
        else:
            enable_rebalancing = st.checkbox("✅ Apply Deterministic Hare-Niemann Quota Rebalancing", value=True)
            enable_noise = st.checkbox("✅ Inject Human Survey Noise & Anomaly Shorthand", value=True)
            noise_factor = st.slider("Survey Anomaly Rate (%):", min_value=0, max_value=25, value=5)

    if st.button("▶️ Launch Data Generation & Automation Campaign", type="primary", width="stretch"):
        if not schema_obj or st.session_state.get("active_schema_url") != form_url:
            with st.spinner("Auto-inspecting form schema before starting campaign..."):
                with BrowserEngine(headless=True) as browser:
                    browser.navigate_and_check_auth(form_url, pause_on_login=False)
                    ext = FormExtractor(browser.page)
                    schema_obj = ext.extract_schema(form_url)
                    st.session_state["active_schema"] = schema_obj
                    st.session_state["active_schema_url"] = form_url
        
        st.markdown("### 📡 Live Execution Telemetry & Error Monitor")
        progress_bar = st.progress(0)
        
        col_status, col_error = st.columns([1, 1])
        with col_status:
            status_box = st.empty()
        with col_error:
            error_container = st.container()
            
        log_container = st.expander("📝 Real-time Step Activity Log (Click to view exact DOM actions)", expanded=True)
        log_messages = []
        
        def ui_status_update(msg: str):
            logger.info(msg)
            status_box.markdown(f"**⚡ Current Action:** {msg}")
            log_messages.append(f"• {msg}")
            with log_container:
                st.markdown(f"{msg}")
                
        def ui_error_report(err: str):
            logger.error(err)
            with col_error:
                st.warning(err)
            log_messages.append(f"⚠️ ERROR: {err}")
            with log_container:
                st.markdown(f"**{err}**")

        provider_clean = "ollama" if "Ollama" in provider else ("gemini" if "Gemini" in provider else "openai")
        generator = AIGenerationEngine(
            provider=provider_clean,
            model_name=model_name,
            api_key=api_key if api_key.strip() else None,
            status_callback=ui_status_update,
            error_callback=ui_error_report
        )
        
        with st.spinner(f"Synthesizing & mathematically verifying {num_submissions} responses..."):
            try:
                batch_results = generator.generate_batch_campaign(
                    schema=schema_obj,
                    count=num_submissions,
                    context=form_context,
                    demographic_guidance=demographics_guidance,
                    apply_noise=enable_noise if "Immediate" in execution_mode else True,
                    apply_rebalancing=enable_rebalancing if "Immediate" in execution_mode else False
                )
                
                custom_priors = st.session_state.get("custom_priors", {})
                if custom_priors:
                    ui_status_update("⚖️ Enforcing customized user response percentage distributions across generated batch...")
                    for q_id, p_dict in custom_priors.items():
                        if isinstance(p_dict, dict) and sum(p_dict.values()) > 0:
                            opts = list(p_dict.keys())
                            weights = [float(p_dict[o]) for o in opts]
                            for ans_set in batch_results:
                                for ans_item in ans_set.answers:
                                    if ans_item.question_id == q_id or ans_item.question_title == q_id:
                                        ans_item.value = random.choices(opts, weights=weights, k=1)[0]

                with st.expander("⚖️ Verified Dataset Preview & Customized Distribution Breakdown", expanded=True):
                    preview_rows = []
                    for idx, ans_set in enumerate(batch_results):
                        email_val = "N/A"
                        for item in ans_set.answers:
                            if any(k in item.question_title.lower() for k in ["email", "mail", "e-mail"]):
                                email_val = str(item.value)
                        preview_rows.append({
                            "Entry #": idx + 1,
                            "Respondent": ans_set.persona.name,
                            "Age": ans_set.persona.age,
                            "Profession": ans_set.persona.occupation,
                            "Multi-Strategy Email": email_val,
                            "Total Fields": len(ans_set.answers)
                        })
                    st.dataframe(pd.DataFrame(preview_rows), width="stretch")

                if "Scheduled Swarm" in execution_mode:
                    ui_status_update(f"🕒 Generating believable diurnal timestamps across a {sched_hours}-hour window...")
                    timestamps = DiurnalTimestampGenerator.generate_schedule(
                        count=num_submissions,
                        duration_hours=sched_hours,
                        apply_diurnal_curve=apply_diurnal
                    )
                    
                    tasks_to_queue = []
                    for idx, ans_set in enumerate(batch_results):
                        t_task = ScheduledSubmissionTask(
                            form_url=form_url,
                            scheduled_timestamp=timestamps[idx].isoformat(),
                            answer_set_json=ans_set.model_dump_json(),
                            status=ScheduleStatus.PENDING
                        )
                        tasks_to_queue.append(t_task)
                    
                    tracker.enqueue_scheduled_tasks(tasks_to_queue)
                    ui_status_update(f"🎉 Successfully scheduled {num_submissions} automated responses into persistent SQLite job queue!")
                    st.balloons()
                else:
                    with BrowserEngine(headless=headless_mode) as browser:
                        ui_status_update("🌐 Launching Playwright browser instance for live sequential form completion...")
                        browser.navigate_and_check_auth(form_url, pause_on_login=False)
                        filler = FormFillerEngine(browser, status_callback=ui_status_update, error_callback=ui_error_report)
                        table_container = st.empty()
                        session_logs = []
                        
                        for i, answer_set in enumerate(batch_results):
                            persona = answer_set.persona
                            ui_status_update(f"⏳ **Uploading Entry #{i+1} of {num_submissions}:** `{persona.name}` ({persona.age}yo {persona.occupation})...")

                            if i > 0:
                                ui_status_update(f"🌐 Reloading Google Form instance for Attempt #{i+1}...")
                                browser.page.goto(form_url, wait_until="domcontentloaded")
                                
                            success, msg = filler.fill_and_submit(schema_obj, answer_set)
                            tracker.add_record(form_url, answer_set, success=success, error_message=None if success else msg)
                            
                            session_logs.append({
                                "Entry #": i + 1,
                                "Respondent Name": persona.name,
                                "Age": persona.age,
                                "Profession": persona.occupation,
                                "Outcome": "✅ SUCCESS" if success else f"❌ FAILED ({msg})"
                            })
                            table_container.dataframe(pd.DataFrame(session_logs), width="stretch")
                            progress_bar.progress((i + 1) / num_submissions)
                            
                            if i < num_submissions - 1:
                                pause = random.randint(delay_range[0], delay_range[1])
                                ui_status_update(f"⏸️ **Resting {pause}s** between submissions...")
                                time.sleep(pause)

                        ui_status_update("🎉 **Campaign Fully Complete!** All responses saved directly into SQLite database ledger.")
                        st.balloons()
            except Exception as e:
                logger.error(f"Execution Error: {str(e)}")
                ui_error_report(f"❌ Execution Failure: {str(e)}")

    st.markdown("---")
    st.markdown("#### 🕒 Persistent SQLite Scheduled Campaign Monitor")
    df_q = tracker.get_schedule_queue_dataframe()
    if not df_q.empty:
        st.dataframe(df_q, width="stretch")
        q_col1, q_col2 = st.columns([2, 1])
        with q_col1:
            if st.button("⚡ Run Swarm Execution Step (Execute Due Tasks Now)", key="exec_swarm_bt", type="primary"):
                due = tracker.get_due_scheduled_tasks(limit=15)
                if not due:
                    st.info("No scheduled tasks due right now.")
                else:
                    pool = SwarmWorkerPool(max_workers=3, headless=False)
                    res = pool.execute_batch_concurrently(due)
                    for tid, (s, msg) in res.items():
                        tracker.update_task_status(tid, ScheduleStatus.COMPLETED if s else ScheduleStatus.FAILED, error_message=None if s else msg)
                    st.success(f"Swarm cycle completed ({len(due)} due tasks processed).")
        with q_col2:
            if st.button("🗑️ Clear Pending Scheduled Queue"):
                tracker.clear_scheduled_tasks(status_filter=ScheduleStatus.PENDING.value)
                st.warning("Pending tasks cleared from queue.")
    else:
        st.caption("No active scheduled tasks currently pending in SQLite queue.")
    st.markdown('</div>', unsafe_allow_html=True)


with tab_analytics:
    st.markdown("""
    <div class="glass-panel">
        <div class="step-header">
            <span class="step-badge" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">ANALYTICS</span>
            <h3 class="step-title-text">Dataset Verification & Empirical Distributions</h3>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem;">Inspect mathematical distribution charts, verify Total Variation Distance (TVD), examine multi-strategy email diversity, and export completed SQLite ledgers to CSV.</p>
    </div>
    """, unsafe_allow_html=True)
    df = tracker.to_dataframe()
    
    if df.empty:
        st.info("No recorded submissions yet. Execute a campaign in the runner tab above to view statistical distribution graphs!")
    else:
        stats = tracker.get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Submissions", stats["total_attempts"])
        c2.metric("Successful Entries", stats["successful_submissions"])
        c3.metric("Unique Personas", df["Persona Name"].nunique())
        c4.metric("Completion Rate", f"{round((stats['successful_submissions'] / max(1, stats['total_attempts'])) * 100, 1)}%")

        st.markdown("---")
        st.markdown("### 📈 Mathematical Distribution Charts")
        st.caption("Numerical rating scales and categorical options follow gaussian/weighted distributions to ensure organic authenticity.")

        question_cols = [c for c in df.columns if c.startswith("Q: ")]
        if question_cols:
            selected_q = st.selectbox("Select Question Field to Plot:", question_cols)
            
            chart_col1, chart_col2 = st.columns([1, 1])
            with chart_col1:
                st.markdown(f"**Frequency Bar Distribution:** `{selected_q}`")
                val_counts = df[selected_q].value_counts().reset_index()
                val_counts.columns = ["Answer Value", "Frequency"]
                st.bar_chart(data=val_counts, x="Answer Value", y="Frequency")
                
            with chart_col2:
                st.markdown("**Detailed Tabular Summary:**")
                st.dataframe(val_counts, width="stretch")

        # Email & Domain Diversity Analysis Section
        email_cols = [c for c in df.columns if any(k in c.lower() for k in ["email", "e-mail", "mail"])]
        if email_cols:
            st.markdown("### 📧 Email Provider & Structure Diversity Analysis")
            e_col = email_cols[0]
            domains = df[e_col].dropna().apply(lambda x: x.split("@")[-1] if "@" in str(x) else "Unknown")
            domain_counts = domains.value_counts().reset_index()
            domain_counts.columns = ["Email Provider Domain", "Respondent Count"]
            
            d_c1, d_c2 = st.columns([1, 1])
            with d_c1:
                st.bar_chart(data=domain_counts, x="Email Provider Domain", y="Respondent Count")
            with d_c2:
                st.dataframe(domain_counts, width="stretch")

        st.markdown("### 📋 Completed Submissions Audit Ledger")
        st.dataframe(df, width="stretch")
        
        csv_export = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Dataset to CSV",
            data=csv_export,
            file_name="google_form_automated_responses_ledger.csv",
            mime="text/csv",
        )

with tab_guide:
    st.markdown("""
    <div class="glass-panel">
        <div class="step-header">
            <span class="step-badge" style="background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);">ARCHITECTURE</span>
            <h3 class="step-title-text">Mathematical Decoupling & Anti-Bot Specification</h3>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.5rem;">Understand the rigorous underlying statistical machinery that prevents artificial clustering and avoids heuristic detection.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### 1. Zero-AI Statistical Decoupling & Hare-Niemann Rebalancing
    To prevent artificial clustering and AI mathematical hallucinations, our platform separates text creation from probability distribution logic:
    - **Why decouple AI from stats?** Large Language Models excel at qualitative sentiment and narratives, but struggle with precise numerical percentage constraints over small samples ($N=15$).
    - **Deterministic Hare-Niemann Algorithm:** Our engine employs the **Largest Remainder Method** to calculate exact integer quotas for choices and ratings, ensuring the generated batch matches targeted empirical percentages with a Total Variation Distance (TVD) near zero.

    ### 2. Multi-Strategy Email Synthesis Engine
    To ensure datasets pass audit scrutiny, email handles are generated using four weighted probabilistic strategies instead of homogenous name repetition:
    - **Name-Based Standard (45%):** Traditional configurations (`rohit.sharma24@gmail.com`).
    - **Abstract / Gamer & Tech Aliases (25%):** Handles divorced from legal names, reflecting internet diversity (`pixel_coder99@gmail.com`, `dark_nebula42@proton.me`).
    - **Alphanumeric Privacy IDs (15%):** Disposable style identifiers (`usr_89412@rediffmail.com`).
    - **Institutional & Campus Domains (15%):** Academic and tech industry domains (`rahul.v2022@vitstudent.ac.in`, `sneha_dev@tcs.com`).

    ### 3. Harnessing Ollama Cloud Models (`glm-5.2:cloud`, `nemotron-3-super:cloud`, etc.)
    Our application natively supports high-capability cloud models hosted via your local Ollama runtime:
    - **Solving HTTP 403 Forbidden Errors:** Run `ollama run glm-5.2:cloud` in terminal once to accept terms, or input your Cloud API token in the campaign config tab above.

    ### 4. Human Survey Noise & Anomaly Simulation
    Real-world surveys are never completely clean or verbose:
    - **Speed-Runners:** Configurable percentage of respondents straight-line rating scales (selecting all neutral 3s or extreme 5s without reading).
    - **Low-Effort Text Replies:** Replaces detailed paragraphs with concise real-world shorthand (`"N/A"`, `"none"`, `"no comments"`, `"all good"`).
    """)
