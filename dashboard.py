"""Streamlit dashboard for Alpha Evolver."""

import queue
import threading
import time

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("streamlit is required. Run: pip install streamlit") from exc

from alpha_evolver import run


def worker(out, generations, pop, seed):
    def progress(event):
        out.put(event)

    try:
        summary = run(generations=generations, pop=pop, seed=seed,
                      mem_path="dashboard_memory.json",
                      history_path="dashboard_history.csv",
                      curve_path="dashboard_learning_curve.png",
                      progress_cb=progress)
        out.put({"done": True, "summary": summary})
    except Exception as exc:
        out.put({"error": str(exc)})


st.set_page_config(page_title="Alpha Evolver", layout="wide")
st.title("Alpha Evolver")

with st.sidebar:
    generations = st.slider("Generations", 4, 60, 20)
    pop = st.slider("Population", 12, 120, 56)
    seed = st.number_input("Seed", value=7, step=1)
    start = st.button("Run")

if "events" not in st.session_state:
    st.session_state.events = queue.Queue()
if "latest" not in st.session_state:
    st.session_state.latest = None
if "thread" not in st.session_state:
    st.session_state.thread = None

if start and (st.session_state.thread is None or not st.session_state.thread.is_alive()):
    st.session_state.events = queue.Queue()
    st.session_state.latest = None
    st.session_state.thread = threading.Thread(
        target=worker,
        args=(st.session_state.events, generations, pop, int(seed)),
        daemon=True,
    )
    st.session_state.thread.start()

while True:
    try:
        st.session_state.latest = st.session_state.events.get_nowait()
    except queue.Empty:
        break

latest = st.session_state.latest
if latest is None:
    st.info("Choose settings and run.")
else:
    if "error" in latest:
        st.error(latest["error"])
    else:
        cols = st.columns(3)
        cols[0].metric("Generation", latest.get("gen", "-"))
        cols[1].metric("Best OOS Sharpe", f"{latest.get('best', 0):.3f}")
        cols[2].metric("Median OOS Sharpe", f"{latest.get('median', 0):.3f}")

        hist = latest.get("history", [])
        if hist:
            st.line_chart({
                "best_oos": [h[1] for h in hist],
                "median_oos": [h[3] for h in hist],
            })

        st.subheader("Current best")
        st.code(latest.get("best_key", ""))
        st.json(latest.get("best_stats", {}))

        mem = latest.get("memory", {})
        st.subheader("Top strength")
        st.dataframe([
            {
                "strength": r.get("strength", 0.0),
                "novelty": r.get("novelty", 0.0),
                "oos": r.get("stats", {}).get("oos_sharpe", 0.0),
                "alpha": r.get("key", ""),
            }
            for r in mem.get("top_strength", [])
        ], use_container_width=True)

        st.subheader("Principles")
        st.json(mem.get("principles", []))

        st.subheader("Motifs")
        st.dataframe(mem.get("motifs", []), use_container_width=True)

        st.subheader("Strategy")
        st.write(mem.get("strategy", {}).get("text", ""))

if st.session_state.thread is not None and st.session_state.thread.is_alive():
    time.sleep(2)
    st.rerun()
