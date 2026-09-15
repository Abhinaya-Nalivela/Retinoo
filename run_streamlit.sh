#!/usr/bin/env bash
# Kill any existing Streamlit processes
pkill -f "streamlit run" || true
# Start Streamlit on default port 8502 and log output
streamlit run app.py --server.port 8502 > streamlit.log 2>&1 &
echo "Streamlit started on port 8502. Log: $(pwd)/streamlit.log"
