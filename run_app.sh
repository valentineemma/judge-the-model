#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"

if [[ -x .venv/Scripts/python.exe ]]; then
    python_cmd=".venv/Scripts/python.exe"
elif [[ -x .venv/bin/python ]]; then
    python_cmd=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
else
    echo "Python is missing. Set up the project's .venv first."
    exit 1
fi

if ! "$python_cmd" -m streamlit --version >/dev/null 2>&1; then
    echo "Streamlit is missing. Run uv pip install -r requirements.txt first."
    exit 1
fi

echo "Open http://localhost:8501 in your browser. Press Ctrl+C to stop."
exec "$python_cmd" -m streamlit run streamlit_app.py --server.address localhost --server.port 8501
