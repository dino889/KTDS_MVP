#!/bin/bash

# Azure Web App 환경에서 실행되는 스크립트

# pip 업그레이드
python -m pip install --upgrade pip

# 의존성 설치 (이미 빌드 단계에서 설치되었을 수 있음)
if [ ! -d "/home/site/wwwroot/.venv" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Streamlit 앱 실행
# Azure Web App은 PORT 환경 변수를 제공합니다
PORT=${PORT:-8000}
python -m streamlit run kos_sql_generator.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.serverAddress 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false