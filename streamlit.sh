# 가상환경에서 나가기
deactivate

# 가상환경 삭제 (Windows)
rmdir /s venv

# 가상환경 재생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 없이 패키지 설치
# pip install --no-cache-dir -r requirements.txt

pip install streamlit==1.31.0
pip install pydantic==2.5.3
pip install langchain==0.2.11
pip install langchain-openai==0.1.17
pip install langchain-core==0.2.23
pip install azure-search-documents==11.4.0
pip install python-dotenv==1.0.0
## 실행과 동시에 수행, 설치 되어야하는 대상들 전부 다 작성 필요

python -m streamlit azure_connection_test.py
python -m streamlit run kos_sql_generator.py --server.port 8000 --server.address 0.0.0.0