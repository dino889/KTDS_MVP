# ===================================
# Azure 연결 테스트
# ===================================

# test_azure_connection.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_search_connection():
    """AI Search 연결 테스트"""
    try:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        api_key = os.getenv("AZURE_SEARCH_KEY")
        index_name = os.getenv("INDEX_NAME")
        
        url = f"{endpoint}/indexes/{index_name}/docs/search?api-version=2023-11-01"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        body = {"search": "test", "top": 1}
        
        response = requests.post(url, headers=headers, json=body, timeout=10)
        
        if response.status_code == 200:
            print("✅ AI Search 연결 성공")
            return True
        else:
            print(f"❌ AI Search 연결 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ AI Search 연결 오류: {str(e)}")
        return False

def test_openai_connection():
    """OpenAI 연결 테스트"""
    try:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5
        }
        
        response = requests.post(url, headers=headers, json=body, timeout=30)
        
        if response.status_code == 200:
            print("✅ OpenAI 연결 성공")
            return True
        else:
            print(f"❌ OpenAI 연결 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI 연결 오류: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Azure 서비스 연결 테스트 ===")
    search_ok = test_search_connection()
    openai_ok = test_openai_connection()
    
    if search_ok and openai_ok:
        print("\n🎉 모든 연결이 정상입니다!")
    else:
        print("\n❌ 일부 연결에 문제가 있습니다.")