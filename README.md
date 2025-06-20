# [KOS SQL Generator System](https://jiwooo-web-001.azurewebsites.net/)

자연어로 SQL을 생성하고, Azure AI 기반 벡터 검색 및 LangChain을 활용한 인터랙션이 가능한 Streamlit 애플리케이션입니다. 이 애플리케이션은 Azure OpenAI, Azure Cognitive Search, LangChain을 기반으로 사용자의 의도를 분석하고, 자동으로 SQL 쿼리를 생성합니다.

## ✅ 주요 기능

* 자연어 입력 기반 SQL 쿼리 생성
* 벡터 기반 또는 키워드 기반 테이블 검색 (Azure Cognitive Search)
* LangChain 기반 LLM 인터랙션 및 JSON Output 자동 파싱
* 테이블 선택 및 JOIN 조건 입력 기능
* Streamlit 인터페이스 기반 사용자 경험

---

## 🔧 설치 방법

### 1. Python 환경 세팅

* Python 3.12.10 버전에서 동작합니다. (다른 버전에서는 동작이 보장되지 않습니다)

### 2. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 아래 환경 변수들을 설정해야 합니다. (실제 키 값은 입력하지 마세요)

```
AZURE_OPENAI_ENDPOINT="https://jiwooo-openai-001.openai.azure.com/"
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_DEPLOYMENT_NAME="dev-gpt-4o-mini"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="dev-text-embedding-3-small"
AZURE_SEARCH_ENDPOINT="https://jiwoo-ai-search-001.search.windows.net/"
AZURE_SEARCH_KEY=""
INDEX_NAME="ktds-mvp-index"
AZURE_STORAGE_CONNECTION_STRING="https://jiwoostorage001.blob.core.windows.net/ktds-mvp"
AZURE_STORAGE_CONTAINER_NAME="ktds-mvp"
```

---

## ▶️ 실행 가이드

### 1. Streamlit 앱 실행(로컬에서 실행 시)

```bash
streamlit run kos_sql_generator.py
```

### 2. 주요 인터페이스

* **질문 입력**: 자연어로 조회 요청 입력 (예: "오늘 발생한 SOIP공사명령송신 내역 시간대별로 조회하고 싶어.")
* **테이블 선택**: 검색된 테이블 중 사용할 테이블 선택
* **JOIN 조건 입력** (선택): 여러 테이블을 사용하는 경우 JOIN 조건 입력 가능
* **SQL 쿼리 확인**: 생성된 SQL 및 설명 확인 및 복사 가능

---

## 🚀 Azure Web App 배포 가이드

1. VSC Extension으로 Deply 또는 해당 레포지토리 push(환경변수는 아래와 같이 세팅)
![image](https://github.com/user-attachments/assets/18ef8ffc-66a0-40db-9cba-ca8ecf5201d4)
2. requirements.txt 기반 종속성 설치 자동 진행
3. `start.txt` 로 streamlit run 자동 실행
---

## 📂 파일 구조

```
├── .deployment              # Azure 배포 설정
├── README.md                # 프로젝트 설명서
├── azure_connection_test.py # Azure 연결 테스트 코드
├── kos_sql_generator.py     # 메인 Streamlit 애플리케이션
├── requirements.txt         # 종속 패키지 목록
├── runtime.txt              # Python 런타임 버전 지정 (Azure용)
└── startup.txt              # 앱 시작 명령어 (streamlit 실행)
```

---

## 📘 프로젝트 구조 및 작동 원리

### 1. 전체 시스템 흐름도

```
사용자 입력 (자연어)
        ↓
테이블 검색 (Azure AI Search)
        ↓
테이블 선택 UI
        ↓
SQL 생성 (Azure OpenAI + LangChain)
        ↓
결과 표시 (Streamlit)
```

### 2. 주요 라이브러리 설명

**Azure 관련 라이브러리**

* `AzureChatOpenAI`: Azure OpenAI의 Chat 모델 사용
* `AzureOpenAIEmbeddings`: 텍스트를 벡터로 변환
* `SearchClient`: Azure AI Search 클라이언트
* `VectorizedQuery`: 벡터 검색 쿼리 구성

**LangChain 관련**

* `ChatPromptTemplate`: 구조화된 프롬프트 생성
* `JsonOutputParser`: JSON 형식 출력 파싱

**Pydantic**

* `BaseModel`: 데이터 모델 및 검증
* `Field`: 필드 설명 및 제약조건

**Streamlit**

* `st.chat_message`, `st.chat_input`: 채팅 UI
* `st.session_state`: 상태 관리

### 3. 핵심 클래스: `KOSOrderSystem`

```python
class KOSOrderSystem:
    def __init__(self):
        self.llm = AzureChatOpenAI(...)
        self.embeddings = AzureOpenAIEmbeddings(...)
        self.search_client = SearchClient(...)
        self.parser = JsonOutputParser(pydantic_object=SQLQuery)
```

### 4. 구조화된 출력 - Pydantic 사용 이유

* LLM 출력을 예측 가능한 구조로 강제
* 타입 검증 가능
* LangChain과 자연스럽게 연동

```python
class ColumnInfo(BaseModel):
    column_name: str = Field(description="컬럼 이름")
    data_type: str = Field(description="데이터 타입")
    comment: str = Field(description="컬럼 설명")

class TableInfo(BaseModel):
    owner: str = Field(description="테이블 소유자")
    table_name: str = Field(description="테이블 이름")
    columns: List[ColumnInfo] = Field(description="사용할 컬럼 목록")
    reason: str = Field(description="이 테이블을 선택한 이유")

class SQLQuery(BaseModel):
    user_intent: str
    tables: List[TableInfo]
    sql_query: str
    explanation: str
    requires_join_condition: bool = False
```

### 5. LangChain 체인 구성

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "시스템 지시사항..."),
    ("human", "사용자 요청: {user_query}")
])

chain = prompt | self.llm | self.parser

result = chain.invoke({
    "user_query": user_query,
    "tables_info": tables_info_str,
    "format_instructions": self.parser.get_format_instructions()
})
```

### 6. 벡터 검색 프로세스

```python
def search_relevant_tables(self, user_query: str):
    query_vector = self.embeddings.embed_query(user_query)
    vector_query = VectorizedQuery(...)
    results = self.search_client.search(vector_queries=[vector_query])
```

### 7. Streamlit 세션 상태 관리

* `messages`: 전체 대화 내역
* `searched_tables`: 검색 결과
* `selected_tables`: 선택된 테이블
* `waiting_for_join`: JOIN 조건 필요 여부
* `last_generated_sql`: 가장 최근 쿼리

### 8. 기능 플로우

**쿼리 생성**

* 사용자 질문 입력 → 검색 → 선택 → SQL 생성

**쿼리 수정**

* 기존 쿼리 참조 → 수정 요청 → 재생성

### 9. 사용자 경험 향상 포인트

* 실패 시 명확한 에러 메시지
* 테이블 목록 페이징 처리
* 컬럼 정보 실시간 확인
* 재검색 및 검색 취소 기능
