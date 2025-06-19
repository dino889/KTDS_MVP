import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import logging
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic 모델 정의 (Structured Output용)
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
    user_intent: str = Field(description="사용자의 의도를 명확하게 정리한 설명")
    tables: List[TableInfo] = Field(description="쿼리에 필요한 테이블 정보")
    sql_query: str = Field(description="생성된 SQL 쿼리")
    explanation: str = Field(description="쿼리에 대한 설명")
    requires_join_condition: bool = Field(description="JOIN 조건이 필요한지 여부", default=False)

class KOSOrderSystem:
    def __init__(self):
        # Azure OpenAI 설정
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0
        )
        
        # Azure OpenAI Embeddings 설정 (벡터 검색용)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "dev-text-embedding-small"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        
        # Azure AI Search 설정
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("INDEX_NAME", "ktds-mvp-index"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        
        # JSON Output Parser 설정
        self.parser = JsonOutputParser(pydantic_object=SQLQuery)
        
    def search_relevant_tables(self, user_query: str, top_k: int = 5, use_vector_search: bool = True) -> List[dict]:
        """사용자 쿼리와 관련된 테이블 정보를 Azure AI Search에서 검색"""
        try:
            if use_vector_search:
                # 쿼리를 벡터로 변환
                query_vector = self.embeddings.embed_query(user_query)
                
                # 벡터 검색 수행
                vector_query = VectorizedQuery(
                    vector=query_vector, 
                    k_nearest_neighbors=top_k, 
                    fields="text_vector"
                )
                
                results = self.search_client.search(
                    search_text=None,  # 벡터 검색시에는 텍스트 검색 안함
                    vector_queries=[vector_query],
                    select=["OWNER", "TABLE_NAME", "COLUMNS", "chunk"],
                    top=top_k
                )
            else:
                # 일반 텍스트 검색 수행
                results = self.search_client.search(
                    search_text=user_query,
                    select=["OWNER", "TABLE_NAME", "COLUMNS", "chunk"],
                    top=top_k,
                    include_total_count=True
                )
            
            relevant_tables = []
            seen_tables = set()  # 중복 제거를 위한 집합
            
            for result in results:
                # 테이블 식별자 생성
                table_id = f"{result.get('OWNER', '')}.{result.get('TABLE_NAME', '')}"
                
                # 중복 테이블 건너뛰기
                if table_id in seen_tables:
                    continue
                seen_tables.add(table_id)
                
                # chunk 필드에서 테이블 설명 추출
                table_comment = ""
                chunk = result.get("chunk", "")
                if chunk and isinstance(chunk, str):
                    # chunk에서 테이블 설명 추출 시도
                    table_comment = chunk.split('\n')[0] if chunk else ""
                
                # COLUMNS가 리스트인지 확인
                columns = result.get("COLUMNS", [])
                if not isinstance(columns, list):
                    columns = []
                
                # JSON 구조에 맞게 데이터 매핑
                table_info = {
                    "owner": result.get("OWNER", ""),
                    "table_name": result.get("TABLE_NAME", ""),
                    "table_comment": table_comment,
                    "columns": columns,
                    "table_id": table_id  # 고유 식별자 추가
                }
                relevant_tables.append(table_info)
                
            logger.info(f"검색된 테이블 수: {len(relevant_tables)}")
            return relevant_tables
            
        except Exception as e:
            logger.error(f"테이블 검색 중 오류 발생: {str(e)}")
            if hasattr(e, 'error'):
                logger.error(f"Error details: {e.error}")
            return []
    
    def generate_sql_query(self, user_query: str, relevant_tables: List[dict], 
                          conversation_history: List[Dict] = None,
                          join_conditions: str = None) -> SQLQuery:
        """사용자 쿼리와 관련 테이블 정보를 바탕으로 SQL 쿼리 생성"""
        
        # 대화 히스토리 포맷팅
        history_context = ""
        if conversation_history:
            history_context = "\n이전 대화 내용:\n"
            for msg in conversation_history[-5:]:  # 최근 5개만 사용
                if msg['role'] in ['user', 'assistant'] and 'content' in msg:
                    history_context += f"{msg['role']}: {msg['content']}\n"
        
        # 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 Oracle SQL 전문가입니다. 
            사용자의 자연어 요청을 분석하여 적절한 SQL 쿼리를 생성해주세요.
            
            제공된 테이블은 CDMOWN(고객 도메인)과 ORDOWN(계약 도메인)으로 구분됩니다.
            
            다음 지침을 따라주세요:
            1. 사용자의 의도를 명확하게 파악하고 정리하세요.
            2. 제공된 테이블 정보를 분석하여 필요한 테이블과 컬럼을 선택하세요.
            3. 여러 테이블을 사용할 경우, 테이블 간 JOIN이 필요한지 판단하세요.
            4. JOIN 조건이 제공되지 않았고 필요한 경우, requires_join_condition을 true로 설정하세요.
            5. Oracle SQL 문법을 사용하세요.
            6. 쿼리는 읽기 쉽게 포맷팅하세요.
            7. 날짜 조건은 TO_DATE 함수를 사용하세요.
            8. 이전 대화 내용을 참고하여 맥락을 이해하세요.
            
            {format_instructions}"""),
            ("human", """사용자 요청: {user_query}
            
            {history_context}
            
            사용 가능한 테이블 정보:
            {tables_info}
            
            {join_context}
            
            위 정보를 바탕으로 SQL 쿼리를 생성해주세요.""")
        ])
        
        # 테이블 정보 포맷팅
        tables_info_str = self._format_tables_info(relevant_tables)
        
        # JOIN 조건 컨텍스트
        join_context = ""
        if join_conditions:
            join_context = f"사용자 제공 JOIN 조건: {join_conditions}"
        
        # 체인 실행
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "user_query": user_query,
                "history_context": history_context,
                "tables_info": tables_info_str,
                "join_context": join_context,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return SQLQuery(**result)
            
        except Exception as e:
            logger.error(f"SQL 생성 중 오류 발생: {str(e)}")
            raise
    
    def _format_tables_info(self, tables: List[dict]) -> str:
        """테이블 정보를 읽기 쉬운 형식으로 포맷팅"""
        formatted = []
        for table in tables:
            table_str = f"소유자: {table.get('owner', 'N/A')}\n"
            table_str += f"테이블명: {table.get('table_name', 'N/A')}\n"
            table_str += f"설명: {table.get('table_comment', 'N/A')}\n"
            table_str += "컬럼:\n"
            
            for col in table.get('columns', []):
                if isinstance(col, dict):
                    col_name = col.get('COLUMN_NAME', '')
                    col_type = col.get('DATA_TYPE', '')
                    col_comment = col.get('COLUMN_COMMENTS', '')
                    table_str += f"  - {col_name}: {col_type} ({col_comment})\n"
            
            formatted.append(table_str)
        
        return "\n\n".join(formatted)

def init_session_state():
    """세션 상태 초기화"""
    if 'kos_system' not in st.session_state:
        st.session_state.kos_system = KOSOrderSystem()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'waiting_for_table_selection' not in st.session_state:
        st.session_state.waiting_for_table_selection = False
    if 'waiting_for_join' not in st.session_state:
        st.session_state.waiting_for_join = False
    if 'searched_tables' not in st.session_state:
        st.session_state.searched_tables = []
    if 'selected_tables' not in st.session_state:
        st.session_state.selected_tables = []
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = ""
    if 'trigger_sql_generation' not in st.session_state:
        st.session_state.trigger_sql_generation = False
    if 'table_selection_state' not in st.session_state:
        st.session_state.table_selection_state = None

def save_current_session():
    """현재 세션을 저장"""
    if st.session_state.messages:
        session_title = st.session_state.messages[0]['content'][:30] + "..."
        st.session_state.chat_sessions[st.session_state.current_session_id] = {
            'title': session_title,
            'messages': st.session_state.messages.copy(),
            'timestamp': datetime.now()
        }

def load_session(session_id):
    """저장된 세션 불러오기"""
    if session_id in st.session_state.chat_sessions:
        st.session_state.messages = st.session_state.chat_sessions[session_id]['messages'].copy()
        st.session_state.current_session_id = session_id
        st.session_state.waiting_for_table_selection = False
        st.session_state.waiting_for_join = False
        st.session_state.searched_tables = []
        st.session_state.selected_tables = []

def display_chat_history():
    """채팅 히스토리 표시"""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "table_selection" in message and message.get("is_active", False):
                    # 테이블 선택 UI 표시
                    st.write(message["content"])
                    
                    # 테이블 선택 폼
                    with st.form(key=f"table_form_{idx}"):
                        st.write("### 📋 사용할 테이블을 선택하세요:")
                        
                        # 전체 선택/해제 버튼
                        col1, col2, col3 = st.columns([1, 1, 3])
                        with col1:
                            if st.form_submit_button("전체 선택", use_container_width=True):
                                st.session_state.table_selection_state = "all"
                        with col2:
                            if st.form_submit_button("전체 해제", use_container_width=True):
                                st.session_state.table_selection_state = "none"
                        
                        # 테이블 목록
                        selected_indices = []
                        for i, table in enumerate(message["tables"]):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                # 전체 선택/해제 상태 확인
                                default_value = True
                                if hasattr(st.session_state, 'table_selection_state'):
                                    if st.session_state.table_selection_state == "all":
                                        default_value = True
                                    elif st.session_state.table_selection_state == "none":
                                        default_value = False
                                
                                is_selected = st.checkbox(
                                    label="선택",
                                    value=default_value,
                                    key=f"table_check_{idx}_{i}"
                                )
                                if is_selected:
                                    selected_indices.append(i)
                            with col2:
                                st.write(f"**{table['owner']}.{table['table_name']}**")
                                st.caption(table.get('table_comment', ''))
                        
                        submitted = st.form_submit_button("✅ 선택 완료 및 SQL 생성", type="primary", use_container_width=True)
                        
                        if submitted:
                            # 선택된 테이블만 추출
                            st.session_state.selected_tables = [
                                message["tables"][i] for i in selected_indices
                            ] if selected_indices else message["tables"]
                            
                            # 현재 메시지를 비활성화
                            message["is_active"] = False
                            
                            # SQL 생성을 위한 플래그 설정
                            st.session_state.trigger_sql_generation = True
                            # SQL 생성 프로세스 시작
                            st.rerun()
                elif "table_selection" in message and not message.get("is_active", False):
                    # 비활성화된 테이블 선택 UI는 텍스트만 표시
                    st.write(message["content"])
                    st.info("✅ 테이블 선택 완료")
                            
                elif "sql_query" in message:
                    st.write(message["content"])
                    st.code(message["sql_query"], language="sql")
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("📋 복사", key=f"copy_{idx}"):
                            st.success("클립보드에 복사되었습니다!")
                else:
                    st.write(message["content"])
            else:
                st.write(message["content"])

def main():
    st.set_page_config(
        page_title="KOS-오더 시스템",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 KOS-오더 시스템")
    st.markdown("자연어로 데이터를 조회하세요. SQL 쿼리를 자동으로 생성해드립니다.")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 설정
    with st.sidebar:
        # 1. 사용 가능한 도메인 (최상단)
        st.header("📚 사용 가능한 도메인")
        with st.expander("도메인 정보", expanded=True):
            st.markdown("**계약 정보 (ORDOWN)**")
            st.caption("- 계약 관련 테이블")
            st.caption("- 상품, 가입 정보 등")
            st.markdown("**고객 정보 (CDMOWN)**")
            st.caption("- 고객 관련 테이블")
            st.caption("- 고객 정보, 연락처 등")
        
        st.markdown("---")
        
        # 2. 설정
        st.header("⚙️ 설정")
        top_k = st.slider("검색할 테이블 수", min_value=1, max_value=20, value=10)
        use_text_search = st.checkbox("텍스트 검색 사용", value=False)
        use_vector_search = not use_text_search  # 벡터 검색이 기본값
        
        if use_vector_search:
            st.info("✅ 벡터 검색: 의미 기반")
        else:
            st.info("📝 텍스트 검색: 키워드 기반")
        
        # 설정 값을 세션에 저장
        st.session_state.search_settings = {
            'top_k': top_k,
            'use_vector_search': use_vector_search
        }
        
        st.markdown("---")
        
        # 3. 저장된 채팅 내역
        st.header("💾 저장된 채팅 내역")
        
        # 새 대화 시작 버튼
        if st.button("🔄 새 대화 시작", use_container_width=True):
            save_current_session()
            st.session_state.messages = []
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.waiting_for_table_selection = False
            st.session_state.waiting_for_join = False
            st.session_state.searched_tables = []
            st.session_state.selected_tables = []
            st.rerun()
        
        # 저장된 세션 목록
        if st.session_state.chat_sessions:
            st.caption("이전 대화를 선택하세요:")
            for session_id, session_data in sorted(
                st.session_state.chat_sessions.items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            ):
                timestamp = session_data['timestamp'].strftime("%m/%d %H:%M")
                if st.button(
                    f"📝 {timestamp} - {session_data['title']}", 
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    save_current_session()
                    load_session(session_id)
                    st.rerun()
        else:
            st.caption("저장된 대화가 없습니다.")

    # 메인 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 채팅 히스토리 표시
        display_chat_history()
        
        # 대기 중인 작업 처리
        if hasattr(st.session_state, 'trigger_sql_generation') and st.session_state.trigger_sql_generation:
            # 테이블 선택 완료 후 SQL 생성
            with st.chat_message("assistant"):
                with st.spinner("선택한 테이블로 SQL을 생성하는 중..."):
                    try:
                        result = st.session_state.kos_system.generate_sql_query(
                            st.session_state.pending_query,
                            st.session_state.selected_tables,
                            conversation_history=st.session_state.messages
                        )
                        
                        # JOIN 조건이 필요한 경우
                        if result.requires_join_condition and len(st.session_state.selected_tables) > 1:
                            st.session_state.waiting_for_join = True
                            response_text = "🔗 여러 테이블을 조인해야 하는 쿼리입니다.\n\n"
                            response_text += f"**사용 테이블:**\n"
                            for table in result.tables:
                                response_text += f"- {table.owner}.{table.table_name}\n"
                            response_text += "\n테이블 간의 JOIN 조건을 입력해주세요. "
                            response_text += "(예: T1.CUST_ID = T2.CUST_ID)"
                            
                            st.write(response_text)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                        else:
                            # SQL 생성 완료
                            response_text = f"**사용자 의도:** {result.user_intent}\n\n"
                            response_text += f"**쿼리 설명:** {result.explanation}"
                            
                            st.write(response_text)
                            st.code(result.sql_query, language="sql")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sql_query": result.sql_query
                            })
                        
                        st.session_state.trigger_sql_generation = False
                        st.session_state.selected_tables = []
                        
                    except Exception as e:
                        error_msg = f"SQL 생성 중 오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.session_state.trigger_sql_generation = False
            
            st.rerun()
        
        # JOIN 조건 대기 중인 경우
        if st.session_state.waiting_for_join:
            st.info("🔗 여러 테이블을 사용하는 쿼리입니다. JOIN 조건을 입력해주세요.")
    
    # 사용자 입력 처리
    if user_input := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with chat_container:
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(user_input)
            
            # 어시스턴트 응답
            with st.chat_message("assistant"):
                if st.session_state.waiting_for_join:
                    # JOIN 조건으로 처리
                    with st.spinner("JOIN 조건을 적용하여 SQL을 재생성하는 중..."):
                        try:
                            # 이전 사용자 쿼리 찾기
                            original_query = st.session_state.pending_query
                            
                            # SQL 재생성
                            result = st.session_state.kos_system.generate_sql_query(
                                original_query,
                                st.session_state.selected_tables,
                                conversation_history=st.session_state.messages[:-1],
                                join_conditions=user_input
                            )
                            
                            # 응답 표시
                            response_text = f"JOIN 조건을 적용하여 SQL을 생성했습니다.\n\n"
                            response_text += f"**사용자 의도:** {result.user_intent}\n\n"
                            response_text += f"**쿼리 설명:** {result.explanation}"
                            
                            st.write(response_text)
                            st.code(result.sql_query, language="sql")
                            
                            # 메시지 저장
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sql_query": result.sql_query
                            })
                            
                            st.session_state.waiting_for_join = False
                            
                        except Exception as e:
                            error_msg = f"SQL 생성 중 오류가 발생했습니다: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                
                else:
                    # 일반 쿼리 처리
                    with st.spinner("관련 테이블을 검색하는 중..."):
                        # 설정값 가져오기
                        settings = st.session_state.get('search_settings', {
                            'top_k': 5,
                            'use_vector_search': True
                        })
                        
                        # 테이블 검색
                        relevant_tables = st.session_state.kos_system.search_relevant_tables(
                            user_input, 
                            top_k=settings['top_k'], 
                            use_vector_search=settings['use_vector_search']
                        )
                        
                        if not relevant_tables:
                            error_msg = "관련 테이블을 찾을 수 없습니다."
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                        else:
                            # 검색된 테이블 저장
                            st.session_state.searched_tables = relevant_tables
                            st.session_state.pending_query = user_input
                            st.session_state.waiting_for_table_selection = True
                            
                            # 테이블 선택 UI 메시지
                            response_text = f"✅ {len(relevant_tables)}개의 관련 테이블을 찾았습니다.\n\n"
                            response_text += "SQL 생성에 사용할 테이블을 선택해주세요:"
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "table_selection": True,
                                "tables": relevant_tables,
                                "is_active": True  # 활성 상태로 설정
                            })
        
        # 페이지 새로고침으로 UI 업데이트
        st.rerun()

if __name__ == "__main__":
    main()