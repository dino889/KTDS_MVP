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
import math

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
        
    def search_relevant_tables(self, user_query: str, top_k: int = 10, use_vector_search: bool = True, 
                             exclude_tables: List[str] = None) -> List[dict]:
        """사용자 쿼리와 관련된 테이블 정보를 Azure AI Search에서 검색"""
        try:
            # 제외할 테이블이 있는 경우 더 많이 검색
            search_top_k = top_k * 2 if exclude_tables else top_k
            
            if use_vector_search:
                # 쿼리를 벡터로 변환
                query_vector = self.embeddings.embed_query(user_query)
                
                # 벡터 검색 수행
                vector_query = VectorizedQuery(
                    vector=query_vector, 
                    k_nearest_neighbors=search_top_k, 
                    fields="text_vector"
                )
                
                results = self.search_client.search(
                    search_text=None,  # 벡터 검색시에는 텍스트 검색 안함
                    vector_queries=[vector_query],
                    select=["OWNER", "TABLE_NAME", "COLUMNS", "chunk"],
                    top=search_top_k
                )
            else:
                # 일반 텍스트 검색 수행
                results = self.search_client.search(
                    search_text=user_query,
                    select=["OWNER", "TABLE_NAME", "COLUMNS", "chunk"],
                    top=search_top_k,
                    include_total_count=True
                )
            
            relevant_tables = []
            seen_tables = set()  # 중복 제거를 위한 집합
            exclude_set = set(exclude_tables) if exclude_tables else set()
            
            for result in results:
                # 테이블 식별자 생성
                table_id = f"{result.get('OWNER', '')}.{result.get('TABLE_NAME', '')}"
                
                # 중복 테이블이거나 제외 목록에 있는 경우 건너뛰기
                if table_id in seen_tables or table_id in exclude_set:
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
                
                # 원하는 개수만큼 찾았으면 중단
                if len(relevant_tables) >= top_k:
                    break
                
            logger.info(f"검색된 테이블 수: {len(relevant_tables)}")
            return relevant_tables
            
        except Exception as e:
            logger.error(f"테이블 검색 중 오류 발생: {str(e)}")
            if hasattr(e, 'error'):
                logger.error(f"Error details: {e.error}")
            return []
    
    def generate_sql_query(self, user_query: str, relevant_tables: List[dict], 
                          conversation_history: List[Dict] = None,
                          join_conditions: str = None,
                          previous_sql: str = None,
                          modification_request: str = None) -> SQLQuery:
        """사용자 쿼리와 관련 테이블 정보를 바탕으로 SQL 쿼리 생성"""
        
        # 대화 히스토리 포맷팅
        history_context = ""
        if conversation_history:
            history_context = "\n이전 대화 내용:\n"
            for msg in conversation_history[-5:]:  # 최근 5개만 사용
                if msg['role'] in ['user', 'assistant'] and 'content' in msg:
                    history_context += f"{msg['role']}: {msg['content']}\n"
        
        # 쿼리 수정 요청인 경우
        if previous_sql and modification_request:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 Oracle SQL 전문가입니다. 
                사용자가 이전에 생성된 SQL 쿼리를 수정해달라고 요청했습니다.
                
                다음 지침을 따라주세요:
                1. 사용자의 수정 요청을 정확히 이해하고 반영하세요.
                2. 기존 쿼리의 구조는 유지하되, 요청된 부분만 수정하세요.
                3. Oracle SQL 문법을 사용하세요.
                4. 모든 쿼리에 ROWNUM <= 10 조건을 유지하세요.
                5. 수정된 부분에 대해 명확히 설명하세요.
                
                {format_instructions}"""),
                ("human", """원래 요청: {original_query}
                
                이전 SQL 쿼리:
                {previous_sql}
                
                수정 요청: {modification_request}
                
                사용 가능한 테이블 정보:
                {tables_info}
                
                위 정보를 바탕으로 SQL 쿼리를 수정해주세요.""")
            ])
            
            # 체인 실행
            chain = prompt | self.llm | self.parser
            
            try:
                result = chain.invoke({
                    "original_query": user_query,
                    "previous_sql": previous_sql,
                    "modification_request": modification_request,
                    "tables_info": self._format_tables_info(relevant_tables),
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                return SQLQuery(**result)
                
            except Exception as e:
                logger.error(f"SQL 수정 중 오류 발생: {str(e)}")
                raise
        
        else:
            # 새로운 쿼리 생성
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 Oracle SQL 전문가입니다. 
                사용자의 자연어 요청을 분석하여 적절한 SQL 쿼리를 생성해주세요.
                
                제공된 테이블은 CDMOWN(고객 도메인)과 ORDOWN(계약 도메인)으로 구분됩니다.
                계약 관계, 상품 이력 등은 ORDOWN, 고객/청구 정보는 CDMOWN에서 찾을 수 있습니다.
                
                다음 지침을 따라주세요:
                1. 사용자의 의도를 명확하게 파악하고 정리하세요.
                2. 제공된 테이블 정보를 분석하여 필요한 테이블과 컬럼을 선택하세요.
                3. 여러 테이블을 사용할 경우, 테이블 간 JOIN이 필요한지 판단하세요.
                4. JOIN 조건이 제공되지 않았고 필요한 경우, requires_join_condition을 true로 설정하세요.
                5. Oracle SQL 문법을 사용하세요.
                6. 쿼리는 읽기 쉽게 포맷팅하세요.
                7. 날짜 조건은 TO_DATE 함수를 사용하세요.
                8. 이전 대화 내용을 참고하여 맥락을 이해하세요.
                9. 모든 쿼리에 ROWNUM <= 10 조건을 추가해주세요.
                10. 쿼리 설명 시 KORNET 관련 테이블은 인터넷으로 설명, TV 또는 IPTV는 IPTV로 설명해주세요.
                11. 상품명, 상품 ID(e.g. 0V1201)이 포함된 경우 상품 기본(PR_PROD_BAS) 테이블을 참조하세요.
                12. 어떤 컬럼을 조회해달라는 명시가 없으면 SELECT * 로 표시해주세요.
                13. 쿼리 생성 후, 사용자가 이해할 수 있도록 쿼리에 대한 설명을 작성해주세요. 
                
                
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

def toggle_table_selection(msg_idx, table_idx):
    """테이블 선택 상태 토글"""
    key = f"table_check_{msg_idx}_{table_idx}"
    if key in st.session_state:
        if st.session_state[key]:
            st.session_state.selected_table_indices[msg_idx].add(table_idx)
        else:
            st.session_state.selected_table_indices[msg_idx].discard(table_idx)

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
    if 'current_page' not in st.session_state:
        st.session_state.current_page = {}
    if 'skip_join_condition' not in st.session_state:
        st.session_state.skip_join_condition = False
    if 'session_to_delete' not in st.session_state:
        st.session_state.session_to_delete = None
    if 'exclude_tables' not in st.session_state:
        st.session_state.exclude_tables = []
    if 'cancel_search' not in st.session_state:
        st.session_state.cancel_search = False
    if 'trigger_re_search' not in st.session_state:
        st.session_state.trigger_re_search = False
    if 'tables_for_join' not in st.session_state:
        st.session_state.tables_for_join = []
    if 'selected_table_indices' not in st.session_state:
        st.session_state.selected_table_indices = {}
    if 'query_mode' not in st.session_state:
        st.session_state.query_mode = "테이블 검색"
    if 'show_columns_for' not in st.session_state:
        st.session_state.show_columns_for = None
    if 'show_columns_table' not in st.session_state:
        st.session_state.show_columns_table = None

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
        st.session_state.selected_table_indices = {}
        st.session_state.query_mode = "테이블 검색"
        
        # 마지막 생성된 SQL 찾기
        st.session_state.last_generated_sql = None
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "assistant" and "sql_query" in msg:
                st.session_state.last_generated_sql = msg["sql_query"]
                break

def display_chat_history():
    """채팅 히스토리 표시"""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "table_selection" in message and message.get("is_active", False):
                    # 테이블 선택 UI 표시
                    st.write(message["content"])
                    
                    # 페이징 처리
                    tables = message["tables"]
                    total_tables = len(tables)
                    tables_per_page = 10
                    total_pages = math.ceil(total_tables / tables_per_page)
                    
                    if idx not in st.session_state.current_page:
                        st.session_state.current_page[idx] = 0
                    
                    current_page = st.session_state.current_page[idx]
                    start_idx = current_page * tables_per_page
                    end_idx = min(start_idx + tables_per_page, total_tables)
                    
                    # 이 메시지에 대한 선택된 인덱스 초기화
                    if idx not in st.session_state.selected_table_indices:
                        st.session_state.selected_table_indices[idx] = set()
                    
                    # 테이블 선택 컨테이너 (전체 선택/해제 버튼 포함)
                    with st.container():
                        st.write("### 📋 사용할 테이블을 선택하세요:")
                        
                        # 전체 선택/해제 버튼 (너비 조정)
                        col1, col2, col3 = st.columns([1, 1, 6])
                        with col1:
                            if st.button("전체 선택", key=f"select_all_{idx}", use_container_width=True):
                                # 모든 테이블 인덱스 추가
                                for i in range(total_tables):
                                    st.session_state.selected_table_indices[idx].add(i)
                                st.rerun()
                        with col2:
                            if st.button("전체 해제", key=f"deselect_all_{idx}", use_container_width=True):
                                # 모든 테이블 인덱스 제거
                                st.session_state.selected_table_indices[idx].clear()
                                st.rerun()
                    
                    # 현재 페이지의 테이블 목록 (폼 외부에서 컬럼 보기 버튼 처리)
                    for i in range(start_idx, end_idx):
                        table = tables[i]
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            # 저장된 선택 상태 확인
                            is_selected = st.checkbox(
                                label="선택",
                                value=i in st.session_state.selected_table_indices[idx],
                                key=f"table_check_{idx}_{i}",
                                on_change=lambda idx=idx, i=i: toggle_table_selection(idx, i)
                            )
                        with col2:
                            st.write(f"**{table['owner']}.{table['table_name']}**")
                            st.caption(table.get('table_comment', ''))
                        with col3:
                            # 컬럼 보기 버튼 (폼 외부)
                            if st.button("📋 컬럼", key=f"view_cols_{idx}_{i}"):
                                st.session_state.show_columns_for = f"{idx}_{i}"
                                st.session_state.show_columns_table = table
                                st.rerun()
                    
                    # 테이블 선택 폼
                    with st.form(key=f"table_form_{idx}"):
                        # 페이지 네비게이션 (테이블 목록 하단)
                        if total_pages > 1:
                            nav_col1, nav_col2, nav_col3 = st.columns([2, 1, 2])
                            with nav_col1:
                                if current_page > 0:
                                    if st.form_submit_button("◀ 이전 페이지", use_container_width=True):
                                        st.session_state.current_page[idx] -= 1
                                        st.rerun()
                            with nav_col2:
                                st.write(f"페이지 {current_page + 1} / {total_pages}")
                            with nav_col3:
                                if current_page < total_pages - 1:
                                    if st.form_submit_button("다음 페이지 ▶", use_container_width=True):
                                        st.session_state.current_page[idx] += 1
                                        st.rerun()
                        
                        # 하단 버튼들
                        button_col1, button_col2, button_col3 = st.columns([1, 1, 2])
                        with button_col1:
                            cancel_submitted = st.form_submit_button("❌ 검색 취소", use_container_width=True)
                        with button_col2:
                            re_search_submitted = st.form_submit_button("🔄 재검색", use_container_width=True)
                        with button_col3:
                            submitted = st.form_submit_button("✅ 선택 완료 및 SQL 생성", type="primary", use_container_width=True)
                        
                        if cancel_submitted:
                            # 검색 취소
                            message["is_active"] = False
                            st.session_state.cancel_search = True
                            st.rerun()
                        
                        if re_search_submitted:
                            # 재검색 트리거
                            st.session_state.trigger_re_search = True
                            # 현재 검색된 테이블들을 제외 목록에 추가
                            current_table_ids = [t['table_id'] for t in tables]
                            st.session_state.exclude_tables.extend(current_table_ids)
                            st.rerun()
                        
                        if submitted:
                            # 폼 제출 시 체크박스 상태는 이미 on_change로 처리됨
                            # 선택된 테이블만 추출
                            selected_indices = list(st.session_state.selected_table_indices[idx])
                            if selected_indices:
                                st.session_state.selected_tables = [tables[i] for i in selected_indices]
                            else:
                                st.error("최소 하나 이상의 테이블을 선택해주세요.")
                                st.stop()
                            
                            # 현재 메시지를 비활성화
                            message["is_active"] = False
                            
                            # SQL 생성을 위한 플래그 설정
                            st.session_state.trigger_sql_generation = True
                            # SQL 생성 프로세스 시작
                            st.rerun()
                    
                    # 검색 결과 개선 안내
                    st.info("💡 원하는 결과가 나오지 않은 경우 좌측 '검색할 테이블 수'를 늘려서 검색해보세요.")
                    
                elif "table_selection" in message and not message.get("is_active", False):
                    # 비활성화된 테이블 선택 UI는 텍스트만 표시
                    st.write(message["content"])
                    if not st.session_state.cancel_search:
                        st.info("✅ 테이블 선택 완료")
                    else:
                        st.info("❌ 검색 취소됨")
                        st.session_state.cancel_search = False
                elif "sql_query" in message:
                    st.write(message["content"])
                    st.code(message["sql_query"], language="sql")
                    # 마지막 생성된 SQL 업데이트
                    st.session_state.last_generated_sql = message["sql_query"]
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
    
    # 삭제 확인 모달 (메인 화면에 표시)
    if st.session_state.session_to_delete:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.error("⚠️ 채팅 삭제 확인")
                st.warning("이 채팅을 삭제하시겠습니까? 삭제 후에는 되돌릴 수 없습니다.")
                
                del_col1, del_col2 = st.columns(2)
                with del_col1:
                    if st.button("삭제", type="primary", use_container_width=True):
                        # 세션 삭제
                        del st.session_state.chat_sessions[st.session_state.session_to_delete]
                        st.session_state.session_to_delete = None
                        st.rerun()
                with del_col2:
                    if st.button("취소", use_container_width=True):
                        st.session_state.session_to_delete = None
                        st.rerun()
        st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        # 1. 사용 가능한 도메인 (최상단)
        st.header("📚 사용 가능한 도메인")
        with st.expander("도메인 정보", expanded=False):
            st.markdown("**계약 정보 (ORDOWN)**")
            st.caption("- 계약 관련 테이블")
            st.caption("- 상품, 가입 정보 등")
            st.markdown("**고객 정보 (CDMOWN)**")
            st.caption("- 고객 관련 테이블")
            st.caption("- 고객 정보, 연락처 등")
        
        st.markdown("---")
        
        # 2. 설정
        st.header("⚙️ 설정")
        
        # 검색할 테이블 수 입력
        col1, col2 = st.columns([2, 1])
        with col1:
            top_k = st.number_input(
                "검색할 테이블 수", 
                min_value=1, 
                max_value=100, 
                value=10, 
                step=5,
                help="1~100 사이의 값을 입력하세요"
            )
        with col2:
            st.write("")  # 빈 공간
            st.write("")  # 정렬을 위한 빈 공간
            if st.button("기본값", use_container_width=True):
                st.session_state['reset_top_k'] = True
                st.rerun()
        
        # 기본값 리셋 처리
        if 'reset_top_k' in st.session_state and st.session_state.reset_top_k:
            top_k = 10
            del st.session_state.reset_top_k
        
        # 검색 방식 선택 (라디오 버튼)
        search_method = st.radio(
            "검색 방식",
            ["벡터 검색", "텍스트 검색"],
            index=0,  # 기본값: 벡터 검색
            help="벡터 검색: 의미 기반 검색\n텍스트 검색: 키워드 기반 검색"
        )
        
        use_vector_search = (search_method == "벡터 검색")
        
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
            st.session_state.table_selection_state = None  # 초기화
            st.session_state.selected_table_indices = {}
            st.session_state.last_generated_sql = None
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
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"📝 {timestamp} - {session_data['title']}", 
                        key=f"session_{session_id}",
                        use_container_width=True
                    ):
                        save_current_session()
                        load_session(session_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}"):
                        st.session_state.session_to_delete = session_id
                        st.rerun()
        else:
            st.caption("저장된 대화가 없습니다.")

    # 메인 채팅 영역과 컬럼 정보 영역을 나란히 배치
    if st.session_state.show_columns_for is not None:
        # 컬럼 정보가 표시될 때는 2열 레이아웃
        main_col, info_col = st.columns([3, 1])
        
        # 우측 컬럼 정보 영역
        with info_col:
            with st.container():
                # 제목과 닫기 버튼을 한 줄에 배치
                title_col, close_col = st.columns([3, 1])
                with title_col:
                    st.subheader("📋 컬럼 정보")
                with close_col:
                    st.write("")  # 높이 맞춤용
                    if st.button("✖", help="닫기"):
                        st.session_state.show_columns_for = None
                        st.session_state.show_columns_table = None
                        st.rerun()
                
                table = st.session_state.show_columns_table
                st.write(f"**{table['owner']}.{table['table_name']}**")
                st.caption(f"{table.get('table_comment', 'N/A')}")
                
                columns = table.get('columns', [])
                if columns:
                    st.write(f"**총 {len(columns)}개 컬럼**")
                    
                    # 스크롤 가능한 컨테이너로 컬럼 목록 표시
                    with st.container(height=600):
                        for i, col in enumerate(columns):
                            if isinstance(col, dict):
                                col_name = col.get('COLUMN_NAME', '')
                                col_type = col.get('DATA_TYPE', '')
                                col_comment = col.get('COLUMN_COMMENTS', '')
                                
                                st.markdown(f"**`{col_name}`**")
                                st.caption(f"{col_type}")
                                if col_comment:
                                    st.info(col_comment)
                                if i < len(columns) - 1:
                                    st.divider()
                else:
                    st.warning("컬럼 정보가 없습니다.")
    else:
        # 컬럼 정보가 없을 때는 전체 너비 사용
        main_col = st.container()
        info_col = None
    
    # 메인 채팅 영역
    with main_col if st.session_state.show_columns_for is not None else st.container():
        chat_container = st.container()
        
        with chat_container:
            # 채팅 히스토리 표시
            display_chat_history()
        
        # 재검색 처리
        if hasattr(st.session_state, 'trigger_re_search') and st.session_state.trigger_re_search:
            with st.chat_message("assistant"):
                with st.spinner("다른 테이블을 검색하는 중..."):
                    # 설정값 가져오기
                    settings = st.session_state.get('search_settings', {
                        'top_k': 10,
                        'use_vector_search': True
                    })
                    
                    # 제외 목록과 함께 재검색
                    relevant_tables = st.session_state.kos_system.search_relevant_tables(
                        st.session_state.pending_query,
                        top_k=settings['top_k'],
                        use_vector_search=settings['use_vector_search'],
                        exclude_tables=st.session_state.exclude_tables
                    )
                    
                    if not relevant_tables:
                        error_msg = "추가로 관련된 테이블을 찾을 수 없습니다. 검색할 테이블 수를 늘려보세요."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                    else:
                        # 재검색 결과 표시
                        response_text = f"🔄 추가로 {len(relevant_tables)}개의 관련 테이블을 찾았습니다.\n\n"
                        response_text += "SQL 생성에 사용할 테이블을 선택해주세요:"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "table_selection": True,
                            "tables": relevant_tables,
                            "is_active": True
                        })
                    
                    st.session_state.trigger_re_search = False
            st.rerun()
        
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
                            # 선택된 테이블들을 저장 (JOIN 조건 입력 후 사용)
                            st.session_state.tables_for_join = st.session_state.selected_tables
                            response_text = "🔗 여러 테이블을 조인해야 하는 쿼리입니다.\n\n"
                            response_text += f"**사용 테이블:**\n"
                            for table in result.tables:
                                response_text += f"- {table.owner}.{table.table_name}\n"
                            response_text += "\n테이블 간의 JOIN 조건을 입력해주세요. "
                            response_text += "(예: T1.CUST_ID = T2.CUST_ID)\n\n"
                            response_text += "⚠️ **JOIN 조건을 입력하지 않으면 첫 번째 테이블만 사용하여 쿼리를 생성합니다.**"
                            
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
                        st.session_state.selected_table_indices = {}
                        
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
            st.info("🔗 여러 테이블을 사용하는 쿼리입니다. JOIN 조건을 입력하거나 Enter를 눌러 첫 번째 테이블만 사용하세요.")
    
    # 사용자 입력 영역
    with st.container():
        # 입력 영역을 하나의 컨테이너로 묶어서 높이 정렬
        input_col1, input_col2 = st.columns([1, 10])
        
        with input_col1:
            # 컨테이너로 감싸서 높이 맞춤
            with st.container():
                # 빈 공간 추가로 높이 조정
                st.write("")  
                if st.session_state.query_mode == "테이블 검색":
                    if st.button("🔍", help="테이블 검색 모드", use_container_width=True):
                        st.session_state.query_mode = "쿼리 수정"
                        st.rerun()
                else:
                    if st.button("✏️", help="쿼리 수정 모드", use_container_width=True):
                        st.session_state.query_mode = "테이블 검색"
                        st.rerun()
        
        with input_col2:
            # 현재 모드 표시
            mode_text = "🔍 테이블 검색" if st.session_state.query_mode == "테이블 검색" else "✏️ 쿼리 수정"
            placeholder_text = "질문을 입력하세요..." if st.session_state.query_mode == "테이블 검색" else "수정사항을 입력하세요..."
            
            # 쿼리 수정 모드에서 이전 SQL이 없는 경우 안내
            if st.session_state.query_mode == "쿼리 수정" and not st.session_state.last_generated_sql:
                st.info(f"{mode_text} | 💡 먼저 테이블 검색을 통해 SQL을 생성해주세요.")
            else:
                st.caption(mode_text)
            
            # 사용자 입력 처리
            user_input = st.chat_input(placeholder_text)
            
            if user_input:
                # 사용자 메시지 추가
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                with chat_container:
                    # 사용자 메시지 표시
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    # 어시스턴트 응답
                    with st.chat_message("assistant"):
                        if st.session_state.waiting_for_join:
                            # JOIN 조건으로 처리 또는 첫 번째 테이블만 사용
                            if user_input.strip() == "":
                                # 빈 입력인 경우 첫 번째 테이블만 사용
                                with st.spinner("첫 번째 테이블만 사용하여 SQL을 생성하는 중..."):
                                    try:
                                        # JOIN을 위해 저장된 테이블들 중 첫 번째만 선택
                                        tables_for_query = st.session_state.get('tables_for_join', st.session_state.selected_tables)
                                        first_table_only = [tables_for_query[0]] if tables_for_query else []
                                        
                                        # SQL 생성
                                        result = st.session_state.kos_system.generate_sql_query(
                                            st.session_state.pending_query,
                                            first_table_only,
                                            conversation_history=st.session_state.messages[:-1]
                                        )
                                        
                                        # 응답 표시
                                        response_text = f"첫 번째 테이블만 사용하여 SQL을 생성했습니다.\n\n"
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
                                # JOIN 조건이 입력된 경우
                                with st.spinner("JOIN 조건을 적용하여 SQL을 재생성하는 중..."):
                                    try:
                                        # 이전 사용자 쿼리 찾기
                                        original_query = st.session_state.pending_query
                                        
                                        # SQL 재생성
                                        result = st.session_state.kos_system.generate_sql_query(
                                            original_query,
                                            st.session_state.get('tables_for_join', st.session_state.selected_tables),
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
                        
                        elif st.session_state.query_mode == "쿼리 수정":
                            # 쿼리 수정 모드
                            if st.session_state.last_generated_sql:
                                with st.spinner("SQL 쿼리를 수정하는 중..."):
                                    try:
                                        # 원래 쿼리 찾기
                                        original_query = st.session_state.pending_query
                                        
                                        # 이전에 사용된 테이블 정보 찾기
                                        tables_for_modification = st.session_state.selected_tables
                                        if not tables_for_modification:
                                            # 테이블 정보가 없으면 마지막 검색된 테이블 사용
                                            tables_for_modification = st.session_state.searched_tables
                                        
                                        # SQL 수정 요청
                                        result = st.session_state.kos_system.generate_sql_query(
                                            original_query,
                                            tables_for_modification,
                                            conversation_history=st.session_state.messages[:-1],
                                            previous_sql=st.session_state.last_generated_sql,
                                            modification_request=user_input
                                        )
                                        
                                        # 응답 표시
                                        response_text = f"SQL 쿼리를 수정했습니다.\n\n"
                                        response_text += f"**수정 내용:** {user_input}\n\n"
                                        response_text += f"**쿼리 설명:** {result.explanation}"
                                        
                                        st.write(response_text)
                                        st.code(result.sql_query, language="sql")
                                        
                                        # 메시지 저장
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": response_text,
                                            "sql_query": result.sql_query
                                        })
                                        
                                    except Exception as e:
                                        error_msg = f"SQL 수정 중 오류가 발생했습니다: {str(e)}"
                                        st.error(error_msg)
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": error_msg
                                        })
                            else:
                                st.error("수정할 SQL 쿼리가 없습니다. 먼저 테이블 검색을 통해 SQL을 생성해주세요.")
                        
                        else:
                            # 일반 쿼리 처리 (테이블 검색 모드)
                            # 테이블 선택 상태 초기화 (새 질문마다 초기화)
                            st.session_state.table_selection_state = None
                            st.session_state.exclude_tables = []  # 제외 목록 초기화
                            st.session_state.selected_table_indices = {}
                            
                            with st.spinner("관련 테이블을 검색하는 중..."):
                                # 설정값 가져오기
                                settings = st.session_state.get('search_settings', {
                                    'top_k': 10,
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