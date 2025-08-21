from settings import create_postgres_engine_by_sqlalchemy
from sqlalchemy import text

def create_rag_tables():
    try:
        # 데이터베이스 연결
        engine = create_postgres_engine_by_sqlalchemy()
        
        # SQL 실행
        with engine.connect() as connection:
            # pgvector 확장 설치
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # 문서 저장 테이블 생성
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS pi_study.axpi_docu_store (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """))
            
            # 쿼리 히스토리 테이블 생성
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS pi_study.axpi_qna_history (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """))
            
            connection.commit()
            print("RAG 테이블 생성이 완료되었습니다.")
            
    except Exception as e:
        print(f"테이블 생성 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    create_rag_tables() 