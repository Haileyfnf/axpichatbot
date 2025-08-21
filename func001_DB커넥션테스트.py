import pandas as pd
from settings import create_postgres_engine_by_sqlalchemy

def test_sql_postgres():
    try:
        # 연결 엔진 생성
        engine = create_postgres_engine_by_sqlalchemy()
        print("데이터베이스 연결 성공")

        # 테이블 존재 여부 확인
        check_table_sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_schema = 'pi_study' 
                AND table_name = 'axpi_hailey_dataset'
            )
        """
        exists_df = pd.read_sql(check_table_sql, engine)
        if exists_df.iloc[0, 0]:
            print("테이블이 존재합니다.")
            
            # 데이터 샘플 확인
            sql_query = """
                SELECT *
                FROM pi_study.axpi_hailey_dataset
                LIMIT 5
            """
            df = pd.read_sql(sql_query, engine)
            print("\n데이터 샘플:")
            print(df)
        else:
            print("테이블이 존재하지 않습니다.")
            
            # pi_study 스키마의 테이블 목록 출력
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'pi_study'
            """
            tables_df = pd.read_sql(tables_query, engine)
            print("\npi_study 스키마의 테이블 목록:")
            print(tables_df)

    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == '__main__':
    test_sql_postgres()