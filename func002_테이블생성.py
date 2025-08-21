"""
커밋 꼭 해주세요!
"""
from settings import create_postgres_engine_by_sqlalchemy
from sqlalchemy import text

creat_table_sql = """
create table pi_toyproject.axpi_hailey_dataset (
    id INT primary key,
    name VARCHAR(200) not null,
    category VARCHAR(100),
    menu TEXT,
    price_avg VARCHAR(50),
    review_cnt INT,
    page_url VARCHAR(500),
    rating FLOAT,
    address TEXT,
    subway_info VARCHAR(100),
    open_hour VARCHAR(100),
    break_time VARCHAR(100),
    closed_day VARCHAR(50),
    contact VARCHAR(50),
    facilities TEXT,
    website VARCHAR(500),
    biz_type VARCHAR(50)
);
commit;
"""

def create_table():
    try:
        # 데이터베이스 연결
        engine = create_postgres_engine_by_sqlalchemy()
        
        # SQL 실행
        with engine.connect() as connection:
            connection.execute(text(creat_table_sql))
            connection.commit()  # 명시적으로 커밋
            print("테이블 생성이 완료되었습니다.")
            
    except Exception as e:
        print(f"테이블 생성 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    create_table()