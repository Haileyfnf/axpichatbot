from settings import create_postgres_engine_by_sqlalchemy
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database():
    try:
        engine = create_postgres_engine_by_sqlalchemy()
        with engine.connect() as conn:
            # 테이블 존재 여부 확인
            check_table = text("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'pi_study' 
                    AND table_name = 'axpi_hailey_dataset'
                )
            """)
            table_exists = conn.execute(check_table).scalar()
            logger.info(f"테이블 존재 여부: {table_exists}")
            
            if table_exists:
                # 데이터 수 확인
                count_query = text("SELECT COUNT(*) FROM pi_study.axpi_hailey_dataset")
                count = conn.execute(count_query).scalar()
                logger.info(f"전체 데이터 수: {count}")
                
                # 시설 유형별 데이터 수 확인
                type_query = text("""
                    SELECT "type", COUNT(*) 
                    FROM pi_study.axpi_hailey_dataset 
                    GROUP BY "type"
                """)
                type_counts = conn.execute(type_query).fetchall()
                logger.info("시설 유형별 데이터 수:")
                for type_name, type_count in type_counts:
                    logger.info(f"- {type_name}: {type_count}")
                
                # 샘플 데이터 확인
                sample_query = text("""
                    SELECT "Name", "type", "address" 
                    FROM pi_study.axpi_hailey_dataset 
                    LIMIT 3
                """)
                samples = conn.execute(sample_query).fetchall()
                logger.info("샘플 데이터:")
                for sample in samples:
                    logger.info(f"- {sample.Name} ({sample.type}): {sample.address}")
            
    except Exception as e:
        logger.error(f"데이터베이스 확인 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    check_database() 