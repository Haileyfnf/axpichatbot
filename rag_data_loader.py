from settings import create_postgres_engine_by_sqlalchemy
from sqlalchemy import text
from rag_system import RAGSystem
import json

def load_hospital_data():
    try:
        # 데이터베이스 연결
        engine = create_postgres_engine_by_sqlalchemy()
        
        # 병원 데이터 예시
        hospitals = [
            {
                "name": "서울대학교병원",
                "category": "대학병원",
                "menu": "이비인후과, 내과, 외과, 소아과, 산부인과, 정형외과, 안과, 치과, 피부과, 신경과, 정신건강의학과, 비뇨기과, 재활의학과, 마취통증의학과, 영상의학과, 진단검사의학과, 응급의학과, 가정의학과",
                "price_avg": "의료보험 적용",
                "review_cnt": 1000,
                "rating": "4.8",
                "address": "서울특별시 종로구 대학로 101",
                "subway_info": "4호선 혜화역 2번 출구에서 도보 5분",
                "open_hour": "평일 08:30-17:30, 토요일 08:30-13:00",
                "break_time": "12:00-13:00",
                "closed_day": "일요일, 공휴일",
                "contact": "02-2072-0000",
                "facilities": "주차장, 휠체어, 엘리베이터, 장애인 화장실",
                "website": "http://www.snuh.org",
                "biz_type": "병원",
                "type": "병원"
            },
            {
                "name": "세브란스병원",
                "category": "대학병원",
                "menu": "이비인후과, 내과, 외과, 소아과, 산부인과, 정형외과, 안과, 치과, 피부과, 신경과, 정신건강의학과, 비뇨기과, 재활의학과, 마취통증의학과, 영상의학과, 진단검사의학과, 응급의학과, 가정의학과",
                "price_avg": "의료보험 적용",
                "review_cnt": 800,
                "rating": "4.7",
                "address": "서울특별시 서대문구 연세로 50-1",
                "subway_info": "2호선 신촌역 2번 출구에서 도보 10분",
                "open_hour": "평일 08:30-17:30, 토요일 08:30-13:00",
                "break_time": "12:00-13:00",
                "closed_day": "일요일, 공휴일",
                "contact": "02-2228-5800",
                "facilities": "주차장, 휠체어, 엘리베이터, 장애인 화장실",
                "website": "http://www.yuhs.or.kr",
                "biz_type": "병원",
                "type": "병원"
            }
        ]
        
        # 데이터베이스에 데이터 삽입
        with engine.connect() as connection:
            for hospital in hospitals:
                connection.execute(
                    text("""
                    INSERT INTO pi_study.axpi_hailey_dataset 
                    (name, category, menu, price_avg, review_cnt, rating, address, 
                     subway_info, open_hour, break_time, closed_day, contact, 
                     facilities, website, biz_type, type)
                    VALUES 
                    (:name, :category, :menu, :price_avg, :review_cnt, :rating, :address,
                     :subway_info, :open_hour, :break_time, :closed_day, :contact,
                     :facilities, :website, :biz_type, :type)
                    """),
                    hospital
                )
            connection.commit()
            print("병원 데이터가 성공적으로 로드되었습니다.")
            
    except Exception as e:
        print(f"병원 데이터 로드 중 오류가 발생했습니다: {str(e)}")

def load_existing_data_to_rag():
    try:
        # 데이터베이스 연결
        engine = create_postgres_engine_by_sqlalchemy()
        rag = RAGSystem()
        
        # 기존 테이블에서 데이터 조회
        with engine.connect() as connection:
            result = connection.execute(
                text("""
                SELECT 
                    id, name, category, menu, price_avg, 
                    review_cnt, page_url, rating, address, 
                    subway_info, open_hour, break_time, 
                    closed_day, contact, facilities, 
                    website, biz_type, type
                FROM pi_study.axpi_hailey_dataset
                """)
            )
            
            # 각 레코드를 RAG 시스템에 저장
            for row in result:
                # 데이터를 JSON 형식으로 변환
                content = f"""
                이름: {row.name}
                카테고리: {row.category}
                메뉴: {row.menu}
                평균 가격: {row.price_avg}
                리뷰 수: {row.review_cnt}
                평점: {row.rating}
                주소: {row.address}
                지하철 정보: {row.subway_info}
                영업 시간: {row.open_hour}
                휴식 시간: {row.break_time}
                휴무일: {row.closed_day}
                연락처: {row.contact}
                시설: {row.facilities}
                웹사이트: {row.website}
                업종: {row.biz_type}
                """
                
                # 메타데이터 생성
                metadata = {
                    "id": row.id,
                    "name": row.name,
                    "category": row.category,
                    "biz_type": row.biz_type,
                    "type": row.type,
                    "source": "axpi_hailey_dataset"
                }
                
                # RAG 시스템에 저장
                rag.store_document(content, metadata)
                print(f"데이터 저장 완료: {row.name}")
        
        print("모든 데이터가 RAG 시스템에 저장되었습니다.")
        
    except Exception as e:
        print(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    load_hospital_data()
    load_existing_data_to_rag() 