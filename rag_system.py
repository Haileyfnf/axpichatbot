from settings import create_postgres_engine_by_sqlalchemy
from sqlalchemy import text
import faiss
import numpy as np
from openai import OpenAI
import os
import json
import logging
from typing import List, Dict
import time
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI 관련 로그 레벨 설정
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class RAGSystem:
    def __init__(self):
        try:
            self.engine = create_postgres_engine_by_sqlalchemy()
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # 테이블 구조 확인
            with self.engine.connect() as connection:
                # 테이블 존재 여부 확인
                check_table = text("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_schema = 'pi_study' 
                        AND table_name = 'axpi_hailey_dataset'
                    )
                """)
                table_exists = connection.execute(check_table).scalar()
                
                if not table_exists:
                    logger.error("테이블이 존재하지 않습니다: pi_study.axpi_hailey_dataset")
                    raise Exception("테이블이 존재하지 않습니다")
                
                # 컬럼 정보 확인
                columns_query = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'pi_study' 
                    AND table_name = 'axpi_hailey_dataset'
                    ORDER BY ordinal_position
                """)
                columns = connection.execute(columns_query).fetchall()
                
                # 데이터 샘플 확인
                sample_query = text("""
                    SELECT * FROM pi_study.axpi_hailey_dataset LIMIT 1
                """)
                sample = connection.execute(sample_query).fetchone()
                if sample:
                    logger.info("데이터 샘플:")
                    for col, val in zip(columns, sample):
                        logger.info(f"- {col.column_name}: {val}")
                else:
                    logger.warning("테이블에 데이터가 없습니다.")
            
            # FAISS 인덱스 초기화 (1536은 OpenAI 임베딩 차원)
            self.dimension = 1536
            
            # 타입별 인덱스와 데이터 저장
            self.indices = {}
            self.documents = {}
            self.metadata_list = {}
            
            # 지원하는 장소 타입들
            self.place_types = ['음식점', '병원', '약국', '카페']
            self.place_type_mapping = {
                '음식점': 'restaurant',
                '병원': 'hospital',
                '약국': 'pharmacy',
                '카페': 'cafe'
            }
            
            # 인덱스 저장 디렉토리 생성
            os.makedirs('faiss_indexes', exist_ok=True)
            
            # 저장된 인덱스가 있다면 로드, 없다면 새로 생성
            if all(os.path.exists(f'faiss_indexes/{self.place_type_mapping[place_type]}.index') 
                   for place_type in self.place_types):
                logger.info("저장된 인덱스를 불러옵니다.")
                self.load_saved_indexes()
            else:
                logger.info("새로운 인덱스를 생성합니다.")
                self.load_all_place_data()
            
            logger.info("RAG 시스템이 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 중 오류 발생: {str(e)}")
            raise

    def load_saved_indexes(self):
        """저장된 FAISS 인덱스를 로드합니다."""
        try:
            for place_type in self.place_types:
                # 영문 파일명 사용
                index_path = f'faiss_indexes/{self.place_type_mapping[place_type]}.index'
                self.indices[place_type] = faiss.read_index(index_path)
                
                metadata_path = f'faiss_indexes/{self.place_type_mapping[place_type]}_metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents[place_type] = data['documents']
                    self.metadata_list[place_type] = data['metadata_list']
                
                logger.info(f"{place_type} 인덱스와 메타데이터를 로드했습니다.")
            logger.info("인덱스 로드 완료")
        except Exception as e:
            logger.error(f"인덱스 로드 오류: {str(e)}")
            raise

    def save_indexes(self):
        """FAISS 인덱스를 파일로 저장합니다."""
        try:
            for place_type in self.place_types:
                # 영문 파일명 사용
                index_path = f'faiss_indexes/{self.place_type_mapping[place_type]}.index'
                faiss.write_index(self.indices[place_type], index_path)
                
                metadata_path = f'faiss_indexes/{self.place_type_mapping[place_type]}_metadata.json'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'documents': self.documents[place_type],
                        'metadata_list': self.metadata_list[place_type]
                    }, f, ensure_ascii=False)
                
                logger.info(f"{place_type} 인덱스와 메타데이터를 저장했습니다.")
            logger.info("인덱스 저장 완료")
        except Exception as e:
            logger.error(f"인덱스 저장 오류: {str(e)}")
            raise

    def load_all_place_data(self):
        """모든 타입의 장소 데이터를 로드하고 임베딩을 생성합니다."""
        try:
            for place_type in self.place_types:
                logger.info(f"\n=== {place_type} 데이터 로드 시작 ===")
                
                # 타입별 초기화
                self.indices[place_type] = faiss.IndexFlatL2(self.dimension)
                self.documents[place_type] = []
                self.metadata_list[place_type] = []
                
                # 데이터 로드
                query = text("""
                    SELECT "Name", "Category", "Menu", "Average_Cost", "Review_no", 
                           "rate", "address", "close_transport", "open_houe", 
                           "break_time", "dayoff", "contact", "convenience", 
                           "website", "type"
                    FROM pi_study.axpi_hailey_dataset 
                    WHERE "type" = :place_type
                """)
                
                try:
                    with self.engine.connect() as connection:
                        result = connection.execute(query, {"place_type": place_type})
                        # 컬럼 이름 가져오기
                        columns = result.keys()
                        places = []
                        for row in result:
                            # Row를 딕셔너리로 변환
                            row_dict = dict(zip(columns, row))
                            place = {
                                'name': row_dict['Name'],
                                'category': row_dict['Category'],
                                'menu': row_dict['Menu'],
                                'average_cost': row_dict['Average_Cost'],
                                'review_no': row_dict['Review_no'],
                                'rate': row_dict['rate'],
                                'address': row_dict['address'],
                                'close_transport': row_dict['close_transport'],
                                'open_hour': row_dict['open_houe'],  # 컬럼명 수정
                                'break_time': row_dict['break_time'],
                                'dayoff': row_dict['dayoff'],
                                'contact': row_dict['contact'],
                                'convenience': row_dict['convenience'],
                                'website': row_dict['website'],
                                'type': row_dict['type']
                            }
                            places.append(place)
                        
                        logger.info(f"{place_type} 데이터 {len(places)}개 로드됨")
                        if len(places) == 0:
                            logger.warning(f"{place_type} 데이터가 없습니다!")
                            continue
                        
                        # 각 장소에 대한 임베딩 생성
                        logger.info(f"{place_type} 임베딩 생성 시작")
                        embeddings = []
                        for place in places:
                            # 장소 정보를 문자열로 변환
                            place_text = f"{place['name']} {place['category']} {place['address']} {place['menu'] or ''}"
                            # 임베딩 생성
                            embedding = self.get_embedding(place_text)
                            if embedding is not None:
                                embeddings.append(embedding)
                                self.documents[place_type].append(place_text)
                                self.metadata_list[place_type].append(place)
                        
                        if embeddings:
                            # numpy 배열로 변환
                            embeddings_array = np.array(embeddings).astype('float32')
                            # FAISS 인덱스에 추가
                            self.indices[place_type].add(embeddings_array)
                            logger.info(f"{place_type} 임베딩 {len(embeddings)}개 생성 완료")
                            
                            # 인덱스와 메타데이터 저장
                            index_path = f'faiss_indexes/{self.place_type_mapping[place_type]}.index'
                            metadata_path = f'faiss_indexes/{self.place_type_mapping[place_type]}_metadata.json'
                            
                            # FAISS 인덱스 저장
                            faiss.write_index(self.indices[place_type], index_path)
                            
                            # 메타데이터 저장
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump({
                                    'documents': self.documents[place_type],
                                    'metadata_list': self.metadata_list[place_type]
                                }, f, ensure_ascii=False, indent=2)
                            
                            logger.info(f"{place_type} 인덱스와 메타데이터 저장 완료")
                        else:
                            logger.warning(f"{place_type} 임베딩 생성 실패")
                            
                except Exception as e:
                    logger.error(f"{place_type} 데이터 처리 중 오류: {str(e)}")
                    continue
            
            logger.info("모든 장소 데이터 로드 및 임베딩 생성 완료")
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def get_embedding(self, text):
        """텍스트의 임베딩 벡터를 생성"""
        try:
            # 최대 3번 재시도
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    embedding = np.array(response.data[0].embedding, dtype=np.float32)
                    return embedding
                except Exception as e:
                    if attempt == 2:  # 마지막 시도였을 경우
                        raise
                    time.sleep(1)  # 1초 대기 후 재시도
        
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            return None

    def search_similar_documents(self, query: str, place_type: str, top_k: int = 5) -> List[Dict]:
        """
        주어진 쿼리와 유사한 문서를 검색합니다.
        """
        logger.info("\n=== RAG 검색 시작 ===")
        logger.info(f"쿼리: '{query}'")
        logger.info(f"시설 유형: '{place_type}'")
        logger.info(f"요청된 결과 수: {top_k}")

        try:
            # 1. 인덱스 존재 여부 확인
            if place_type not in self.indices:
                logger.error(f"'{place_type}' 유형의 인덱스가 존재하지 않습니다.")
                return []

            index = self.indices[place_type]
            index_size = index.ntotal
            logger.info(f"인덱스 크기: {index_size}")
            logger.info(f"메타데이터 개수: {len(self.metadata_list.get(place_type, []))}")

            # 2. 데이터베이스에서 해당 유형의 데이터 확인
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    """
                    SELECT COUNT(*), array_agg("Name") as names 
                    FROM pi_study.axpi_hailey_dataset 
                    WHERE "type" = :place_type
                    """
                ), {"place_type": place_type})
                row = result.fetchone()
                db_count = row[0]
                sample_names = row[1] if row[1] else []
                logger.info(f"DB의 {place_type} 데이터 수: {db_count}")
                if sample_names:
                    logger.info(f"샘플 이름들: {', '.join(sample_names[:3])}...")

            # 3. 쿼리 임베딩 생성
            try:
                query_embedding = self.get_embedding(query)
                logger.info(f"쿼리 임베딩 생성 완료 (차원: {len(query_embedding)})")
            except Exception as e:
                logger.error(f"쿼리 임베딩 생성 실패: {str(e)}")
                return []

            # 4. FAISS 검색 수행
            D, I = index.search(np.array([query_embedding]).astype('float32'), min(top_k, index_size))
            logger.info(f"검색된 인덱스: {I[0].tolist()}")
            logger.info(f"유사도 점수: {D[0].tolist()}")

            # 5. 결과 가공
            results = []
            for idx, (distance, doc_idx) in enumerate(zip(D[0], I[0])):
                if doc_idx < 0 or doc_idx >= len(self.metadata_list.get(place_type, [])):
                    continue
                
                metadata = self.metadata_list[place_type][doc_idx]
                result = {
                    'rank': idx + 1,
                    'similarity': float(1 - distance),  # 거리를 유사도로 변환
                    **metadata
                }
                results.append(result)
                logger.info(f"결과 {idx+1}: {metadata.get('name', 'N/A')} (유사도: {result['similarity']:.3f})")

            return results

        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def query(self, query):
        """RAG 시스템을 통한 질의응답"""
        try:
            logger.info(f"검색 쿼리: {query}")
            
            # 질문 분석하여 적절한 type 결정
            place_type = self.analyze_place_type(query)
            logger.info(f"분석된 장소 타입: {place_type}")
            
            # 유사한 문서 검색
            similar_docs = self.search_similar_documents(query, place_type)
            if not similar_docs:
                logger.warning("유사한 문서를 찾을 수 없습니다.")
                return "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            
            logger.info(f"검색된 유사 문서 수: {len(similar_docs)}")
            context = "\n".join([doc['document'] for doc in similar_docs])
            
            # 응답 생성
            response = self.generate_response(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"질의응답 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

    def generate_response(self, query, context):
        """컨텍스트를 기반으로 응답 생성"""
        try:
            # 위치 정보 추출
            location_info = ""
            if "위치(" in query and ")" in query:
                location = query[query.find("(")+1:query.find(")")]
                location_info = f"\n\n현재 위치: {location}"
                # 위치 정보를 제외한 실제 검색어 추출
                real_query = query[query.find(")")+1:].strip()
            else:
                real_query = query
            
            prompt = f"""
            다음은 관련 장소 정보입니다:
            {context}
            {location_info}
            
            다음 질문에 답변해주세요:
            {real_query}
            
            답변은 한국어로 작성해주세요.
            위치 정보가 있는 경우, 해당 위치를 기준으로 답변해주세요.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful place recommendation assistant that answers questions based on the provided information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content
            logger.info(f"생성된 응답: {result}")
            return result
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            raise

    def analyze_place_type(self, query):
        """사용자 질문을 분석하여 적절한 type을 결정"""
        try:
            # 위치 정보가 포함된 경우 처리
            if "위치(" in query and ")" in query:
                # 위치 정보를 제외한 실제 검색어 추출
                real_query = query[query.find(")")+1:].strip()
                logger.info(f"위치 정보가 제거된 검색어: {real_query}")
            else:
                real_query = query
            
            # 병원 관련 키워드 체크
            hospital_keywords = [
                '이비인후과', '내과', '외과', '소아과', '산부인과', '정형외과', '안과', '치과',
                '피부과', '신경과', '정신과', '비뇨기과', '재활의학과', '마취통증의학과',
                '영상의학과', '진단검사의학과', '응급의학과', '가정의학과', '한의원', '의원',
                '병원', '코', '귀', '목', '감기', '수술', '아기', '산모', '관절', '눈', '치아',
                '피부', '뇌', '우울증', '전립선', '재활', '통증', 'MRI', 'CT', '검사', '응급',
                '건강검진', '한방', '한약'
            ]
            
            # 질문에 병원 관련 키워드가 포함되어 있는지 확인
            if any(keyword in real_query for keyword in hospital_keywords):
                logger.info(f"병원 관련 질문 감지: {real_query}")
                return "병원"
            
            prompt = f"""
            다음 질문에서 어떤 종류의 장소를 찾고 있는지 분석해주세요.
            가능한 type은 '음식점', '카페', '병원', '약국' 중 하나입니다.
            
            질문: {real_query}
            
            JSON 형식으로 응답해주세요:
            {{
                "type": "분석된 type",
                "reason": "분석 이유"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes Korean queries to determine the type of place being searched for."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"장소 타입 분석 결과: {analysis}")
            return analysis["type"]
            
        except Exception as e:
            logger.error(f"질문 분석 중 오류 발생: {str(e)}")
            return "음식점"  # 기본값 