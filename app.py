from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import logging
import sys
import traceback
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_system import RAGSystem
from sqlalchemy import text
from settings import create_postgres_engine_by_sqlalchemy
from difflib import SequenceMatcher
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import re

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI 관련 로그 레벨 설정
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# .env 파일을 절대경로로, override=True로 강제 적용
load_dotenv(dotenv_path="C:/Users/haenee/Desktop/naver-map-chat3/.env", override=True)

# API 키 디버그 로깅 추가
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
logger.debug(f"OpenAI API 키 로드됨: {OPENAI_API_KEY[:8]}...")  # 보안을 위해 앞부분만 로깅

if not OPENAI_API_KEY:
    logger.error("OpenAI API 키가 설정되지 않았습니다.")
    raise ValueError("OpenAI API 키가 필요합니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# 세션별 추천 결과 저장용 딕셔너리
session_recommendations = {}

try:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }})

    NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
    NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.error("네이버 API 키가 설정되지 않았습니다.")
        raise ValueError("네이버 API 키가 필요합니다.")

    # RAG 시스템 초기화 로깅 추가
    logger.debug("RAG 시스템 초기화 시작")
    rag = RAGSystem()
    logger.debug("RAG 시스템 초기화 완료")

    # 한국의 주요 지역 정보 - GPT 컨텍스트 강화용
    KOREA_LOCATIONS = """
    한국의 주요 지역 정보:
    - 서울: 강남구(압구정동, 청담동, 삼성동, 역삼동, 논현동), 서초구(서초동, 반포동, 방배동), 
           종로구(인사동, 삼청동, 북촌), 중구(명동, 을지로, 동대문), 마포구(홍대, 연남동, 합정동), 
           용산구(이태원, 한남동), 강서구, 강동구, 노원구 등
    - 부산: 해운대구(해운대, 마린시티), 수영구(광안리), 남구(용호동), 중구(남포동, 광복동), 서구(송도) 등
    - 인천: 중구(차이나타운, 월미도), 연수구(송도), 남동구(구월동) 등
    - 대구: 중구(동성로), 수성구(범어동, 두산동), 달서구(상인동) 등
    - 대전: 중구(은행동), 서구(둔산동), 유성구(봉명동) 등
    - 광주: 동구(충장로), 서구(상무지구), 북구(용봉동) 등
    - 제주: 제주시(이도동, 연동), 서귀포시(중문, 성산) 등
    """

    # 장소 유형 정보 - GPT 컨텍스트 강화용
    PLACE_TYPES = """
    장소 유형별 키워드:
    - 음식점/맛집: 식당, 레스토랑, 맛집, 음식점, 한식, 중식, 일식, 양식, 분식, 패스트푸드, 뷔페, 베이커리, 디저트
    - 카페: 카페, 커피숍, 디저트, 브런치, 베이커리, 차, 음료
    - 쇼핑: 백화점, 쇼핑몰, 마트, 상점, 시장, 아울렛, 편의점, 가게
    - 의료: 병원, 의원, 약국, 치과, 한의원, 보건소, 의료원, 클리닉
    - 교통: 지하철역, 버스정류장, 터미널, 공항, 기차역, 정류장, 역
    - 숙박: 호텔, 모텔, 게스트하우스, 리조트, 펜션, 숙소, 콘도
    - 관광: 명소, 관광지, 박물관, 미술관, 전시관, 유적지, 공원, 테마파크
    - 문화/예술: 영화관, 공연장, 전시장, 극장, 문화센터, 스튜디오
    - 교육: 학교, 학원, 도서관, 독서실, 교육원, 대학교, 유치원
    - 스포츠/레저: 체육관, 수영장, 헬스장, 운동장, 골프장, 스키장
    """

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # 대화 기록을 저장할 전역 딕셔너리
    conversation_history = {}

    # 대화 기록 관리 클래스
    class ConversationManager:
        def __init__(self, max_history=5, expiry_hours=1):
            self.max_history = max_history
            self.expiry_hours = expiry_hours
            self.conversations = {}

        def add_message(self, session_id, role, content):
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    'messages': [],
                    'last_updated': datetime.now()
                }
            
            # 만료된 대화 기록 삭제
            self._cleanup_expired()
            
            # 새 메시지 추가
            self.conversations[session_id]['messages'].append({
                'role': role,
                'content': content,
                'timestamp': datetime.now()
            })
            
            # 최대 기록 수 제한
            if len(self.conversations[session_id]['messages']) > self.max_history:
                self.conversations[session_id]['messages'] = self.conversations[session_id]['messages'][-self.max_history:]
            
            self.conversations[session_id]['last_updated'] = datetime.now()

        def get_messages(self, session_id):
            if session_id not in self.conversations:
                return []
            return self.conversations[session_id]['messages']

        def _cleanup_expired(self):
            current_time = datetime.now()
            expired_sessions = [
                session_id for session_id, data in self.conversations.items()
                if current_time - data['last_updated'] > timedelta(hours=self.expiry_hours)
            ]
            for session_id in expired_sessions:
                del self.conversations[session_id]

    # 대화 관리자 인스턴스 생성
    conversation_manager = ConversationManager()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/search', methods=['POST'])
    def search():
        try:
            if not request.is_json:
                logger.error("JSON이 아닌 요청이 들어왔습니다.")
                return jsonify({'error': '요청이 JSON 형식이어야 합니다.'}), 400

            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                logger.error("검색어가 비어있습니다.")
                return jsonify({'error': '검색어를 입력해주세요.', 'type': 'error'}), 400
            
            # 시설 유형 분석
            facility_type, confidence_score, matched_keyword = analyze_facility_type(query)
            
            if not facility_type or confidence_score < 0.3:
                logger.warning(f"낮은 신뢰도 점수: {confidence_score}")
                return jsonify({
                    'error': '시설 유형을 정확히 파악할 수 없습니다. 더 구체적으로 말씀해 주세요.',
                    'type': 'error'
                }), 400

            try:
                logger.info("\n[RAG 검색 시작]")
                logger.info(f"- 시설 유형: {facility_type} (신뢰도: {confidence_score}, 키워드: {matched_keyword})")
                logger.info(f"검색 파라미터 - 쿼리: '{query}', 시설 유형: '{facility_type}', top_k: 3")
                
                # RAG 시스템으로 시설 검색
                search_results = rag.search_similar_documents("신사옥", facility_type, top_k=3)
                
                if search_results:
                    logger.info(f"검색 결과 {len(search_results)}개 찾음")
                    logger.debug(f"검색 결과 상세: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
                    
                    # GPT로 자연어 응답 생성
                    response_prompt = f"""
                    F&F 신사옥 주변의 {facility_type} 검색 결과입니다:
                    {json.dumps(search_results, ensure_ascii=False, indent=2)}
                    
                    이 정보를 자연스러운 한국어로 설명해주세요. 각 시설의 이름, 위치(지하철역 기준), 영업시간, 특징적인 정보를 포함해주세요.
                    응답은 친근하고 대화체로 작성해주세요.
                    """
                    
                    natural_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 친근한 말투로 주변 시설을 추천해주는 도우미입니다."},
                            {"role": "user", "content": response_prompt}
                        ]
                    )
                    
                    response_text = natural_response.choices[0].message.content
                    logger.info(f"GPT 응답: {response_text}")
                    
                    # 검색 결과를 지도에 표시하기 위한 좌표 정보 추가
                    for place in search_results:
                        try:
                            # 네이버 지도 API로 좌표 검색
                            url = "https://openapi.naver.com/v1/search/local.json"
                            headers = {
                                "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                                "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
                            }
                            params = {
                                "query": f"{place['name']} {place['address']}",
                                "display": 1
                            }
                            
                            map_response = requests.get(url, headers=headers, params=params)
                            map_data = map_response.json()
                            
                            if map_data.get('items'):
                                item = map_data['items'][0]
                                place['x'] = float(item['mapx'])  # 경도
                                place['y'] = float(item['mapy'])  # 위도
                        except Exception as e:
                            logger.error(f"좌표 검색 실패: {str(e)}")
                            continue
                    
                    # 추천 결과를 세션별로 저장
                    session_id = data.get('session_id', 'default')
                    session_recommendations[session_id] = search_results
                    
                    response_data = {
                        'type': 'places',
                        'places': search_results,
                        'response': response_text
                    }
                    logger.info("\n[응답 전송]")
                    logger.info(f"응답 데이터: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                    return jsonify(response_data)
                    
                else:
                    logger.warning("검색 결과가 없습니다.")
                    return jsonify({
                        'type': 'chat',
                        'response': f'신사옥 주변에 등록된 {facility_type} 정보가 없습니다. 혹시 찾으시는 진료과목이나 매장명을 더 알려주실 수 있나요?'
                    })
                
            except Exception as e:
                logger.error(f"\n[오류 발생]\n{str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    'error': '시설 검색 중 오류가 발생했습니다.',
                    'type': 'error'
                }), 500

        except Exception as e:
            logger.error(f"\n[오류 발생]\n{str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

    @app.route('/api/search', methods=['POST', 'GET'])
    def search_api():
        if request.method == 'GET':
            query = request.args.get('query', '')
        else:  # POST
            data = request.get_json()
            query = data.get('query', '')
            session_id = data.get('session_id', 'default')
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        logger.debug(f"검색 요청: {query}")
        
        try:
            # 사용자 메시지 저장
            conversation_manager.add_message(session_id, 'user', query)
            
            # 이전 대화 기록 가져오기
            conversation_history = conversation_manager.get_messages(session_id)
            

            
            # 사용자 위치 정보 가져오기
            user_location = None
            if request.method == 'POST' and request.get_json():
                user_location = request.get_json().get('user_location')
            
            # 쿼리 분석
            analysis = analyze_query(query, user_location, session_id)
            logger.debug(f"쿼리 분석 결과: {analysis}")
            
            # 순서 지칭 패턴 직접 검사 (GPT 분석 보완)
            ordinal_match = re.search(r'(\d+)\s*번', query)
            if ordinal_match:
                ordinal = int(ordinal_match.group(1))
                if session_id in session_recommendations and len(session_recommendations[session_id]) >= ordinal:
                    target_place = session_recommendations[session_id][ordinal-1]
                    logger.info(f"직접 순서 지칭 감지: {ordinal}번째 - {target_place['name']}")
                    
                    # 길찾기 요청인지 확인
                    is_directions = any(keyword in query for keyword in ['어떻게 가', '가는 길', '가는 방법', '찾아가', '길찾기', '안내'])
                    
                    if is_directions:
                        # 길찾기 처리
                        target_address = target_place['address']
                        location_result = geocode_location(target_address)
                        
                        if location_result.get('addresses'):
                            target_coords = location_result['addresses'][0]
                            start_coords = {
                                'x': '127.0310195',  # 경도
                                'y': '37.4982517'    # 위도
                            }
                            
                            route_result = get_directions_data(
                                f"{start_coords['x']},{start_coords['y']}",
                                f"{target_coords['x']},{target_coords['y']}"
                            )
                            
                            if route_result.get('route'):
                                summary = route_result['route']['trafast'][0]['summary']
                                distance = summary['distance']
                                duration_minutes = int(summary['duration'] / (1000 * 60))
                                
                                directions_prompt = f"""
                                F&F 신사옥에서 {target_place['name']}까지의 길찾기 정보입니다:
                                - 주소: {target_address}
                                - 총 거리: {distance}m
                                - 예상 소요 시간: {duration_minutes}분
                                
                                이 정보를 자연스러운 대화체 한국어로 설명해주세요.
                                인사말 없이 바로 길찾기 정보부터 시작해주세요.
                                """
                                
                                natural_response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "당신은 친근한 말투로 길 안내를 해주는 도우미입니다. 인사말 없이 바로 핵심 정보를 전달하세요."},
                                        {"role": "user", "content": directions_prompt}
                                    ]
                                )
                                
                                response_text = natural_response.choices[0].message.content
                                conversation_manager.add_message(session_id, 'assistant', response_text)
                                
                                # 현재 참조 장소 업데이트 (길찾기 대상을 현재 참조로 설정)
                                session_recommendations[session_id + '_current'] = target_place
                                
                                return jsonify({
                                    'type': 'directions',
                                    'response': response_text,
                                    'route': route_result['route'],
                                    'summary': {
                                        'distance': distance,
                                        'duration_minutes': duration_minutes
                                    },
                                    'start': start_coords,
                                    'end': {
                                        'name': target_place['name'],
                                        'address': target_address,
                                        'x': target_coords['x'],
                                        'y': target_coords['y']
                                    },
                                    'session_id': session_id
                                })
                    else:
                        # 순서 지칭 + 정보 요청 (메뉴, 영업시간 등)
                        info_prompt = f"""
                        사용자가 "{target_place['name']}"에 대해 다음과 같이 질문했습니다: "{query}"
                        
                        장소 정보:
                        - 이름: {target_place['name']}
                        - 카테고리: {target_place.get('category', '정보 없음')}
                        - 메뉴: {target_place.get('menu', '정보 없음')}
                        - 주소: {target_place['address']}
                        - 영업시간: {target_place.get('open_hour', '정보 없음')}
                        - 연락처: {target_place.get('contact', '정보 없음')}
                        - 평점: {target_place.get('rate', '정보 없음')}
                        
                        사용자의 질문에 대해 친근하고 자연스러운 대화체로 답변해주세요.
                        """
                        
                        info_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 친근한 말투로 장소 정보를 제공하는 도우미입니다."},
                                {"role": "user", "content": info_prompt}
                            ]
                        )
                        
                        response_text = info_response.choices[0].message.content
                        conversation_manager.add_message(session_id, 'assistant', response_text)
                        
                        # 현재 참조 장소 업데이트
                        session_recommendations[session_id + '_current'] = target_place
                        
                        return jsonify({
                            'type': 'chat',
                            'response': response_text,
                            'place': target_place,
                            'session_id': session_id
                        })
                else:
                    return jsonify({
                        'type': 'chat',
                        'response': f"{ordinal}번째 추천 결과가 없습니다.",
                        'session_id': session_id
                    })

            # GPT가 순서 지칭으로 분석한 경우 처리 (백업)
            if analysis['query_type'] in ['ordinal_reference', 'ordinal_directions']:
                ordinal = analysis.get('ordinal_number')
                if ordinal and session_id in session_recommendations:
                    try:
                        # ordinal_number가 문자열일 수 있으므로 정수로 변환
                        ordinal = int(ordinal) if isinstance(ordinal, str) else ordinal
                        target_place = session_recommendations[session_id][ordinal-1]
                        logger.info(f"GPT 순서 지칭 감지: {ordinal}번째 - {target_place['name']}")
                        
                        # 길찾기 요청인지 확인 (ordinal_directions이거나 키워드 포함)
                        is_directions = (analysis['query_type'] == 'ordinal_directions')
                        
                        if is_directions:
                            # 길찾기 처리
                            target_address = target_place['address']
                            location_result = geocode_location(target_address)
                            
                            if location_result.get('addresses'):
                                target_coords = location_result['addresses'][0]
                                start_coords = {
                                    'x': '127.0310195',  # 경도
                                    'y': '37.4982517'    # 위도
                                }
                                
                                route_result = get_directions_data(
                                    f"{start_coords['x']},{start_coords['y']}",
                                    f"{target_coords['x']},{target_coords['y']}"
                                )
                                
                                if route_result.get('route'):
                                    summary = route_result['route']['trafast'][0]['summary']
                                    distance = summary['distance']
                                    duration_minutes = int(summary['duration'] / (1000 * 60))
                                    
                                    directions_prompt = f"""
                                    F&F 신사옥에서 {target_place['name']}까지의 길찾기 정보입니다:
                                    - 주소: {target_address}
                                    - 총 거리: {distance}m
                                    - 예상 소요 시간: {duration_minutes}분
                                    
                                    이 정보를 자연스러운 대화체 한국어로 설명해주세요.
                                    인사말 없이 바로 길찾기 정보부터 시작해주세요.
                                    """
                                    
                                    natural_response = client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "당신은 친근한 말투로 길 안내를 해주는 도우미입니다. 인사말 없이 바로 핵심 정보를 전달하세요."},
                                            {"role": "user", "content": directions_prompt}
                                        ]
                                    )
                                    
                                    response_text = natural_response.choices[0].message.content
                                    conversation_manager.add_message(session_id, 'assistant', response_text)
                                    return jsonify({
                                        'type': 'directions',
                                        'response': response_text,
                                        'route': route_result['route'],
                                        'summary': {
                                            'distance': distance,
                                            'duration_minutes': duration_minutes
                                        },
                                        'start': start_coords,
                                        'end': {
                                            'name': target_place['name'],
                                            'address': target_address,
                                            'x': target_coords['x'],
                                            'y': target_coords['y']
                                        },
                                        'session_id': session_id
                                    })
                        else:
                            # 단순 정보 요청
                            response_text = f"{ordinal}번째 추천: {target_place['name']}\n주소: {target_place['address']}\n연락처: {target_place.get('contact', '정보 없음')}\n영업시간: {target_place.get('open_hour', '정보 없음')}"
                            conversation_manager.add_message(session_id, 'assistant', response_text)
                            return jsonify({
                                'type': 'chat',
                                'response': response_text,
                                'place': target_place,
                                'session_id': session_id
                            })
                            
                    except IndexError:
                        return jsonify({
                            'type': 'chat',
                            'response': "해당 순서의 추천 결과가 없습니다.",
                            'session_id': session_id
                        })
            
            if analysis['query_type'] == 'directions':
                logger.info("\n[길찾기 검색 시작]")
                target_name = analysis['target_location']
                logger.info(f"- 목적지: {target_name}")
                
                # target_location이 없거나 None인 경우 쿼리에서 직접 추출 시도
                if not target_name or target_name == 'null':
                    # "XXX는 어떻게 가?" 패턴에서 XXX 추출
                    match = re.search(r'(.+?)(?:는|은|을|를)?\s*어떻게\s*가', query)
                    if match:
                        target_name = match.group(1).strip()
                        logger.info(f"- 쿼리에서 목적지 재추출: {target_name}")
                    else:
                        return jsonify({
                            'type': 'chat',
                            'response': "어디로 가시려는지 명확하지 않습니다. 예: '다미는 어떻게 가?' 형태로 질문해주세요.",
                            'session_id': session_id
                        })
                
                try:
                    # 먼저 세션에서 최근 추천된 장소 중에 검색
                    search_results = None
                    target_place = None
                    
                    # 1. 세션의 추천 결과에서 먼저 찾기
                    if session_id in session_recommendations and session_recommendations[session_id]:
                        for place in session_recommendations[session_id]:
                            if target_name.lower() in place['name'].lower() or place['name'].lower() in target_name.lower():
                                target_place = place
                                logger.info(f"- 세션 추천 결과에서 발견: {place['name']}")
                                break
                    
                    # 2. 세션에서 못 찾으면 RAG에서 검색 (카페를 먼저 검색)
                    if not target_place:
                        for facility_type in ["카페", "음식점", "병원", "약국"]:
                            search_results = rag.search_similar_documents(target_name, facility_type, top_k=5)
                            if search_results:
                                logger.info(f"- {facility_type}에서 검색 결과 발견")
                                break
                    
                    # 3. 세션과 RAG 모두에서 못 찾은 경우
                    if not target_place and not search_results:
                        return jsonify({
                            'type': 'chat',
                            'response': f"죄송합니다. '{target_name}'의 정보를 찾을 수 없습니다."
                        })
                    
                    # 4. 세션에서 찾은 경우 바로 사용, RAG에서 찾은 경우 유사도 계산
                    if target_place:
                        # 세션에서 찾은 경우 바로 사용
                        best_match = target_place
                        highest_similarity = 1.0  # 세션에서 찾았으므로 높은 신뢰도
                    else:
                        # RAG 검색 결과 중에서 가장 유사한 장소 찾기
                        best_match = None
                        highest_similarity = 0
                        
                        for result in search_results:
                            # 전체 이름과의 유사도 계산
                            full_similarity = similar(target_name, result['name'])
                            
                            # 부분 문자열 포함 여부 확인 (공백 제거 후 비교)
                            target_clean = target_name.replace(" ", "").lower()
                            result_clean = result['name'].replace(" ", "").lower()
                            contains_partial = target_clean in result_clean
                            
                            # 입력된 단어들과의 개별 유사도 계산
                            target_words = target_name.split()
                            name_words = result['name'].split()
                            word_similarities = []
                            for tw in target_words:
                                tw_clean = tw.lower()
                                best_word_match = max([similar(tw_clean, nw.lower()) for nw in name_words])
                                word_similarities.append(best_word_match)
                            avg_word_similarity = sum(word_similarities) / len(word_similarities)
                            
                            # 종합 유사도 점수 계산 (가중치 조정)
                            similarity_score = max(
                                full_similarity * 0.3,      # 전체 문자열 유사도 (30%)
                                avg_word_similarity * 0.3,  # 단어별 평균 유사도 (30%)
                                0.9 if contains_partial else 0  # 부분 문자열 포함 보너스 (90%)
                            )
                            
                            logger.debug(f"""
                            유사도 분석:
                            - 검색어: {target_name}
                            - 병원명: {result['name']}
                            - 전체 유사도: {full_similarity}
                            - 단어별 평균 유사도: {avg_word_similarity}
                            - 부분 문자열 포함: {contains_partial}
                            - 최종 유사도 점수: {similarity_score}
                            """)
                            
                            if similarity_score > highest_similarity:
                                highest_similarity = similarity_score
                                best_match = result
                    
                    if highest_similarity >= 0.2:  # 유사도 임계값 낮춤 (20%)
                        target_place = best_match
                        target_name = target_place['name']  # 정확한 병원 이름으로 업데이트
                        target_address = target_place['address']
                        
                        logger.info(f"- 검색된 병원: {target_name}")
                        logger.info(f"- 검색된 주소: {target_address}")
                        logger.info(f"- 유사도 점수: {highest_similarity}")
                        
                        # 주소를 좌표로 변환
                        location_result = geocode_location(target_address)
                    else:
                        return jsonify({
                            'type': 'chat',
                            'response': f"죄송합니다. '{target_name}'와 유사한 장소를 찾을 수 없습니다."
                        })
                    
                    if not location_result.get('addresses'):
                        return jsonify({
                            'type': 'chat',
                            'response': f"죄송합니다. '{target_name}'의 주소를 찾을 수 없습니다."
                        })
                    
                    target_coords = location_result['addresses'][0]
                    
                    # 출발지 좌표 (F&F 신사옥)
                    start_coords = {
                        'x': '127.0310195',  # 경도
                        'y': '37.4982517'    # 위도
                    }
                    
                    # 길찾기 API 호출
                    route_result = get_directions_data(
                        f"{start_coords['x']},{start_coords['y']}",
                        f"{target_coords['x']},{target_coords['y']}"
                    )
                    
                    if route_result.get('route'):
                        summary = route_result['route']['trafast'][0]['summary']
                        distance = summary['distance']
                        duration_minutes = int(summary['duration'] / (1000 * 60))
                        
                        logger.debug(f"""
                        [길찾기 결과]
                        - 목적지 이름: {target_name}
                        - 목적지 주소: {target_address}
                        - 목적지 좌표: {target_coords['x']}, {target_coords['y']}
                        - 거리: {distance}m
                        - 시간: {duration_minutes}분
                        """)
                        
                        # GPT로 길찾기 결과 자연어 변환
                        directions_prompt = f"""
                        F&F 신사옥에서 {target_name}까지의 길찾기 정보입니다:
                        - 병원 주소: {target_address}
                        - 총 거리: {distance}m
                        - 예상 소요 시간: {duration_minutes}분
                        
                        이 정보를 자연스러운 대화체 한국어로 설명해주세요.
                        인사말 없이 바로 길찾기 정보부터 시작해주세요.
                        """
                        
                        natural_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 친근한 말투로 길 안내를 해주는 도우미입니다. 인사말 없이 바로 핵심 정보를 전달하세요."},
                                {"role": "user", "content": directions_prompt}
                            ]
                        )
                        
                        response_text = natural_response.choices[0].message.content
                        conversation_manager.add_message(session_id, 'assistant', response_text)
                        
                        # 현재 참조 장소 업데이트 (길찾기 대상을 현재 참조로 설정)
                        session_recommendations[session_id + '_current'] = target_place
                        
                        return jsonify({
                            'type': 'directions',
                            'response': response_text,
                            'route': route_result['route'],
                            'summary': {
                                'distance': distance,
                                'duration_minutes': duration_minutes
                            },
                            'start': start_coords,
                            'end': {
                                'name': target_name,
                                'address': target_address,
                                'x': target_coords['x'],
                                'y': target_coords['y']
                            },
                            'session_id': session_id
                        })
                except Exception as e:
                    logger.error(f"길찾기 중 오류 발생: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'type': 'chat',
                        'response': '죄송합니다. 길찾기 중 오류가 발생했습니다.'
                    })
            
            # "한군데만", "하나만" 등의 개수 조정 요청 처리
            if analysis['query_type'] == 'facility_search' and any(keyword in query for keyword in ['한군데', '하나만', '한 곳', '1개', '하나']):
                # 이전 추천 결과가 있다면 첫 번째만 반환
                if session_id in session_recommendations and session_recommendations[session_id]:
                    # 이전 조건과 같은 정렬 기준 적용
                    search_results = session_recommendations[session_id][:1]  # 첫 번째만
                    
                    response_prompt = f"""
                    이전에 추천한 결과 중 가장 좋은 한 곳을 선별해서 알려드립니다:
                    {json.dumps(search_results, ensure_ascii=False, indent=2)}
                    
                    이 정보를 자연스러운 한국어로 설명해주세요. 시설의 이름, 위치, 특징을 포함하되, 
                    "요청하신 대로 한 곳만" 추천한다는 점을 언급해주세요.
                    """
                    
                    natural_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 친근한 말투로 주변 시설을 추천해주는 도우미입니다."},
                            {"role": "user", "content": response_prompt}
                        ]
                    )
                    
                    response_text = natural_response.choices[0].message.content
                    conversation_manager.add_message(session_id, 'assistant', response_text)
                    
                    return jsonify({
                        'type': 'chat',
                        'response': response_text,
                        'places': search_results,
                        'session_id': session_id
                    })
                # 이전 추천 결과가 없으면 새로 검색 (아래 facility_search 로직으로 진행)
                
            if analysis['query_type'] == 'facility_search':
                try:
                    logger.info("\n[RAG 검색 시작]")
                    logger.info(f"- 시설 유형: {analysis['place_type']}")
                    
                    # 요청 개수 판단 (하나만, 한군데만 등의 키워드가 있으면 1개만)
                    top_k = 1 if any(keyword in query for keyword in ['한군데', '하나만', '한 곳', '1개', '하나']) else 3
                    
                    # RAG 시스템으로 시설 검색
                    search_results = rag.search_similar_documents(query, analysis['place_type'], top_k=top_k)
                    
                    if search_results:
                        logger.info(f"검색 결과 {len(search_results)}개 찾음")
                        
                        # 정렬 기준에 따라 결과 재정렬
                        sort_criteria = analysis.get('sort_criteria', 'relevance')
                        if sort_criteria == 'rating':
                            # 평점 순으로 정렬 (평점이 있는 것 우선, 높은 순으로)
                            search_results = sorted(search_results, key=lambda x: float(x.get('rate', 0) or 0), reverse=True)
                            logger.info("평점 기준으로 결과 재정렬됨")
                        
                        logger.debug(f"검색 결과 상세: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
                        
                        # GPT로 자연어 응답 생성
                        if sort_criteria == 'rating':
                            response_prompt = f"""
                            F&F 신사옥 주변의 {analysis['place_type']} 중 평점이 높은 곳들을 평점 순으로 정렬한 결과입니다:
                            {json.dumps(search_results, ensure_ascii=False, indent=2)}
                            
                            이 정보를 자연스러운 한국어로 설명해주세요. 각 시설의 이름, 평점, 위치(지하철역 기준), 특징적인 정보를 포함하되, 평점이 높은 순서대로 추천한다는 점을 강조해주세요.
                            응답은 친근하고 대화체로 작성하되, 인사말 없이 바로 추천 정보부터 시작해주세요.
                            """
                        else:
                            response_prompt = f"""
                            F&F 신사옥 주변의 {analysis['place_type']} 검색 결과입니다:
                            {json.dumps(search_results, ensure_ascii=False, indent=2)}
                            
                            이 정보를 자연스러운 한국어로 설명해주세요. 각 시설의 이름, 위치(지하철역 기준),특징적인 정보를 포함해주세요.
                            응답은 친근하고 대화체로 작성하되, 인사말 없이 바로 추천 정보부터 시작해주세요.
                            """
                        
                        natural_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 친근한 말투로 주변 시설을 추천해주는 도우미입니다. 인사말 없이 바로 핵심 정보를 전달하세요."},
                                {"role": "user", "content": response_prompt}
                            ]
                        )
                        
                        response_text = natural_response.choices[0].message.content
                        conversation_manager.add_message(session_id, 'assistant', response_text)
                        
                        # 추천 결과를 세션별로 저장 (순서 지칭을 위해)
                        session_recommendations[session_id] = search_results
                        
                        return jsonify({
                            'type': 'chat',
                            'response': response_text,
                            'places': search_results,
                            'session_id': session_id
                        })
                    else:
                        logger.warning("검색 결과가 없습니다.")
                        return jsonify({
                            'type': 'chat',
                            'response': f'신사옥 주변에 등록된 {analysis["place_type"]} 정보가 없습니다. 혹시 찾으시는 진료과목이나 매장명을 더 알려주실 수 있나요?'
                        })
                
                except Exception as e:
                    logger.error(f"시설 검색 중 오류 발생: {str(e)}")
                    logger.error(traceback.format_exc())
                return jsonify({
                        'error': '시설 검색 중 오류가 발생했습니다.',
                        'type': 'error'
                    }), 500
            
            elif analysis['query_type'] == 'info_request':
                # 먼저 현재 참조 장소가 있는지 확인 (길찾기나 순서 지칭으로 설정된 장소)
                current_place_key = session_id + '_current'
                target_place = None
                
                if current_place_key in session_recommendations:
                    target_place = session_recommendations[current_place_key]
                    logger.info(f"현재 참조 장소 사용: {target_place['name']}")
                elif session_id in session_recommendations and session_recommendations[session_id]:
                    # 현재 참조가 없으면 첫번째 추천 장소 사용
                    target_place = session_recommendations[session_id][0]
                    logger.info(f"첫번째 추천 장소 사용: {target_place['name']}")
                
                if target_place:
                    # 메뉴나 정보 요청에 대한 응답 생성
                    info_prompt = f"""
                    사용자가 "{target_place['name']}"에 대해 다음과 같이 질문했습니다: "{query}"
                    
                    장소 정보:
                    - 이름: {target_place['name']}
                    - 카테고리: {target_place.get('category', '정보 없음')}
                    - 메뉴: {target_place.get('menu', '정보 없음')}
                    - 주소: {target_place['address']}
                    - 영업시간: {target_place.get('open_hour', '정보 없음')}
                    - 연락처: {target_place.get('contact', '정보 없음')}
                    - 평점: {target_place.get('rate', '정보 없음')}
                    - 편의시설: {target_place.get('convenience', '정보 없음')}
                    
                    사용자의 질문에 대해 친근하고 자연스러운 대화체로 답변해주세요.
                    """
                    
                    info_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 친근한 말투로 장소 정보를 제공하는 도우미입니다."},
                            {"role": "user", "content": info_prompt}
                        ]
                    )
                    
                    response_text = info_response.choices[0].message.content
                    conversation_manager.add_message(session_id, 'assistant', response_text)
                    return jsonify({
                        'type': 'chat',
                        'response': response_text,
                        'place': target_place,
                        'session_id': session_id
                    })
                else:
                    return jsonify({
                        'type': 'chat',
                        'response': '어떤 장소에 대한 정보를 원하시는지 명확하지 않습니다. 먼저 장소를 검색하거나 추천받아보세요.',
                        'session_id': session_id
                    })
            
            # 위치 검색인 경우
            elif analysis.get('is_location_query'):
                location_results = geocode_location(analysis['location_query'])
                
                if not location_results.get('addresses') or len(location_results['addresses']) == 0:
                    logger.debug(f"위치 검색 실패, 원본 쿼리로 재시도: {query}")
                    location_results = geocode_location(query)
                
                if location_results.get('addresses') and len(location_results['addresses']) > 0:
                    enhanced_results = enhance_results_with_context(
                        location_results, 
                        analysis,
                        query
                    )
                    return jsonify(enhanced_results)
                else:
                    return jsonify({
                        "status": "ZERO_RESULTS",
                        "original_query": query,
                        "addresses": []
                    })
            else:
                # 기본 처리: 분류되지 않은 경우
                logger.warning(f"분류되지 않은 쿼리: {query}, 분석 결과: {analysis}")
                return jsonify({
                    'type': 'chat',
                    'response': '죄송합니다. 요청을 이해하지 못했습니다. 다시 말씀해 주세요.',
                    'session_id': session_id
                })
                
        except Exception as e:
            logger.error(f"검색 처리 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"검색 처리 중 오류가 발생했습니다: {str(e)}"}), 500

    def analyze_query(query: str, user_location=None, session_id='default') -> dict:
        """
        GPT를 사용하여 사용자 쿼리의 의도를 자연어로 분석
        Returns: 분석 결과 딕셔너리
        """
        try:
            # 이전 대화 기록 가져오기
            conversation_history = conversation_manager.get_messages(session_id)
            context_info = ""
            if conversation_history:
                recent_messages = conversation_history[-3:]  # 최근 3개 메시지만
                context_info = "\n이전 대화 맥락:\n"
                for msg in recent_messages:
                    context_info += f"- {msg['role']}: {msg['content'][:100]}...\n"

            # GPT를 사용한 의도 분석
            analysis_prompt = f"""
            사용자의 질문을 분석하여 의도를 파악해주세요.
            
            현재 질문: "{query}"
            {context_info}
            
            다음 중 하나로 분류해주세요:
            1. facility_search: 새로운 시설/매장 검색 요청 (예: "병원 추천해줘", "카페 어디 있어?")
            2. directions: 길찾기 요청 (예: "어떻게 가?", "스터번 어떻게 가?")
            3. ordinal_reference: 순서 지칭 (예: "1번", "첫번째", "2번 정보")
            4. ordinal_directions: 순서 + 길찾기 (예: "1번 어떻게 가?")
            5. info_request: 특정 시설 정보 요청 (예: "메뉴 뭐있어?", "영업시간", "X번 메뉴", "몇 번 메인메뉴")
            6. unknown: 위에 해당하지 않는 경우
            
            **대화 맥락 주의사항**:
            - 이전에 추천을 받은 후 "한군데만", "하나만", "더 적게" 등은 같은 조건으로 개수만 줄이려는 의도
            - 평점/리뷰 관련 키워드가 있으면 sort_criteria를 "rating"으로 설정
            - 길찾기에서 목적지명이 명시되면 target_location에 정확히 기록
            
            JSON 형태로 답변:
            {{
                "query_type": "위 분류 중 하나",
                "place_type": "병원/음식점/카페/약국/없음",
                "target_location": "길찾기 목적지 (없으면 null)",
                "ordinal_number": "순서 번호 (없으면 null)",
                "sort_criteria": "rating 또는 relevance",
                "confidence": 0.0-1.0
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 사용자가 원하는 것을 정확히 이해하는 AI입니다. 대화 맥락을 고려해서 JSON으로만 답변하세요."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1  # 일관성을 위해 낮은 temperature 사용
            )
            
            # GPT 응답을 JSON으로 파싱
            gpt_response = response.choices[0].message.content.strip()
            logger.debug(f"GPT 의도 분석 응답: {gpt_response}")
            
            try:
                analysis_result = json.loads(gpt_response)
                logger.info(f"GPT 분석 결과: {analysis_result}")
                return analysis_result
            except json.JSONDecodeError:
                logger.error(f"GPT 응답 JSON 파싱 실패: {gpt_response}")
                # 파싱 실패 시 기존 방식으로 폴백
                return _fallback_analysis(query)
                
        except Exception as e:
            logger.error(f"GPT 의도 분석 중 오류 발생: {str(e)}")
            # 오류 시 기존 방식으로 폴백
            return _fallback_analysis(query)
    
    def _fallback_analysis(query: str) -> dict:
        """GPT 분석 실패 시 사용할 기존 키워드 기반 분석"""
        try:
            # 길찾기 관련 키워드
            directions_keywords = ['어떻게 가', '가는 길', '가는 방법', '찾아가', '가는길', '까지 가는']
            
            # 평점 관련 키워드
            rating_keywords = ['리뷰', '평점', '평가', '좋은', '맛있는', '인기', '추천']
            is_rating_query = any(keyword in query for keyword in rating_keywords)
            
            # 길찾기 요청인지 확인
            for keyword in directions_keywords:
                if keyword in query:
                    target_location = query.split(keyword)[0].strip()
                    return {
                        'query_type': 'directions',
                        'target_location': target_location,
                        'place_type': '없음',
                        'sort_criteria': 'relevance',
                        'confidence': 0.8
                    }
            
            # 시설 유형 분석
            place_type = analyze_place_type(query)
            
            if place_type != '없음':
                return {
                    'query_type': 'facility_search',
                    'place_type': place_type,
                    'target_location': None,
                    'sort_criteria': 'rating' if is_rating_query else 'relevance',
                    'confidence': 0.7
                }
            
            return {
                'query_type': 'unknown',
                'place_type': '없음',
                'target_location': None,
                'sort_criteria': 'relevance',
                'confidence': 0.3
            }
            
        except Exception as e:
            logger.error(f"폴백 분석 중 오류 발생: {str(e)}")
            return {
                'query_type': 'error',
                'error_message': str(e),
                'confidence': 0.0
            }

    def geocode_location(location_query):
        """
        네이버 Geocoding API를 사용하여 위치 정보를 좌표로 변환합니다.
        """
        url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
        }
        params = {"query": location_query}
        
        try:
            logger.debug(f"Geocoding API 요청: {url}, 파라미터: {params}")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            results = response.json()
            logger.debug(f"Geocoding API 응답: {results}")
            
            return results
        except Exception as e:
            logger.error(f"Geocoding API 요청 중 오류 발생: {str(e)}")
            return {"error": f"위치 검색 중 오류가 발생했습니다: {str(e)}"}

    def enhance_results_with_context(location_results, analysis, original_query):
        """
        검색 결과에 추가 컨텍스트를 더하여 강화합니다.
        """
        try:
            enhanced_results = location_results.copy()
            
            # 원본 쿼리와 분석 정보 추가
            enhanced_results["original_query"] = original_query
            enhanced_results["analysis"] = analysis
            
            # 장소 유형이 있는 경우, GPT를 활용해 결과에 추가 정보 제공
            if analysis.get('place_type'):
                # 첫 번째 주소 정보 가져오기
                first_location = location_results['addresses'][0] if location_results.get('addresses') else None
                
                if first_location:
                    place_type = analysis['place_type']
                    location_name = first_location.get('roadAddress') or first_location.get('jibunAddress')
                    
                    # GPT를 사용하여 해당 위치와 장소 유형에 대한 추가 정보 생성
                    context_prompt = f"{location_name}의 {place_type}에 대한 간략한 정보를 제공해주세요."
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"""
                            당신은 위치 기반 정보 제공 전문가입니다. 
                            특정 위치와 장소 유형에 관해 사용자가 알면 도움이 될 간략한 정보를 제공하세요.
                            답변은 3-5줄 내외로 짧고 유용하게 작성하세요.
                            """},
                            {"role": "user", "content": context_prompt}
                        ]
                    )
                    
                    additional_info = response.choices[0].message.content
                    
                    # 검색 결과에 추가 정보 포함
                    enhanced_results["additional_info"] = additional_info
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"결과 강화 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 결과 반환
            return location_results

    # 상태 확인 엔드포인트
    @app.route('/health', methods=['GET'])
    def health_check():
        logger.debug("상태 확인 요청 수신")
        return jsonify({"status": "ok"}), 200

    # 루트 경로 처리
    @app.route('/', methods=['GET'])
    def root():
        logger.debug("루트 경로 요청 수신")
        return "네이버 지도 검색 API 서버가 실행 중입니다. /api/search 엔드포인트를 통해 요청하세요."

    # favicon 요청 처리
    @app.route('/favicon.ico')
    def favicon():
        return "", 204

    # 길찾기 API 엔드포인트 추가
    @app.route('/api/directions', methods=['GET'])
    def get_directions():
        start = request.args.get('start')  # 출발지 좌표 (경도,위도)
        goal = request.args.get('goal')    # 도착지 좌표 (경도,위도)
        waypoints = request.args.get('waypoints', '')  # 경유지 (선택사항)
        option = request.args.get('option', 'trafast')  # 옵션: trafast(최속), tracomfort(편안)
        
        if not start or not goal:
            return jsonify({"error": "출발지와 도착지 좌표가 필요합니다"}), 400
        
        try:
            url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
            headers = {
                "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
            }
            params = {
                "start": start,
                "goal": goal,
                "option": option
            }
            
            if waypoints:
                params["waypoints"] = waypoints
                
            response = requests.get(url, params=params, headers=headers)
            result = response.json()
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 주소를 좌표로 변환하는 엔드포인트 (기존 geocode 기능 활용)
    @app.route('/api/geocode', methods=['GET'])
    def geocode_address():
        address = request.args.get('address')
        
        if not address:
            return jsonify({"error": "주소가 필요합니다"}), 400
        
        try:
            # geocoding API 직접 호출
            location_data = geocode_location(address)
            if location_data.get('addresses') and len(location_data['addresses']) > 0:
                first_address = location_data['addresses'][0]
                coordinates = [float(first_address['x']), float(first_address['y'])]
                return jsonify({"coordinates": coordinates})
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 길찾기 데이터를 가져오는 헬퍼 함수
    def get_directions_data(start, goal, option='trafast'):
        """
        네이버 Direction API를 사용하여 길찾기 정보를 가져옵니다.
        """
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
        }
        params = {
            "start": start,
            "goal": goal,
            "option": option
        }
        
        try:
            logger.debug(f"Direction API 요청: {url}, 파라미터: {params}")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            results = response.json()
            
            # 네이버 API 응답 오류 처리
            if results.get('code') != 0:
                error_msg = results.get('message', '알 수 없는 오류')
                logger.error(f"Direction API 오류: {error_msg}")
                return {
                    "error": f"길찾기 오류: {error_msg}",
                    "code": results.get('code')
                }
            
            logger.debug(f"Direction API 응답: {results}")
            return results
        except Exception as e:
            logger.error(f"Direction API 요청 중 오류 발생: {str(e)}")
            return {"error": f"길찾기 중 오류가 발생했습니다: {str(e)}"}

    # search_similar_documents 호출 전 디버그 로깅 추가
    def debug_search(query, facility_type, top_k=3):
        logger.debug(f"""
        RAG 검색 시도:
        - 쿼리: {query}
        - 시설 유형: {facility_type}
        - top_k: {top_k}
        """)
        try:
            results = rag.search_similar_documents(query, facility_type, top_k)
            logger.debug(f"RAG 검색 결과: {json.dumps(results, ensure_ascii=False, indent=2)}")
            return results
        except Exception as e:
            logger.error(f"RAG 검색 중 오류 발생: {str(e)}\n{traceback.format_exc()}")
            return None

    def analyze_facility_type(query: str) -> tuple:
        """
        사용자 쿼리에서 시설 유형을 분석합니다.
        Returns: (facility_type, confidence_score, matched_keyword)
        """
        facility_keywords = {
            '병원': {
                'high_confidence': ['병원', '의원', '치과', '한의원'],
                'medium_confidence': ['진료', '진찰', '의료', '검진'],
                'context': ['아파서', '아픈데', '아프다', '치료', '진단', '가까운']
            },
            '음식점': {
                'high_confidence': ['음식점', '식당', '레스토랑', '맛집'],
                'medium_confidence': ['밥집', '식사'],
                'context': ['배고파', '먹을', '먹고', '식사', '가까운']
            },
            '카페': {
                'high_confidence': ['카페', '커피숍', '커피집'],
                'medium_confidence': ['커피', '디저트', '베이커리'],
                'context': ['마시고', '달달한', '차', '가까운']
            },
            '약국': {
                'high_confidence': ['약국', '약방', '드럭스토어'],
                'medium_confidence': ['약', '처방', '조제'],
                'context': ['약이', '약을', '처방전', '가까운']
            }
        }
        
        logger.info("\n=== 시설 유형 분석 시작 ===")
        logger.info(f"분석할 쿼리: '{query}'")
        
        best_match = None
        highest_score = 0
        matched_keyword = None
        
        for facility_type, keywords in facility_keywords.items():
            current_score = 0
            current_keyword = None
            
            # 높은 신뢰도 키워드 체크 (가중치: 1.0)
            for keyword in keywords['high_confidence']:
                if keyword in query:
                    current_score = 1.0
                    current_keyword = keyword
                    logger.info(f"- 높은 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                    break
            
            # 중간 신뢰도 키워드 체크 (가중치: 0.7)
            if current_score == 0:
                for keyword in keywords['medium_confidence']:
                    if keyword in query:
                        current_score = 0.7
                        current_keyword = keyword
                        logger.info(f"- 중간 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            # 문맥 키워드 체크 (가중치: 0.3)
            if current_score == 0:
                for keyword in keywords['context']:
                    if keyword in query:
                        current_score = 0.3
                        current_keyword = keyword
                        logger.info(f"- 문맥 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            if current_score > highest_score:
                highest_score = current_score
                best_match = facility_type
                matched_keyword = current_keyword
        
        logger.info(f"=== 분석 결과 ===")
        logger.info(f"- 시설 유형: {best_match or '없음'}")
        logger.info(f"- 신뢰도 점수: {highest_score}")
        logger.info(f"- 매칭된 키워드: {matched_keyword or '없음'}")
        
        return best_match, highest_score, matched_keyword

    def analyze_place_type(query: str) -> str:
        """
        사용자 쿼리에서 시설 유형을 분석합니다.
        Returns: '병원', '음식점', '카페', '약국', '없음' 중 하나
        """
        facility_keywords = {
            '병원': {
                'high_confidence': ['병원', '의원', '치과', '한의원'],
                'medium_confidence': ['진료', '진찰', '의료', '검진'],
                'context': ['아파서', '아픈데', '아프다', '치료', '진단', '가까운']
            },
            '음식점': {
                'high_confidence': ['음식점', '식당', '레스토랑', '맛집'],
                'medium_confidence': ['밥집', '식사'],
                'context': ['배고파', '먹을', '먹고', '식사', '가까운']
            },
            '카페': {
                'high_confidence': ['카페', '커피숍', '커피집'],
                'medium_confidence': ['커피', '디저트', '베이커리'],
                'context': ['마시고', '달달한', '차', '가까운']
            },
            '약국': {
                'high_confidence': ['약국', '약방', '드럭스토어'],
                'medium_confidence': ['약', '처방', '조제'],
                'context': ['약이', '약을', '처방전', '가까운']
            }
        }
        
        logger.info("\n=== 시설 유형 분석 시작 ===")
        logger.info(f"분석할 쿼리: '{query}'")
        
        best_match = None
        highest_score = 0
        
        for facility_type, keywords in facility_keywords.items():
            current_score = 0
            
            # 높은 신뢰도 키워드 체크 (가중치: 1.0)
            for keyword in keywords['high_confidence']:
                if keyword in query:
                    current_score = 1.0
                    logger.info(f"- 높은 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                    break
            
            # 중간 신뢰도 키워드 체크 (가중치: 0.7)
            if current_score == 0:
                for keyword in keywords['medium_confidence']:
                    if keyword in query:
                        current_score = 0.7
                        logger.info(f"- 중간 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            # 문맥 키워드 체크 (가중치: 0.3)
            if current_score == 0:
                for keyword in keywords['context']:
                    if keyword in query:
                        current_score = 0.3
                        logger.info(f"- 문맥 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            if current_score > highest_score:
                highest_score = current_score
                best_match = facility_type
        
        logger.info(f"=== 분석 결과 ===")
        logger.info(f"- 시설 유형: {best_match or '없음'}")
        logger.info(f"- 신뢰도 점수: {highest_score}")
        
        return best_match or '없음'

    def is_sinsaok_query(user_query):
        # 신사옥 관련 키워드만 허용
        allowed_keywords = ["신사옥", "본사", "회사", "F&F"]
        return any(keyword in user_query for keyword in allowed_keywords)

    def handle_user_query(user_query):
        """사용자 쿼리를 처리하는 함수 (현재 미사용)"""
        # 1. 위치 관련 키워드 추출 (예: "강남역", "홍대" 등)
        location = extract_location(user_query)
        
        # 2. 신사옥 관련이 아니면 안내
        if location and not is_sinsaok_query(location):
            return "이 챗봇은 F&F 신사옥 주변 편의시설만 안내해드립니다."
        
        # 3. 신사옥 기준으로 검색 (실제 구현은 search_api에서 처리)
        place_type = extract_type(user_query)
        return f"{place_type} 검색 요청이 접수되었습니다."

    def extract_ordinal(query):
        # 숫자 패턴 먼저 확인 (1번, 2번, 3번 등)
        number_match = re.search(r'(\d+)\s*번', query)
        if number_match:
            return int(number_match.group(1))
        
        # 한글 순서 패턴 확인 (첫번째, 두번째 등)
        korean_match = re.search(r'(첫|두|세|네|다섯|여섯|일|이|삼|사|오|육|칠|팔|구|십)\s*(번째|번)?', query)
        if korean_match:
            word = korean_match.group(1)
            mapping = {
                "첫": 1, "일": 1,
                "두": 2, "이": 2,
                "세": 3, "삼": 3,
                "네": 4, "사": 4,
                "다섯": 5, "오": 5,
                "여섯": 6, "육": 6,
            }
            return mapping.get(word, None)
        
        return None

    def extract_location(query):
        """쿼리에서 위치 정보를 추출하는 함수"""
        # 간단한 위치 키워드 추출 (필요에 따라 확장 가능)
        location_keywords = ["강남역", "홍대", "잠실", "신촌", "명동", "이태원", "압구정", "청담", "역삼", "삼성"]
        for keyword in location_keywords:
            if keyword in query:
                return keyword
        return None

    def extract_type(query):
        """쿼리에서 시설 유형을 추출하는 함수"""
        return analyze_place_type(query)

    if __name__ == '__main__':
        try:
            logger.info("백엔드 서버 시작: http://localhost:5000")
            app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        except Exception as e:
            logger.error(f"서버 시작 중 오류 발생: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
except Exception as e:
    logger.error(f"백엔드 초기화 중 오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 