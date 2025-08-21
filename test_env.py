import os
from dotenv import load_dotenv

# 절대 경로로 .env 파일 로드, 기존 환경변수 덮어쓰기
load_dotenv(dotenv_path="C:/Users/haenee/Desktop/naver-map-chat3/.env", override=True)

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))