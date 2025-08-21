# 네이버 지도 검색 애플리케이션

이 프로젝트는 Naver Maps API를 활용하여 검색어를 통해 장소를 검색하고 지도에 표시하는 웹 애플리케이션입니다.

## 주요 기능

- 채팅창 형태의 검색 인터페이스
- Naver Maps API를 통한 위치 검색
- 검색 결과를 지도에 마커로 표시
- 검색 결과에 대한 정보 윈도우 제공

## 기술 스택

- **프론트엔드**: React.js
- **백엔드**: Flask (Python)
- **API**: Naver Maps API

## 설치 및 실행 방법

### 필수 조건

- Node.js 및 npm
- Python 3.7 이상
- 필요한 패키지: requests, flask, flask-cors

### 빠른 실행 방법 (한 번에 실행)

아래 명령어로 백엔드와 프론트엔드를 동시에 실행할 수 있습니다:

```bash
# 필요한 패키지 설치 (최초 1회만)
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 애플리케이션 실행 (백엔드 + 프론트엔드)
python run.py
```

### 각각 실행하는 방법 (기존 방식)

#### 백엔드 설치 및 실행

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 백엔드 서버 실행
python app.py
```

#### 프론트엔드 설치 및 실행

```bash
# frontend 디렉토리로 이동
cd frontend

# 필요한 패키지 설치
npm install

# 프론트엔드 서버 실행
npm start
```

## 사용 방법

1. 애플리케이션이 실행되면 웹 브라우저가 자동으로 열립니다 (http://localhost:3000).
2. 오른쪽 채팅창에 검색하고 싶은 장소의 이름을 입력합니다.
3. 검색 결과가 지도에 표시됩니다.
4. 종료하려면 커맨드 창에서 Ctrl+C를 누르세요.

## API 키 설정

이 프로젝트는 네이버 클라우드 플랫폼 API를 사용합니다:

- Client ID: 07gq168p3y
- Client Secret: ofunz4XdLSqOLnYW0pwKJP3gLaTZOsGhrKjuP1ns 