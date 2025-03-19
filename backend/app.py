from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        raw_data = request.data.decode('utf-8', 'ignore')
        data = json.loads(raw_data)
        print(f"Received data: {data}")

        if data.get('message') == '점심 먹을만한 곳 추천해줘':
            response = [
                {"name": "서브웨이", "distance": "도보 2분", "type": "샌드위치"},
                {"name": "샐러디", "distance": "도보 3분", "type": "샐러드"},
                {"name": "김밥천국", "distance": "도보 3분", "type": "한식"}
            ]
            return app.response_class(
                response=json.dumps(response, ensure_ascii=False), 
                mimetype='application/json'
            )

        return app.response_class(
            response=json.dumps({'response': f"입력한 메시지: {data.get('message')}"}, ensure_ascii=False),
            mimetype='application/json'
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
