from rag_system import RAGSystem

def test_rag_with_data():
    # RAG 시스템 초기화
    rag = RAGSystem()
    
    # 테스트용 질문
    test_questions = [
        "맛집 추천해줘",
        "평점이 높은 식당 알려줘",
        "가성비 좋은 식당 추천해줘",
        "한식 맛집 추천해줘",
        "분위기 좋은 식당 찾아줘",
        "무료로 주차할 수 있는 식당 추천해줘"
    ]
    
    print("\n질의응답 테스트...")
    for question in test_questions:
        print(f"\n질문: {question}")
        response = rag.query(question)
        print(f"답변: {response}")
    
    # 유사 문서 검색 테스트
    print("\n유사 문서 검색 테스트...")
    similar_docs = rag.search_similar_documents("맛집 추천", limit=3)
    for doc in similar_docs:
        print(f"\n유사 문서: {doc['content'][:100]}...")
        print(f"메타데이터: {doc['metadata']}")
        print(f"유사도: {doc['distance']}")

if __name__ == "__main__":
    test_rag_with_data() 