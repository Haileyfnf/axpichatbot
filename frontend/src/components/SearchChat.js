import React, { useState, useRef, useEffect } from 'react';
import './SearchChat.css';

const Message = ({ type, content }) => {
  return (
    <div className={`message ${type}`}>
      {content}
    </div>
  );
};

const SearchChat = ({ 
  onSearch, 
  messages, 
  isLoading = false, 
  isBackendOnline = true,
  hasLocation = false
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  // 새 메시지가 추가될 때마다 스크롤을 아래로 이동
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && isBackendOnline) {
      onSearch(input.trim());
      setInput('');
    }
  };

  // 위치 기반 검색 추천어
  const locationSuggestions = [
    "내 주변 맛집 추천해줘",
    "가까운 병원 어디 있어?",
    "근처 카페 알려줘",
    "내 위치에서 강남역까지 가는 길"
  ];

  // 일반 검색 추천어
  const generalSuggestions = [
    "서울 강남역",
    "부산 해운대 맛집",
    "서울역에서 명동까지 가는 길",
    "강남역에서 종로3가역까지 어떻게 가나요"
  ];

  // 추천 검색어 선택
  const handleSuggestionClick = (suggestion) => {
    if (!isLoading && isBackendOnline) {
      onSearch(suggestion);
    }
  };

  return (
    <div className="search-chat">
      <div className="chat-header">
        <h2>장소 검색</h2>
        <div className="status-indicators">
          {isBackendOnline ? null : (
            <div className="backend-status">오프라인</div>
          )}
        </div>
      </div>
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>신사옥 주변 편의시설을 검색해보세요.</p>
            <p>예: "병원", "약국", "음식점", "카페" 등</p>
            
            <div className="suggestions-container">
              <p className="suggestions-title">
                {hasLocation ? "위치 기반 검색 예시:" : "검색 예시:"}
              </p>
              <div className="suggestions-list">
                {(hasLocation ? locationSuggestions : generalSuggestions).map((suggestion, index) => (
                  <button 
                    key={index} 
                    className="suggestion-button"
                    onClick={() => handleSuggestionClick(suggestion)}
                    disabled={!isBackendOnline || isLoading}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
            
            {!isBackendOnline && (
              <p className="backend-warning">
                ⚠️ 백엔드 서버에 연결할 수 없습니다. 검색 기능이 작동하지 않을 수 있습니다.
              </p>
            )}
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, index) => (
              <Message 
                key={index} 
                type={msg.type} 
                content={msg.content} 
              />
            ))}
            {isLoading && (
              <Message 
                type="system" 
                content="검색 중..." 
              />
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            !isBackendOnline ? "서버에 연결할 수 없습니다..." :
            !hasLocation ? "위치 정보 권한이 필요합니다..." :
            "무엇을 도와드릴까요?"
          }
          disabled={!isBackendOnline || !hasLocation}
        />
        <button type="submit" disabled={!isBackendOnline || !hasLocation || !input.trim()}>
          검색
        </button>
      </form>
    </div>
  );
};

export default SearchChat; 