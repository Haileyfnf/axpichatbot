import React from 'react';
import './DirectionsResult.css';

const DirectionsResult = ({ directions, onClose }) => {
  if (!directions || !directions.route) {
    return null;
  }

  // 첫 번째 경로 정보 가져오기
  const optimalRoute = directions.route.traoptimal?.[0];
  if (!optimalRoute) return null;

  const { summary } = optimalRoute;
  
  // 소요 시간을 시/분/초로 변환
  const formatDuration = (milliseconds) => {
    if (!milliseconds) return '정보 없음';
    
    const seconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    return hours > 0 
      ? `${hours}시간 ${minutes}분` 
      : `${minutes}분`;
  };
  
  // 거리를 km/m 단위로 변환
  const formatDistance = (meters) => {
    if (!meters) return '정보 없음';
    
    return meters >= 1000 
      ? `${(meters / 1000).toFixed(1)}km` 
      : `${meters}m`;
  };

  return (
    <div className="directions-result">
      <div className="directions-header">
        <h3>경로 안내</h3>
        <button className="close-button" onClick={onClose}>✖</button>
      </div>
      
      <div className="directions-summary">
        <div className="summary-item">
          <span className="label">출발</span>
          <span className="value">{directions.summary?.start_name || '출발지'}</span>
        </div>
        <div className="summary-item">
          <span className="label">도착</span>
          <span className="value">{directions.summary?.end_name || '도착지'}</span>
        </div>
        <div className="summary-item">
          <span className="label">총 거리</span>
          <span className="value">{formatDistance(summary.distance)}</span>
        </div>
        <div className="summary-item">
          <span className="label">예상 시간</span>
          <span className="value">{formatDuration(summary.duration)}</span>
        </div>
        {summary.tollFare > 0 && (
          <div className="summary-item">
            <span className="label">통행료</span>
            <span className="value">{summary.tollFare.toLocaleString()}원</span>
          </div>
        )}
        {summary.taxiFare && (
          <div className="summary-item">
            <span className="label">예상 택시요금</span>
            <span className="value">{summary.taxiFare.toLocaleString()}원</span>
          </div>
        )}
      </div>
      
      {optimalRoute.guide && optimalRoute.guide.length > 0 ? (
        <div className="directions-guide">
          <h4>상세 안내</h4>
          <ul>
            {optimalRoute.guide.map((step, index) => (
              <li key={index} className="guide-step">
                <div className="step-instruction">{step.instructions}</div>
                <div className="step-distance">{formatDistance(step.distance)}</div>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="directions-guide">
          <p>상세 경로 안내가 제공되지 않습니다.</p>
        </div>
      )}
    </div>
  );
};

export default DirectionsResult; 