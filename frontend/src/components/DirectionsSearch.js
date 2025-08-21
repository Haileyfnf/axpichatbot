import React, { useState } from 'react';
import './DirectionsSearch.css';

const DirectionsSearch = ({ onSearch, userLocation }) => {
  const [startLocation, setStartLocation] = useState('');
  const [endLocation, setEndLocation] = useState('');
  const [useCurrentLocation, setUseCurrentLocation] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    let start = startLocation;
    // 현재 위치 사용 옵션이 켜져 있으면 사용자 위치 사용
    if (useCurrentLocation && userLocation) {
      start = `${userLocation.lng},${userLocation.lat}`;
    }
    
    // 주소 -> 좌표 변환이 필요한 경우
    if (start && !start.includes(',') && !useCurrentLocation) {
      try {
        const response = await fetch(`/api/geocode?address=${encodeURIComponent(start)}`);
        const data = await response.json();
        if (data.coordinates) {
          start = `${data.coordinates[0]},${data.coordinates[1]}`;
        } else if (data.addresses && data.addresses.length > 0) {
          // 기존 API 응답 구조에 맞게 처리
          const address = data.addresses[0];
          start = `${address.x},${address.y}`;
        }
      } catch (error) {
        console.error('출발지 좌표 변환 실패:', error);
      }
    }
    
    // 도착지 주소 -> 좌표 변환
    let end = endLocation;
    if (end && !end.includes(',')) {
      try {
        const response = await fetch(`/api/geocode?address=${encodeURIComponent(end)}`);
        const data = await response.json();
        if (data.coordinates) {
          end = `${data.coordinates[0]},${data.coordinates[1]}`;
        } else if (data.addresses && data.addresses.length > 0) {
          // 기존 API 응답 구조에 맞게 처리
          const address = data.addresses[0];
          end = `${address.x},${address.y}`;
        }
      } catch (error) {
        console.error('도착지 좌표 변환 실패:', error);
      }
    }
    
    if (start && end) {
      onSearch(start, end);
    } else {
      alert('출발지와 도착지를 모두 입력해주세요.');
    }
  };

  return (
    <div className="directions-search">
      <h3>길찾기</h3>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={useCurrentLocation}
              onChange={() => setUseCurrentLocation(!useCurrentLocation)}
            />
            현재 위치에서 출발
          </label>
        </div>
        
        {!useCurrentLocation && (
          <div className="form-group">
            <input
              type="text"
              placeholder="출발지"
              value={startLocation}
              onChange={(e) => setStartLocation(e.target.value)}
              disabled={useCurrentLocation}
              required={!useCurrentLocation}
            />
          </div>
        )}
        
        <div className="form-group">
          <input
            type="text"
            placeholder="도착지"
            value={endLocation}
            onChange={(e) => setEndLocation(e.target.value)}
            required
          />
        </div>
        
        <button type="submit" className="search-button">
          길찾기
        </button>
      </form>
    </div>
  );
};

export default DirectionsSearch; 