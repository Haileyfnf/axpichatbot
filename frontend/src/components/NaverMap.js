import React, { useEffect, useRef, useState } from 'react';
import './NaverMap.css';
import './DirectionsSearch.css';
import './DirectionsResult.css';

const NaverMap = ({ searchResult, userLocation }) => {
  const mapRef = useRef(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);
  const markersRef = useRef([]);  // 마커들을 저장할 ref 추가

  // 기존 마커들을 제거하는 함수
  const clearMarkers = () => {
    markersRef.current.forEach(marker => {
      marker.setMap(null);  // 지도에서 마커 제거
    });
    markersRef.current = [];  // 마커 배열 초기화
  };

  // 네이버 지도 API 로드 확인
  useEffect(() => {
    const checkMapLoad = () => {
      if (window.naver && window.naver.maps) {
        setIsMapLoaded(true);
      } else {
        setTimeout(checkMapLoad, 100);
      }
    };
    checkMapLoad();
  }, []);

  // 지도 초기화
  useEffect(() => {
    if (!isMapLoaded || !mapRef.current || !window.naver || !window.naver.maps) return;

    try {
      console.log('✅ 네이버 지도 API 로드 성공');

      // ✅ 지도 생성
      const mapOptions = {
        center: new window.naver.maps.LatLng(37.4982517, 127.0310195), // F&F 신사옥 위치
        zoom: 17,
        mapTypeControl: false,
        mapTypeId: window.naver.maps.MapTypeId.NORMAL,
        scaleControl: true,
        logoControl: true,
        mapDataControl: false,
        zoomControl: true,
      };

      const map = new window.naver.maps.Map(mapRef.current, mapOptions);

      // ✅ 사용자 위치 마커 표시 (항상 표시)
      if (userLocation) {
        new window.naver.maps.Marker({
          position: new window.naver.maps.LatLng(userLocation.lat, userLocation.lng),
          map: map,
          icon: {
            content: `
              <div class="user-location-marker">
                <div class="pulse"></div>
                <div class="marker-dot"></div>
              </div>
            `,
            anchor: new window.naver.maps.Point(15, 15)
          },
          title: '현재 위치'
        });
      }

      // 검색 결과가 있을 때 지도 업데이트
      if (searchResult) {
        // 기존 마커들 제거
        clearMarkers();

        // 길찾기 결과 처리
        if (searchResult.type === 'directions' && searchResult.start && searchResult.end) {
          // 출발지와 도착지 좌표 설정
          const startPosition = new window.naver.maps.LatLng(
            searchResult.start.y,
            searchResult.start.x
          );
          const endPosition = new window.naver.maps.LatLng(
            searchResult.end.y,
            searchResult.end.x
          );

          // 두 지점의 중간 지점을 계산하여 center로 설정
          const center = new window.naver.maps.LatLng(
            (parseFloat(searchResult.start.y) + parseFloat(searchResult.end.y)) / 2,
            (parseFloat(searchResult.start.x) + parseFloat(searchResult.end.x)) / 2
          );
          
          // 지도 중심과 줌 레벨 설정
          map.setCenter(center);
          map.setZoom(17);

          // 출발지 마커 (F&F 신사옥 - 초록색)
          const startMarker = new window.naver.maps.Marker({
            position: startPosition,
            map,
            icon: {
              content: `
                <div style="position: relative;">
                  <div style="
                    width: 24px; 
                    height: 24px; 
                    background: #00B700; 
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    border: 2px solid white;
                    box-shadow: 0 2px 4px #00B700;
                  ">
                  </div>
                  <div style="
                    position: absolute;
                    white-space: nowrap;
                    top: -25px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: white;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #00B700;
                    box-shadow: 0 2px 4px #00B700;
                  ">
                    F&F 신사옥
                  </div>
                </div>
              `,
              anchor: new window.naver.maps.Point(12, 12)
            }
          });

          // 도착지 마커 (빨간색)
          const endMarker = new window.naver.maps.Marker({
            position: endPosition,
            map,
            icon: {
              content: `
                <div style="position: relative;">
                  <div style="
                    width: 24px; 
                    height: 24px; 
                    background: #FF0000; 
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    border: 2px solid white;
                    box-shadow: 0 2px 4px #FF0000;
                  ">
                  </div>
                  <div style="
                    position: absolute;
                    white-space: nowrap;
                    top: -25px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: white;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #FF0000;
                    box-shadow: 0 2px 4px #FF0000;
                  ">
                    ${searchResult.end.name}
                  </div>
                </div>
              `,
              anchor: new window.naver.maps.Point(12, 12)
            }
          });

          // 마커들을 배열에 저장
          markersRef.current.push(startMarker, endMarker);
        }
        // 일반 장소 검색 결과 처리
        else if (searchResult.type === 'places' && searchResult.places) {
          const bounds = new window.naver.maps.LatLngBounds();
          
          searchResult.places.forEach(place => {
            const position = new window.naver.maps.LatLng(place.y, place.x);
            bounds.extend(position);
            
            new window.naver.maps.Marker({
              position: position,
              map: map,
              icon: {
                content: `<div class="marker place">${place.name}</div>`,
                anchor: new window.naver.maps.Point(15, 31)
              }
            });
          });
          
          map.fitBounds(bounds);
        }
      }
    } catch (error) {
      console.error('지도 초기화 중 오류 발생:', error);
    }
  }, [isMapLoaded, searchResult, userLocation]);

  return (
    <div className="map-container" style={{ height: '100vh' }}>
      <div id="map" ref={mapRef} className="naver-map">
        {!isMapLoaded ? (
          <div className="map-overlay">
            <div className="map-instructions">
              <h2>지도 로딩 중...</h2>
              <p>잠시만 기다려주세요.</p>
            </div>
          </div>
        ) : !searchResult && !userLocation && (
          <div className="map-overlay">
            <div className="map-instructions">
              <h2>네이버 지도 검색</h2>
              <p>오른쪽 채팅창에 장소 이름이나 주소를 입력하세요</p>
              <p>위치 기반 검색을 위해 위치 정보 공유를 허용해주세요</p>
            </div>
          </div>
        )}
      </div>
      
      {/* ✅ 길찾기 결과일 경우 거리 및 시간 정보 출력 */}
      {searchResult?.type === 'directions' && searchResult.summary && (
        <div className="directions-info">
          <p>
            🚶‍♂️ 거리: {searchResult.summary.distance}m
            | 🕒 소요 시간: {searchResult.summary.duration_minutes} 분
          </p>
        </div>
      )}
    </div>
  );
};

export default NaverMap;