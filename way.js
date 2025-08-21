import React, { useEffect, useRef, useState, useCallback } from 'react';

const Map = ({ start, end, directions }) => {
    const mapRef = useRef(null);
    const [startPoint, setStartPoint] = useState(null);
    const [endPoint, setEndPoint] = useState(null);

    useEffect(() => {
        if (window.naver && window.naver.maps) {
            console.log('✅ 네이버 지도 API 로드 성공');

            // ✅ 지도 생성
            const map = new window.naver.maps.Map(mapRef.current, {
                center: new window.naver.maps.LatLng(37.5665, 126.9780),
                zoom: 14,
            });

            // ✅ 출발지 마커 표시
            if (start) {
                const [lng, lat] = start.split(',');
                new window.naver.maps.Marker({
                    position: new window.naver.maps.LatLng(lat, lng),
                    map,
                    title: '출발지'
                });
            }

            // ✅ 도착지 마커 표시
            if (end) {
                const [lng, lat] = end.split(',');
                new window.naver.maps.Marker({
                    position: new window.naver.maps.LatLng(lat, lng),
                    map,
                    title: '도착지'
                });
            }

            // ✅ 경로 표시
            if (directions?.path) {
                const path = directions.path.map(
                    ([lng, lat]) => new window.naver.maps.LatLng(lat, lng)
                );

                new window.naver.maps.Polyline({
                    path: path,
                    strokeColor: '#FF0000',
                    strokeWeight: 4,
                    strokeOpacity: 0.8,
                    map: map,
                });

                map.setCenter(path[0]); // 경로 시작점으로 이동
            }
        }
    }, [start, end, directions]);

    const handleDirectionsSearch = useCallback(async (start, end) => {
        try {
            setStartPoint(start);
            setEndPoint(end);
            
            // 기존 경로 초기화
            clearDirections();
            
            // 길찾기 API 호출 (백엔드로 요청)
            const response = await fetch('/api/directions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    start: start,
                    end: end
                })
            });

            if (!response.ok) {
                throw new Error('길찾기 검색에 실패했습니다.');
            }

            const directionsData = await response.json();
            setDirections(directionsData);
            setShowDirectionsPanel(true);
            displayDirections(directionsData);
        } catch (error) {
            console.error('길찾기 검색 중 오류:', error);
        }
    }, [clearDirections]);

    const displayDirections = useCallback((result) => {
        if (!mapRef.current || !result.directions || !result.directions.route) return;
        
        const { directions, start_location, end_location } = result;
        
        // 경로가 있는지 확인
        if (directions.route && directions.route.traoptimal && directions.route.traoptimal[0]) {
            const pathCoordinates = directions.route.traoptimal[0].path;
            
            // 기존 경로 제거
            clearDirections();
            
            // 새 경로 그리기 (way.js 스타일로 수정)
            const path = pathCoordinates.map(coord => 
                new window.naver.maps.LatLng(coord[1], coord[0]));
            const polyline = new window.naver.maps.Polyline({
                map: mapRef.current,
                path: path,
                strokeColor: '#5347AA', // 스타일 유지
                strokeWeight: 5,
                strokeOpacity: 0.8
            });
            
            setDirectionsPolyline(polyline);
            
            // 출발지와 도착지 마커 추가
            const newMarkers = [];
            
            // 출발지 마커
            const startMarker = new window.naver.maps.Marker({
                position: new window.naver.maps.LatLng(start_location.y, start_location.x),
                map: mapRef.current,
                icon: {
                    content: '<div class="start-location-marker">출발</div>',
                    anchor: new window.naver.maps.Point(15, 30)
                },
                title: '출발지'
            });
            newMarkers.push(startMarker);
            
            // 도착지 마커
            const endMarker = new window.naver.maps.Marker({
                position: new window.naver.maps.LatLng(end_location.y, end_location.x),
                map: mapRef.current,
                icon: {
                    content: '<div class="end-location-marker">도착</div>',
                    anchor: new window.naver.maps.Point(15, 30)
                },
                title: '도착지'
            });
            newMarkers.push(endMarker);
            
            setDirectionsMarkers(newMarkers);
            
            // 경로 전체가 보이도록 지도 영역 조정
            const bounds = new window.naver.maps.LatLngBounds();
            path.forEach(coord => bounds.extend(coord));
            mapRef.current.fitBounds(bounds);

            // way.js 스타일의 거리/시간 정보 표시 추가
            if (directions.summary) {
                // 여기에 거리와 시간 정보 표시 로직 추가
                // DirectionsResult 컴포넌트에 전달하거나 UI에 직접 표시
            }
        }
    }, [clearDirections]);

    return (
        <div>
            <div
                ref={mapRef}
                style={{
                    width: '100%',
                    height: '400px',
                    border: '1px solid #ddd',
                    borderRadius: '8px',
                    marginTop: '20px',
                }}
            />

            {/* ✅ 거리 및 시간 정보 출력 */}
            {directions?.summary && (
                <div style={{ marginTop: '10px' }}>
                    🚗 거리: {(directions.summary.distance / 1000).toFixed(2)} km
                    | 🕒 소요 시간: {Math.round(directions.summary.duration / 1000 / 60)} 분
                </div>
            )}
        </div>
    );
};

export default Map;