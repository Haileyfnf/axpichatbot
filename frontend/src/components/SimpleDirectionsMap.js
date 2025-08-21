import React, { useEffect, useRef } from 'react';

// 단순한 경로 표시를 위한 지도 컴포넌트
const SimpleDirectionsMap = ({ start, end, directions }) => {
    const mapRef = useRef(null);

    useEffect(() => {
        if (window.naver && window.naver.maps) {
            console.log('✅ 네이버 지도 API 로드 성공');

            // 지도 생성
            const map = new window.naver.maps.Map(mapRef.current, {
                center: new window.naver.maps.LatLng(37.5665, 126.9780),
                zoom: 14,
            });

            // 출발지 마커 표시
            if (start) {
                const [lng, lat] = start.split(',');
                new window.naver.maps.Marker({
                    position: new window.naver.maps.LatLng(lat, lng),
                    map,
                    title: '출발지'
                });
            }

            // 도착지 마커 표시
            if (end) {
                const [lng, lat] = end.split(',');
                new window.naver.maps.Marker({
                    position: new window.naver.maps.LatLng(lat, lng),
                    map,
                    title: '도착지'
                });
            }

            // 경로 표시
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
            
            // 네이버 API 경로 포맷 처리
            if (directions?.route?.traoptimal && directions.route.traoptimal[0]) {
                const pathCoordinates = directions.route.traoptimal[0].path;
                const path = pathCoordinates.map(coord => 
                    new window.naver.maps.LatLng(coord[1], coord[0]));
                
                // 경로 그리기
                new window.naver.maps.Polyline({
                    path: path,
                    strokeColor: '#5347AA',
                    strokeWeight: 5,
                    strokeOpacity: 0.8,
                    map: map,
                });
                
                // 경로 전체가 보이도록 지도 영역 조정
                const bounds = new window.naver.maps.LatLngBounds();
                path.forEach(coord => bounds.extend(coord));
                map.fitBounds(bounds);
            }
        }
    }, [start, end, directions]);

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

            {/* 거리 및 시간 정보 출력 */}
            {directions?.summary && (
                <div style={{ marginTop: '10px' }}>
                    🚗 거리: {(directions.summary.distance / 1000).toFixed(2)} km
                    | 🕒 소요 시간: {Math.round(directions.summary.duration / 1000 / 60)} 분
                </div>
            )}
        </div>
    );
};

export default SimpleDirectionsMap; 