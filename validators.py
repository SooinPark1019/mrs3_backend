import json
from typing import List
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError, field_validator

class PolygonsModel(BaseModel):
    """
    polygons: List of polygons, each is a list of [x, y] integer points.
    예시: [[[x1, y1], [x2, y2], [x3, y3]], ...]
    """
    polygons: List[List[List[int]]]  # [[[x, y], ...], ...]

    @field_validator('polygons')
    @classmethod
    def check_polygons(cls, v):
        """
        각 다각형이 꼭짓점 3개 이상인지, 각 점이 [x, y] 형식(int)인지 검증
        """
        if not isinstance(v, list) or not v:
            raise ValueError('polygons 필드는 다각형 좌표 리스트(2중 리스트)여야 함')
        for poly in v:
            if not isinstance(poly, list) or len(poly) < 3:
                raise ValueError('각 다각형은 꼭짓점 3개 이상 필요')
            for point in poly:
                if not (
                    isinstance(point, list) and
                    len(point) == 2 and
                    all(isinstance(coord, int) for coord in point)
                ):
                    raise ValueError('각 점은 [x, y] (int) 쌍이어야 함')
        return v

def parse_polygons(polygons_str: str) -> List[List[List[int]]]:
    """
    polygons_str: 프론트에서 받은 JSON.stringify된 polygons 데이터(문자열)
    - 예시: "[[[10,20],[30,40],[50,60]],[[100,200],[120,220],[130,230]]]"
    정상 구조/타입이 아니면 HTTP 422 예외 발생

    Returns:
        파싱된 polygons(3중 int 리스트)
    """
    try:
        data = json.loads(polygons_str)
        parsed = PolygonsModel(polygons=data)
        return parsed.polygons
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f'잘못된 polygons 입력값: {e}')

def validate_scaler(scaler: int) -> int:
    """
    scaler가 2, 3, 4 중 하나인지 검증 (그 외면 422)
    """
    if scaler not in [2, 3, 4]:
        raise HTTPException(status_code=422, detail='scaler는 2, 3, 4만 허용')
    return scaler
