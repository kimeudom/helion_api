from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
import geopandas as gpd  # type: ignore
import pandas as pd
import numpy as np  # type: ignore
import joblib  # type: ignore
from shapely.geometry import box, mapping  # type: ignore
from ...core.config import Settings
import os

router = APIRouter(tags=["v1"])
settings = Settings()


class PredictRequest(BaseModel):
    bbox: List[float] = Field(...,
                              description="[min_lon, min_lat, max_lon, max_lat]")
    budget: float = Field(..., ge=0)
    preferences: Optional[Dict[str, Any]] = None


class PredictItem(BaseModel):
    id: Any
    geometry: Dict[str, Any]
    socio_need: float
    predicted_cost: float
    impact_per_dollar: float
    score: float
    extra: Dict[str, Any] = {}


class PredictResponse(BaseModel):
    results: List[PredictItem]
    meta: Dict[str, Any]


# Lazy-loaded dataset and models
_gdf: Optional[gpd.GeoDataFrame] = None
_model1 = None
_model3a = None


def _load_data():
    global _gdf
    if _gdf is None:
        path = settings.data_geojson
        if not os.path.exists(path):
            raise RuntimeError(f"GeoJSON data not found at {path}")
        _gdf = gpd.read_file(path)
    return _gdf


def _load_models():
    global _model1, _model3a
    if _model1 is None and settings.model1_path:
        _model1 = joblib.load(settings.model1_path)
    if _model3a is None and settings.model3a_path:
        _model3a = joblib.load(settings.model3a_path)
    return _model1, _model3a


def _filter_gdf_by_bbox(gdf: gpd.GeoDataFrame, bbox: List[float]) -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    return gdf[gdf.geometry.intersects(bbox_geom)].copy()


@router.post("/v1/predict", response_model=PredictResponse, summary="Predict impacts for candidate areas")
async def predict(req: PredictRequest):
    gdf = _load_data()
    _model1, _model3a = _load_models()

    # validate bbox
    if len(req.bbox) != 4:
        raise HTTPException(
            status_code=400, detail="bbox must be [min_lon, min_lat, max_lon, max_lat]")

    filtered = _filter_gdf_by_bbox(gdf, req.bbox)

    if filtered.empty:
        return PredictResponse(results=[], meta={"count": 0})

    # derive feature dataframe for models
    df = pd.DataFrame(filtered.drop(columns='geometry').copy())

    # mock/model predictions if models are not provided
    if _model1 is None:
        df['socio_need'] = np.random.rand(len(df)) * 100
    else:
        # expect model1.predict or similar
        df['socio_need'] = _model1.predict(df)

    if _model3a is None:
        df['predicted_cost'] = np.random.rand(len(df)) * (req.budget * 1.5)
    else:
        df['predicted_cost'] = _model3a.predict(df)

    # compute impact per dollar
    df['impact_per_dollar'] = df['socio_need'] / (df['predicted_cost'] + 1e-6)

    # score: combine impact_per_dollar and closeness to budget
    df['budget_diff'] = (df['predicted_cost'] - req.budget).abs()
    df['score'] = df['impact_per_dollar'] / \
        (1 + df['budget_diff'] / (req.budget + 1e-6))

    df = df.sort_values('score', ascending=False)

    results = []
    for idx, row in df.iterrows():
        geom = mapping(filtered.loc[idx, 'geometry']
                       ) if 'geometry' in filtered.columns else None
        results.append(
            PredictItem(
                id=row.get('id', idx),
                geometry=geom,
                socio_need=float(row['socio_need']),
                predicted_cost=float(row['predicted_cost']),
                impact_per_dollar=float(row['impact_per_dollar']),
                score=float(row['score']),
                extra={},
            )
        )

    return PredictResponse(results=results, meta={"count": len(results)})
