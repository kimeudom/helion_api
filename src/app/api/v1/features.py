import json
from fastapi import APIRouter, HTTPException, Body, Request
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from shapely.geometry import shape, Polygon, MultiPolygon
from fastapi.encoders import jsonable_encoder

router = APIRouter()


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Union[List[List[List[float]]], List[List[List[List[float]]]]]

    @validator('type')
    def validate_type(cls, v):
        if v not in ['Polygon', 'MultiPolygon']:
            raise ValueError(
                'Geometry type must be either Polygon or MultiPolygon')
        return v

    @validator('coordinates')
    def validate_coordinates(cls, v, values):
        if 'type' not in values:
            return v

        if values['type'] == 'Polygon' and not isinstance(v[0][0], list):
            raise ValueError('Invalid Polygon coordinates format')
        elif values['type'] == 'MultiPolygon' and not isinstance(v[0][0][0], list):
            raise ValueError('Invalid MultiPolygon coordinates format')
        return v


class RecommendationRequest(BaseModel):
    area_of_interest: GeoJSONGeometry = Field(
        ...,
        description="GeoJSON Polygon or MultiPolygon representing the area of interest"
    )
    num_recommendations: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Number of recommendations to return (max 20)"
    )
    weights: Dict[str, float] = Field(
        default={
            "spatial_overlap": 0.3,
            "urgency": 0.3,
            "cost_effectiveness": 0.4
        },
        description="Optional weights for scoring criteria (must sum to 1.0)"
    )

    @validator('weights')
    def validate_weights(cls, v):
        if abs(sum(v.values()) - 1.0) > 0.001:
            raise ValueError('Weights must sum to 1.0')
        return v


@router.get("/features",
            summary="Get all geographic features",
            response_description="Returns the full GeoJSON FeatureCollection")
async def get_features(request: Request) -> Dict[str, Any]:
    """
    Return all features from the pre-loaded GeoDataFrame.

    Returns:
        Dict[str, Any]: A GeoJSON FeatureCollection containing all features
    """
    return request.app.state.geospatial_service.get_features()


@router.post("/recommendations",
             summary="Get ranked investment recommendations",
             response_description="Returns ranked GeoJSON features based on area of interest")
async def get_recommendations(request: RecommendationRequest, req: Request) -> Dict[str, Any]:
    """
    Generate ranked investment recommendations based on area of interest.

    Args:
        request: RecommendationRequest containing:
            - area_of_interest: GeoJSON Polygon/MultiPolygon
            - num_recommendations: Number of recommendations to return
            - weights: Optional scoring criteria weights

    Returns:
        Dict[str, Any]: A GeoJSON FeatureCollection containing ranked recommendations
                       with scoring information in properties
    """
    try:
        # Convert input area to shapely geometry
        area_geom = shape(request.area_of_interest.dict())

        # Get ranked recommendations using the service
        ranked_gdf = req.app.state.geospatial_service.get_recommendations(
            area_geom,
            request.num_recommendations,
            request.weights
        )

        # Convert to GeoJSON with additional properties
        result = {
            "type": "FeatureCollection",
            "metadata": {
                "weights_used": request.weights,
                "returned_features": len(ranked_gdf)
            },
            "features": json.loads(ranked_gdf.to_json())['features']
        }

        return jsonable_encoder(result)

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
