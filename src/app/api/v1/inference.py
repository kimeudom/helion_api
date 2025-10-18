from typing import List, Optional, Dict, Any, Literal
from math import fabs
from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
import random

router = APIRouter(tags=["v1"])


class BBox(BaseModel):
    """A simple bounding box: [min_lon, min_lat, max_lon, max_lat]"""

    min_lon: float = Field(..., description="Minimum longitude")
    min_lat: float = Field(..., description="Minimum latitude")
    max_lon: float = Field(..., description="Maximum longitude")
    max_lat: float = Field(..., description="Maximum latitude")


class InferenceRequest(BaseModel):
    bbox: BBox
    budget: float = Field(..., ge=0, description="Budget in project currency")
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional preferences from the frontend"
    )
    max_suggestions: int = Field(6, ge=1, le=50)


class Suggestion(BaseModel):
    id: str
    centroid: Dict[str, float]
    estimated_cost: float
    score: float = Field(..., ge=0, le=1)
    layout: Dict[str, Any]
    notes: Optional[str] = None


class InferenceResponse(BaseModel):
    suggestions: List[Suggestion]
    meta: Dict[str, Any]


def _area_size_approx_km2(bbox: BBox) -> float:
    # Very rough area approximation (not accurate for large areas) assuming degrees ~ km
    lon_span = fabs(bbox.max_lon - bbox.min_lon)
    lat_span = fabs(bbox.max_lat - bbox.min_lat)
    # approx: 1 degree lat ~ 111 km, lon varies; use 111 km for both for simplicity
    return (lon_span * 111.0) * (lat_span * 111.0) / 1_000_000.0


def _generate_suggestions(bbox: BBox, budget: float, max_suggestions: int, prefs: Dict[str, Any]):
    # Deterministic pseudo-random generator seeded from bbox+budget so repeated calls are stable
    seed = int((bbox.min_lon + bbox.min_lat + bbox.max_lon +
               bbox.max_lat) * 100000) ^ int(budget)
    rnd = random.Random(seed)

    area_km2 = _area_size_approx_km2(bbox)

    # base unit cost per km2 (mock)
    base_cost_per_km2 = prefs.get("base_cost_per_km2", 1000000)

    suggestions = []
    for i in range(max_suggestions):
        # sample a centroid inside the bbox
        lon = rnd.uniform(bbox.min_lon, bbox.max_lon)
        lat = rnd.uniform(bbox.min_lat, bbox.max_lat)

        # TODO
        # pseudo complexity factor from preferences
        complexity = float(prefs.get("complexity", rnd.uniform(0.5, 1.5)))

        # TODO
        estimated_cost = max(1000.0, area_km2 * base_cost_per_km2 *
                             complexity * (1 + rnd.uniform(-0.3, 0.6)))

        # score: higher means better fit to budget and to preference for lower cost
        budget_diff = fabs(budget - estimated_cost)
        # normalize score between 0 and 1 using an ad-hoc formula
        score = max(
            0.0, 1.0 - min(1.0, (budget_diff / max(1.0, budget + 1e-6))))
        # slightly boost score if estimated_cost <= budget
        if estimated_cost <= budget:
            score = min(1.0, score + 0.1)

        layout = {
            "units_estimated": int(max(1, (budget // 50000) * (1 if complexity < 1.0 else 1))),
            "recommended_build_type": "mixed-use" if complexity > 1.0 else "residential",
            "approx_footprint_m2": int(max(50, area_km2 * 1_000_000 * (0.1 + rnd.random() * 0.4))),
        }

        suggestions.append(
            Suggestion(
                id=f"sugg-{i + 1}",
                centroid={"lon": lon, "lat": lat},
                estimated_cost=round(estimated_cost, 2),
                score=round(score, 3),
                layout=layout,
                notes=None,
            )
        )

    # sort suggestions by score descending
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions, {"area_km2": round(area_km2, 6)}


@router.post("/v1/infer", response_model=InferenceResponse, summary="Get build suggestions for an area and budget")
async def infer(req: InferenceRequest):
    """Return a list of suggestions/plans for the selected area and budget.

    The frontend should POST a JSON body with `bbox` and `budget` plus optional
    `preferences`. The response contains scored suggestions the frontend can show
    on the map and allow users to select and drill into details.
    """
    bbox = req.bbox
    budget = req.budget

    if bbox.min_lon >= bbox.max_lon or bbox.min_lat >= bbox.max_lat:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates")

    suggestions, meta = _generate_suggestions(
        bbox, budget, req.max_suggestions, req.preferences)

    # also provide a GeoJSON FeatureCollection of suggestion centroids and render hints
    suggestion_centroids = _suggestions_to_featurecollection(
        suggestions, bbox=bbox)
    render_hints = {
        "suggestions": {"marker_color": "#8A2BE2", "marker_radius": 7},
    }

    return InferenceResponse(
        suggestions=suggestions,
        meta={**meta, "budget": budget, "suggestion_centroids": suggestion_centroids, "render_hints": render_hints,
              "source_bbox": {"min_lon": bbox.min_lon, "min_lat": bbox.min_lat, "max_lon": bbox.max_lon, "max_lat": bbox.max_lat}},
    )


def haversine_distance_km(a_lon: float, a_lat: float, b_lon: float, b_lat: float) -> float:
    # approximate great-circle distance between two points (km)
    from math import radians, sin, cos, atan2, sqrt

    r = 6371.0
    dlon = radians(b_lon - a_lon)
    dlat = radians(b_lat - a_lat)
    a = sin(dlat / 2) ** 2 + cos(radians(a_lat)) * \
        cos(radians(b_lat)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def _point_in_polygon(pt: Dict[str, float], polygon: List[List[float]]) -> bool:
    # ray-casting algorithm for point-in-polygon. polygon is list of [lon, lat]
    x = pt["lon"]
    y = pt["lat"]
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def _point_in_circle(pt: Dict[str, float], center: Dict[str, float], radius_m: float) -> bool:
    # radius in meters
    dist_km = haversine_distance_km(
        pt["lon"], pt["lat"], center["lon"], center["lat"])
    return (dist_km * 1000.0) <= radius_m


# Region shape models
class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Dict[str, float]
    radius_m: float = Field(..., ge=0)


class PolygonShape(BaseModel):
    type: Literal["polygon"] = "polygon"
    # list of [lon, lat]
    coordinates: List[List[float]]


class RegionBBox(BaseModel):
    type: Literal["bbox"] = "bbox"
    bbox: BBox


Region = Optional[Dict[str, Any]]


class FeatureModel(BaseModel):
    # simple GeoJSON-like feature with geometry.type="Point" and properties
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class ProposeRequest(BaseModel):
    region: Dict[str, Any]
    budget: float = Field(..., ge=0)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_suggestions: int = Field(6, ge=1, le=50)
    # optional features: list of geo features (points) containing population/amenities/infrastructure
    features: Optional[List[FeatureModel]] = Field(default_factory=list)


def _feature_point_coords(feat: FeatureModel) -> Optional[Dict[str, float]]:
    geom = feat.geometry
    if not geom:
        return None
    if geom.get("type") == "Point":
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            return {"lon": float(coords[0]), "lat": float(coords[1])}
    return None


def _feature_layer(feat: FeatureModel) -> str:
    # Expect properties to contain a layer key or type-like key. Normalize to one of three layers.
    props = feat.properties or {}
    layer = props.get("layer") or props.get(
        "type") or props.get("category") or "data"
    layer = str(layer).lower()
    if "amenit" in layer or "shop" in layer or "market" in layer:
        return "amenities"
    if "infra" in layer or "road" in layer or "power" in layer or "rail" in layer:
        return "infrastructure"
    return "data"


def _point_within_region(point: Dict[str, float], region: Dict[str, Any]) -> bool:
    rtype = region.get("type")
    if rtype == "circle":
        center = region.get("center")
        radius_m = float(region.get("radius_m", 0))
        return _point_in_circle(point, center, radius_m)
    if rtype == "polygon":
        coords = region.get("coordinates", [])
        return _point_in_polygon(point, coords)
    if rtype == "bbox":
        bbox_obj = region.get("bbox")
        if isinstance(bbox_obj, dict):
            return (
                point["lon"] >= float(bbox_obj.get("min_lon"))
                and point["lon"] <= float(bbox_obj.get("max_lon"))
                and point["lat"] >= float(bbox_obj.get("min_lat"))
                and point["lat"] <= float(bbox_obj.get("max_lat"))
            )
    return False


# New endpoint: propose region-aware suggestions
@router.post("/v1/propose_region", response_model=InferenceResponse, summary="Get suggestions constrained to an arbitrary region (circle, polygon, bbox)")
async def propose_region(req: ProposeRequest):
    region = req.region
    budget = req.budget

    if not region or not isinstance(region, dict) or "type" not in region:
        raise HTTPException(
            status_code=400, detail="Region must be provided and include a 'type' field (circle|polygon|bbox)")

    # Reuse generation to create candidate suggestions across the bounding box of the region
    # Derive a bbox for sampling if possible
    sample_bbox = None
    if region.get("type") == "bbox":
        sample_bbox = BBox(**region.get("bbox"))
    elif region.get("type") == "circle":
        c = region.get("center")
        # small bbox around center approximating radius in degrees (very rough)
        radius_km = float(region.get("radius_m", 0)) / 1000.0
        deg = radius_km / 111.0
        sample_bbox = BBox(min_lon=c["lon"] - deg, min_lat=c["lat"] -
                           deg, max_lon=c["lon"] + deg, max_lat=c["lat"] + deg)
    elif region.get("type") == "polygon":
        coords = region.get("coordinates", [])
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        sample_bbox = BBox(min_lon=min(lons), min_lat=min(
            lats), max_lon=max(lons), max_lat=max(lats))
    else:
        raise HTTPException(status_code=400, detail="Unknown region type")

    # generate candidates using same prefs
    candidates, meta = _generate_suggestions(
        sample_bbox, budget, req.max_suggestions * 3, req.preferences)

    # filter candidates to those whose centroid falls within the region
    filtered = [c for c in candidates if _point_within_region(
        {"lon": c.centroid["lon"], "lat": c.centroid["lat"]}, region)]

    # If none inside, fall back to selecting top-scoring candidates but include distance to region center (if circle) or nearest feature
    if not filtered:
        # compute distance to region center if possible
        region_center = None
        if region.get("type") == "circle":
            region_center = region.get("center")
        elif region.get("type") == "bbox":
            b = region.get("bbox")
            region_center = {"lon": (
                b["min_lon"] + b["max_lon"]) / 2.0, "lat": (b["min_lat"] + b["max_lat"]) / 2.0}
        else:
            # use bbox center for polygon
            b = sample_bbox
            region_center = {
                "lon": (b.min_lon + b.max_lon) / 2.0, "lat": (b.min_lat + b.max_lat) / 2.0}

        for c in candidates:
            c._distance_to_region_m = haversine_distance_km(
                c.centroid["lon"], c.centroid["lat"], region_center["lon"], region_center["lat"]) * 1000.0
        candidates.sort(key=lambda s: getattr(s, "_distance_to_region_m", 0.0))
        filtered = candidates[: req.max_suggestions]
    else:
        filtered = filtered[: req.max_suggestions]

    # From provided features, assemble applicable geopositions inside region grouped by layer
    provided = req.features or []
    layers = {"amenities": [], "infrastructure": [], "data": []}
    for feat in provided:
        coords = _feature_point_coords(feat)
        if not coords:
            continue
        if _point_within_region(coords, region):
            layers[_feature_layer(feat)].append(
                {"lon": coords["lon"], "lat": coords["lat"], "properties": feat.properties})

    # build response meta
    meta_out = {**meta, "budget": budget, "region_type": region.get("type")}
    meta_out["applicable_geopositions"] = layers

    # Add GeoJSON featurecollection of filtered suggestion centroids and render hints for frontend
    suggestion_centroids = _suggestions_to_featurecollection(
        filtered, bbox=sample_bbox)
    render_hints = {
        "suggestions": {"marker_color": "#8A2BE2", "marker_radius": 7},
        "amenities": {"marker_color": "#FF8C00", "marker_radius": 6},
        "infrastructure": {"marker_color": "#0077BE", "marker_radius": 6},
        "data": {"marker_color": "#4CAF50", "marker_radius": 5},
    }
    meta_out["suggestion_centroids"] = suggestion_centroids
    meta_out["render_hints"] = render_hints

    # ensure returned suggestions are Pydantic models (they already are) but convert to list
    return InferenceResponse(suggestions=filtered, meta=meta_out)


def _suggestions_to_featurecollection(suggestions: List[Suggestion], bbox: Optional[BBox] = None) -> Dict[str, Any]:
    """Convert a list of Suggestion models into a GeoJSON FeatureCollection where
    each feature represents the suggestion centroid with useful properties.
    """
    features: List[Dict[str, Any]] = []
    for s in suggestions:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s.centroid["lon"], s.centroid["lat"]]},
            "properties": {
                "id": s.id,
                "estimated_cost": s.estimated_cost,
                "score": s.score,
                "layout": s.layout,
            },
        })

    fc: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if bbox is not None:
        fc["bbox"] = [round(bbox.min_lon, 6), round(bbox.min_lat, 6), round(
            bbox.max_lon, 6), round(bbox.max_lat, 6)]
    return fc


class GeodataRequest(BaseModel):
    features: List[FeatureModel]


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = Field(default_factory=dict)


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list)
    bbox: Optional[List[float]] = None


class GeodataResponse(BaseModel):
    """Response structure containing per-layer GeoJSON FeatureCollections and metadata

    layers: Dict[str, GeoJSONFeatureCollection] -- keys: 'amenities', 'infrastructure', 'data'
    meta: Dict[str, Any] -- counts, render hints and source bbox for clarity
    """
    layers: Dict[str, GeoJSONFeatureCollection]
    meta: Dict[str, Any]


@router.post(
    "/v1/geodata/kenya",
    response_model=GeodataResponse,
    summary="Return geodata layers (amenities, infrastructure, data) filtered to Kenya from supplied features",
)
async def geodata_kenya(req: GeodataRequest):
    """
    Filter supplied point features to those inside an approximate Kenya bbox and return
    layer-separated GeoJSON FeatureCollections that can be rendered directly by browser
    mapping libraries (Leaflet, Mapbox GL, OpenLayers).

    Usage:
      POST /v1/geodata/kenya
      Body: { "features": [ {"geometry": {"type":"Point","coordinates":[lon,lat]}, "properties": {...} }, ... ] }

    Returned JSON:
      {
        "layers": {
          "amenities": { "type": "FeatureCollection", "features": [...], "bbox": [min_lon,min_lat,max_lon,max_lat] },
          "infrastructure": { ... },
          "data": { ... }
        },
        "meta": {
          "counts": {"amenities": n, "infrastructure": m, "data": k},
          "total": n+m+k,
          "render_hints": {"amenities": {"marker_color": "#...", "marker_radius": 6}, ...},
          "source_bbox": {"min_lon":...,"max_lon":...,"min_lat":...,"max_lat":...}
        }
      }

    Notes:
      - The Kenya bbox used for filtering is a simple approximation and intended only
        as a quick client-side filter; for exact country boundaries use polygon-based
        checking on the client or a more detailed dataset on the server.
      - Each layer is returned as a valid GeoJSON FeatureCollection which most mapping
        libraries can consume directly. The `render_hints` object provides simple
        presentation guidance (colors/radii) the frontend can use when styling markers.
    """
    # Kenya approximate bbox (used for quick filtering)
    kenya_bbox = {"min_lon": 33.5, "max_lon": 42.0,
                  "min_lat": -5.5, "max_lat": 5.5}

    # Prepare accumulators per-layer
    layers_fc: Dict[str, List[Dict[str, Any]]] = {
        "amenities": [], "infrastructure": [], "data": []}
    counts: Dict[str, int] = {"amenities": 0, "infrastructure": 0, "data": 0}

    # Track per-layer bbox [min_lon, min_lat, max_lon, max_lat]
    bboxes: Dict[str, Optional[List[float]]] = {
        "amenities": [float("inf"), float("inf"), float("-inf"), float("-inf")],
        "infrastructure": [float("inf"), float("inf"), float("-inf"), float("-inf")],
        "data": [float("inf"), float("inf"), float("-inf"), float("-inf")],
    }

    for feat in req.features:
        coords = _feature_point_coords(feat)
        if not coords:
            continue
        # quick bbox filter
        if (
            coords["lon"] < kenya_bbox["min_lon"]
            or coords["lon"] > kenya_bbox["max_lon"]
            or coords["lat"] < kenya_bbox["min_lat"]
            or coords["lat"] > kenya_bbox["max_lat"]
        ):
            continue

        layer_name = _feature_layer(feat)

        # Build a GeoJSON feature (compatible with browser mapping libs)
        geo_feat = {
            "type": "Feature",
            "geometry": feat.geometry,
            "properties": feat.properties or {},
        }

        layers_fc[layer_name].append(geo_feat)
        counts[layer_name] += 1

        # update layer bbox
        lbb = bboxes[layer_name]
        lbb[0] = min(lbb[0], coords["lon"])
        lbb[1] = min(lbb[1], coords["lat"])
        lbb[2] = max(lbb[2], coords["lon"])
        lbb[3] = max(lbb[3], coords["lat"])

    # Normalize bboxes: convert empty (inf) to None and round values
    for k, v in list(bboxes.items()):
        if v[0] == float("inf"):
            bboxes[k] = None
        else:
            bboxes[k] = [round(v[0], 6), round(v[1], 6),
                         round(v[2], 6), round(v[3], 6)]

    # Build GeoJSON FeatureCollections and assemble response
    layers_out: Dict[str, Dict[str, Any]] = {}
    for k, feats in layers_fc.items():
        fc: Dict[str, Any] = {"type": "FeatureCollection", "features": feats}
        if bboxes[k] is not None:
            fc["bbox"] = bboxes[k]
        layers_out[k] = fc

    total = sum(counts.values())

    # Simple render hints frontends can use for styling
    render_hints = {
        "amenities": {"marker_color": "#FF8C00", "marker_radius": 6},
        "infrastructure": {"marker_color": "#0077BE", "marker_radius": 6},
        "data": {"marker_color": "#4CAF50", "marker_radius": 5},
    }

    meta = {"counts": counts, "total": total,
            "render_hints": render_hints, "source_bbox": kenya_bbox}

    return GeodataResponse(layers=layers_out, meta=meta)
