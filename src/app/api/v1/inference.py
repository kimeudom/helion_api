from typing import List, Optional, Dict, Any, Literal
from math import fabs
from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field, validator  # type: ignore
import random
from pathlib import Path
import json as _json
import glob

router = APIRouter(tags=["v1"])


class BBox(BaseModel):
    """A simple bounding box: [min_lon, min_lat, max_lon, max_lat]"""

    min_lon: float = Field(..., description="Minimum longitude")
    min_lat: float = Field(..., description="Minimum latitude")
    max_lon: float = Field(..., description="Maximum longitude")
    max_lat: float = Field(..., description="Maximum latitude")


class InferenceRequest(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4,
                              description="Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]")
    budget: float = Field(..., ge=0, description="Budget in project currency")
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional preferences from the frontend"
    )
    max_suggestions: int = Field(6, ge=1, le=50)

    @validator("bbox")
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError(
                "bbox must be [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = v
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        return v


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


@router.post("/v1/infer", response_model=InferenceResponse, summary="Get build suggestions for an area and budget", status_code=200)
async def infer(req: InferenceRequest):
    """Return a list of suggestions/plans for the selected area and budget.

    Parameters:
    - bbox: array[4] - Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
      - min_lon: float - Minimum longitude (western bound)
      - min_lat: float - Minimum latitude (southern bound)
      - max_lon: float - Maximum longitude (eastern bound)
      - max_lat: float - Maximum latitude (northern bound)
      Example: [36.5, -1.5, 37.0, -1.0]

    - budget: float - Available budget in project currency (must be >= 0)
    - preferences: object (optional) - Additional preferences for suggestion generation
    - max_suggestions: int - Maximum number of suggestions to return (1-50, default: 6)

    Returns:
    200 OK: List of scored suggestions with metadata

    Raises:
    400 Bad Request:
    - If bbox is not exactly [min_lon, min_lat, max_lon, max_lat]
    - If min_lon >= max_lon or min_lat >= max_lat
    - If budget is negative

    The endpoint uses GeoJSON data containing recommended build locations with properties:
    - estimated_cost: Estimated construction cost in project currency
    - urgency_score_normalized: Priority score (0-1) indicating need/urgency
    - impact_per_1k_usd: Impact metric per $1000 spent
    - mean_population: Average population in the area
    - mean_solar_potential: Solar resource potential
    - recommendation: Recommended building/infrastructure type
    - notes/justification: Explanation of the recommendation

    Alternative property names are supported for compatibility:
    - Cost: estimated_cost, predicted_cost, cost_usd, etc.
    - Urgency: urgency_score, urgency_normalized
    - Impact: impact_per_1k, benefit_per_1k_usd, impact_est

    If no GeoJSON data is available, the endpoint falls back to generating
    synthetic suggestions based on area and budget constraints.
    """
    min_lon, min_lat, max_lon, max_lat = req.bbox
    budget = req.budget

    bbox = BBox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat
    )

    # Try to load Turkana sample features and derive suggestions from them
    sample_feats = _load_turkana_sample()
    if sample_feats:
        suggestions = _suggestions_from_features(
            sample_feats, bbox=bbox, max_suggestions=req.max_suggestions)
        meta = {"source": "turkana_sample", "area_km2": round(
            _area_size_approx_km2(bbox), 6)}
    else:
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
@router.post("/v1/propose_region", response_model=InferenceResponse, summary="Get build suggestions constrained to a specific region")
async def propose_region(req: ProposeRequest):
    """Get build suggestions constrained to a specific region shape.

    Supports three region types:
    - circle: Defined by center point (lon,lat) and radius_m in meters
    - polygon: Defined by list of [lon,lat] coordinates forming a closed shape
    - bbox: Simple bounding box with min/max lon/lat

    The endpoint will:
    1. Find all GeoJSON features (build recommendations) that fall within the region
    2. Score them based on urgency, impact and cost metrics
    3. Filter and sort suggestions by score
    4. If no features found in region, fall back to nearest features by distance

    The suggestion scoring considers:
    - Urgency score (0-1) from urgency_score_normalized or similar fields
    - Impact per cost from impact_per_1k_usd or similar impact metrics
    - Estimated build cost relative to provided budget
    - For fallback suggestions: distance from region center

    Response includes:
    - Scored and ranked suggestions with locations and metadata
    - Original feature properties (population, solar potential, etc.)
    - GeoJSON outputs for map visualization
    - Layer-specific styling hints for the frontend
    """
    region = req.region
    budget = req.budget

    if not region or not isinstance(region, dict) or "type" not in region:
        raise HTTPException(
            status_code=400, detail="Region must be provided and include a 'type' field (circle|polygon|bbox)")

    # Derive a sampling bbox for spatial queries (unchanged from before)
    sample_bbox = None
    if region.get("type") == "bbox":
        sample_bbox = BBox(**region.get("bbox"))
    elif region.get("type") == "circle":
        c = region.get("center")
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

    # If Turkana sample available, use it as candidate pool and filter by region
    sample_feats = _load_turkana_sample()
    if sample_feats:
        # filter features by region
        candidates = []
        for f in sample_feats:
            coords = _feature_point_coords(f)
            if not coords:
                continue
            if _point_within_region(coords, region):
                candidates.append(f)
        # build suggestions from the features inside region
        filtered_suggestions = _suggestions_from_features(
            candidates, bbox=None, max_suggestions=req.max_suggestions)
        if not filtered_suggestions:
            # no features inside region; fall back to nearest features by distance to region center
            # compute region center
            region_center = None
            if region.get("type") == "circle":
                region_center = region.get("center")
            elif region.get("type") == "bbox":
                b = region.get("bbox")
                region_center = {"lon": (
                    b["min_lon"] + b["max_lon"]) / 2.0, "lat": (b["min_lat"] + b["max_lat"]) / 2.0}
            else:
                b = sample_bbox
                region_center = {
                    "lon": (b.min_lon + b.max_lon) / 2.0, "lat": (b.min_lat + b.max_lat) / 2.0}
            # annotate distance and sort
            feat_with_dist = []
            for f in sample_feats:
                coords = _feature_point_coords(f)
                if not coords:
                    continue
                d = haversine_distance_km(
                    coords["lon"], coords["lat"], region_center["lon"], region_center["lat"]) * 1000.0
                feat_with_dist.append((d, f))
            feat_with_dist.sort(key=lambda x: x[0])
            near_feats = [f for _, f in feat_with_dist[: req.max_suggestions]]
            filtered_suggestions = _suggestions_from_features(
                near_feats, bbox=None, max_suggestions=req.max_suggestions)

        # assemble layers from provided features in region (if any were supplied in request)
        provided = req.features or []
        layers = {"amenities": [], "infrastructure": [], "data": []}
        for feat in provided:
            coords = _feature_point_coords(feat)
            if not coords:
                continue
            if _point_within_region(coords, region):
                layers[_feature_layer(feat)].append(
                    {"lon": coords["lon"], "lat": coords["lat"], "properties": feat.properties})

        meta_out = {"source": "turkana_sample", "region_type": region.get("type"), "area_bbox": [
            sample_bbox.min_lon, sample_bbox.min_lat, sample_bbox.max_lon, sample_bbox.max_lat]}

        suggestion_centroids = _suggestions_to_featurecollection(
            filtered_suggestions, bbox=sample_bbox)
        render_hints = {
            "suggestions": {"marker_color": "#8A2BE2", "marker_radius": 7},
            "amenities": {"marker_color": "#FF8C00", "marker_radius": 6},
            "infrastructure": {"marker_color": "#0077BE", "marker_radius": 6},
            "data": {"marker_color": "#4CAF50", "marker_radius": 5},
        }
        meta_out["suggestion_centroids"] = suggestion_centroids
        meta_out["render_hints"] = render_hints

        return InferenceResponse(suggestions=filtered_suggestions, meta=meta_out)

    # Fallback: previous behaviour using generated candidates
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


def _load_turkana_sample() -> List[FeatureModel]:
    """Load the bundled Turkana recommendations GeoJSON (if present) and return a
    list of FeatureModel objects. This version will search several likely
    locations (src/app, repo root, any .geojson in the repository) so the
    endpoint works even if files are named differently or placed at repo root.
    For non-Point geometries the function will create a Point geometry using
    centroid_lon/centroid_lat properties when available so the frontend and
    suggestion builders can work with point centroids.
    """
    try:
        # primary path: src/app/turkana_recommendations_full_data.geojson
        base = Path(__file__).resolve()
        candidates: List[Path] = []
        try:
            repo_root = base.parents[4]
        except Exception:
            repo_root = Path.cwd()

        candidates.append(
            base.parents[2] / "turkana_recommendations_full_data.geojson")
        candidates.append(repo_root / "export.geojson")
        candidates.append(repo_root / "export.json")
        candidates.append(repo_root / "data" / "areas.geojson")

        # also include any .geojson file discovered under repo_root
        for p in glob.glob(str(repo_root / "**" / "*.geojson"), recursive=True):
            candidates.append(Path(p))

        # dedupe while preserving order
        seen = set()
        uniq_candidates = []
        for p in candidates:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                uniq_candidates.append(p)

        sample_path: Optional[Path] = None
        for p in uniq_candidates:
            if p and p.exists():
                sample_path = p
                break

        if sample_path is None:
            return []

        with sample_path.open("r", encoding="utf-8") as fh:
            ej = _json.load(fh)
        out: List[FeatureModel] = []
        for f in ej.get("features", []):
            geom = f.get("geometry")
            props = f.get("properties") or {}
            # If geometry is not a Point but centroid props exist, create a Point geometry
            if geom and geom.get("type") != "Point" and "centroid_lon" in props and "centroid_lat" in props:
                point_geom = {"type": "Point", "coordinates": [
                    float(props["centroid_lon"]), float(props["centroid_lat"])]}
                out.append(FeatureModel(geometry=point_geom, properties=props))
            else:
                # keep geometry as-is (Point or Polygon) — FeatureModel accepts dict geometry
                out.append(FeatureModel(geometry=geom, properties=props))
        return out
    except Exception:
        return []


def _suggestions_from_features(features: List[FeatureModel], bbox: Optional[BBox] = None, max_suggestions: int = 6) -> List[Suggestion]:
    """Create Suggestion models from GeoJSON features (expects point geometries
    or features converted to points by _load_turkana_sample). Filters to an
    optional bbox. Uses a tolerant property mapping to find estimated cost,
    urgency and impact values from features produced by different exports.
    """
    suggestions: List[Suggestion] = []

    # candidate property keys for common fields in different datasets
    est_cost_keys = ["estimated_cost", "est_cost", "predicted_cost", "cost",
                     "cost_usd", "predicted_cost_usd", "estimated_cost_usd", "estimated_cost_mean"]
    urgency_keys = ["urgency_score_normalized",
                    "urgency", "urgency_score", "urgency_normalized"]
    impact_keys = ["impact_per_1k_usd", "impact_per_1k", "impact_per_1000_usd",
                   "benefit_per_1k_usd", "impact", "impact_usd_per_1k", "impact_est"]

    for i, feat in enumerate(features):
        coords = _feature_point_coords(feat)
        if not coords:
            continue
        # bbox filter if provided
        if bbox is not None:
            if (
                coords["lon"] < bbox.min_lon
                or coords["lon"] > bbox.max_lon
                or coords["lat"] < bbox.min_lat
                or coords["lat"] > bbox.max_lat
            ):
                continue
        props = feat.properties or {}

        # tolerant extraction helpers
        def _first_numeric(keys_list):
            for k in keys_list:
                v = props.get(k)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
            return 0.0

        est_cost = _first_numeric(est_cost_keys)
        urgency = _first_numeric(urgency_keys)
        impact = _first_numeric(impact_keys)

        # If no estimated cost found, try to infer from other hints (population * unit cost)
        if est_cost == 0.0:
            pop = None
            try:
                pop = float(props.get("mean_population")
                            or props.get("population") or 0.0)
            except Exception:
                pop = 0.0
            if pop and pop > 0:
                # heuristic: cost per person fallback
                est_cost = pop * 100.0

        # Ensure values are finite and non-negative
        est_cost = max(0.0, float(est_cost or 0.0))
        urgency = max(0.0, min(1.0, float(urgency or 0.0)))
        impact = max(0.0, float(impact or 0.0))

        # simple combined score: urgency (0-1) plus small fraction of impact (scaled)
        score = max(0.0, min(1.0, urgency + min(1.0, impact / 100.0)))
        layout = {
            "recommended_build_type": props.get("recommendation") or props.get("recommended_build_type"),
            "mean_population": props.get("mean_population") or props.get("population"),
            "mean_solar_potential": props.get("mean_solar_potential") or props.get("solar_potential"),
        }
        suggestions.append(
            Suggestion(
                id=f"turkana-{i + 1}",
                centroid={"lon": coords["lon"], "lat": coords["lat"]},
                estimated_cost=round(est_cost, 2),
                score=round(score, 3),
                layout=layout,
                notes=props.get("justification") or props.get("notes"),
            )
        )
    # sort and trim
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:max_suggestions]


def _suggestions_to_featurecollection(suggestions: List[Suggestion], bbox: Optional[BBox] = None) -> Dict[str, Any]:
    """Convert a list of Suggestion models into a GeoJSON FeatureCollection where
    each feature represents the suggestion centroid with useful properties.

    (This function remained but we added above helpers that produce Suggestion
    objects from the Turkana dataset.)
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
    summary="Filter and organize geospatial features into layers for Kenya",
)
async def geodata_kenya(req: GeodataRequest):
    """Process and organize geospatial features for visualization in Kenya.

    Feature Categories:
    1. Amenities (orange markers):
       - Markets and shops
       - Community facilities
       - Public services
       Properties examined: layer, type, category containing "amenity", "shop", "market"

    2. Infrastructure (blue markers):
       - Roads and transportation
       - Power infrastructure
       - Water systems
       - Railways
       Properties examined: layer, type, category containing "infra", "road", "power", "rail"

    3. Data (green markers):
       - Population centers
       - Solar resource measurements
       - Other uncategorized points
       Default category for features not matching other rules

    Input Format:
      POST /v1/geodata/kenya
      {
        "features": [
          {
            "geometry": {
              "type": "Point",
              "coordinates": [longitude, latitude]
            },
            "properties": {
              "layer": "amenities",  // or "infrastructure" or inferred from type
              "type": "market",      // or other type hints
              "name": "...",         // optional
              "data": { ... }        // additional properties preserved
            }
          },
          ...
        ]
      }

    Processing:
    1. Features are filtered to Kenya bounds:
       lon: 33.5°E to 42.0°E
       lat: 5.5°S to 5.5°N

    2. Features are categorized by examining properties:
       - Explicit "layer" property
       - Type/category property keywords
       - Default to "data" layer

    3. For each layer:
       - Compute bounds
       - Track feature counts
       - Preserve all original properties
       - Add styling hints

    Response Structure:
      {
        "layers": {
          "amenities": {
            "type": "FeatureCollection",
            "features": [...],
            "bbox": [min_lon, min_lat, max_lon, max_lat]
          },
          "infrastructure": { ... },
          "data": { ... }
        },
        "meta": {
          "counts": {"amenities": n, "infrastructure": m, "data": k},
          "total": n+m+k,
          "render_hints": {
            "amenities": {"marker_color": "#FF8C00", "marker_radius": 6},
            "infrastructure": {"marker_color": "#0077BE", "marker_radius": 6},
            "data": {"marker_color": "#4CAF50", "marker_radius": 5}
          },
          "source_bbox": Kenya bounds
        }
      }

    The response is directly usable by mapping libraries (Leaflet, Mapbox GL, etc.)
    with the provided styling hints.
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


class MapLoadRequest(BaseModel):
    """Request to load initial map data for the frontend.

    - bbox: optional area to generate suggestions for (if omitted, will be derived)
    - max_suggestions: number of suggestions to produce
    - include_sample_geodata: when true, attempt to load export.geojson from workspace root
    - features: optional list of FeatureModel to include as geodata
    """

    bbox: Optional[BBox] = None
    max_suggestions: int = Field(6, ge=1, le=100)
    include_sample_geodata: bool = False
    features: Optional[List[FeatureModel]] = Field(default_factory=list)


class MapLoadResponse(BaseModel):
    layers: Dict[str, GeoJSONFeatureCollection]
    suggestions: List[Suggestion]
    suggestion_centroids: GeoJSONFeatureCollection
    meta: Dict[str, Any]


@router.post(
    "/v1/map_load",
    response_model=MapLoadResponse,
    summary="Load initial map data with features and suggestions",
)
async def map_load(req: MapLoadRequest):
    """Bootstrap a map view with features, suggestions and metadata.

    This endpoint combines:
    1. GeoJSON features from the dataset and/or provided features
    2. Build suggestions derived from feature properties
    3. Layer organization and styling metadata

    Data Sources:
    - Primary: GeoJSON dataset (loaded if include_sample_geodata=true)
      Contains build recommendations with:
      - Point or Polygon geometries (polygons get centroid points)
      - Properties like cost, impact, population, solar potential
      - Build type recommendations and justifications
    - Secondary: Additional features provided in request
      Grouped into layers: amenities, infrastructure, data

    Feature Processing:
    - Points are used as-is
    - Polygons are included with centroids for suggestion placement
    - Features are filtered to request bbox if provided
    - Properties are mapped flexibly (multiple field names supported)

    Response Structure:
    1. layers: GeoJSON FeatureCollections by category
       - amenities: Markets, shops, community facilities
       - infrastructure: Roads, power, water systems
       - data: Other data points and measurements
    2. suggestions: Scored build recommendations
       - Location (centroid point)
       - Cost and impact metrics
       - Population and solar potential
       - Build type recommendation
    3. metadata:
       - Feature counts by layer
       - Area calculations
       - Render hints (colors, marker sizes)
       - Bounding boxes
    """
    # Start with provided features
    provided: List[FeatureModel] = req.features or []

    # Optionally try to load sample turkana_recommendations_full_data.geojson from src/app
    if req.include_sample_geodata:
        try:
            sample_feats = _load_turkana_sample()
            # add to provided list (these are FeatureModel instances already)
            provided.extend(sample_feats)
        except Exception:
            pass

    # If bbox not provided, attempt to derive from provided features
    bbox = req.bbox
    if bbox is None:
        # collect coordinates
        lons: List[float] = []
        lats: List[float] = []
        for feat in provided:
            coords = _feature_point_coords(feat)
            if coords:
                lons.append(coords["lon"])
                lats.append(coords["lat"])
        if lons and lats:
            bbox = BBox(min_lon=min(lons), min_lat=min(lats),
                        max_lon=max(lons), max_lat=max(lats))
        else:
            # fallback to Kenya bbox used elsewhere
            bbox = BBox(min_lon=33.5, min_lat=-5.5, max_lon=42.0, max_lat=5.5)

    # Build geodata layers from provided features but *do not* re-filter to Kenya
    layers_fc: Dict[str, List[Dict[str, Any]]] = {
        "amenities": [], "infrastructure": [], "data": []}
    counts: Dict[str, int] = {"amenities": 0, "infrastructure": 0, "data": 0}

    for feat in provided:
        coords = _feature_point_coords(feat)
        if not coords:
            # if feature is not a point but has polygon geometry and centroid props we still include original polygon in layers
            geom = feat.geometry
            if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
                # keep polygon geometry; attempt to use centroid props for grouping
                props = feat.properties or {}
                layer_name = _feature_layer(feat)
                geo_feat = {"type": "Feature",
                            "geometry": feat.geometry, "properties": props}
                layers_fc[layer_name].append(geo_feat)
                counts[layer_name] += 1
            continue
        # if a bbox was explicitly provided in the request, filter features to it
        if req.bbox is not None:
            if (
                coords["lon"] < req.bbox.min_lon
                or coords["lon"] > req.bbox.max_lon
                or coords["lat"] < req.bbox.min_lat
                or coords["lat"] > req.bbox.max_lat
            ):
                continue

        layer_name = _feature_layer(feat)
        geo_feat = {"type": "Feature", "geometry": feat.geometry,
                    "properties": feat.properties or {}}
        layers_fc[layer_name].append(geo_feat)
        counts[layer_name] += 1

    # assemble feature collections
    layers_out: Dict[str, Dict[str, Any]] = {}
    for k, feats in layers_fc.items():
        layers_out[k] = {"type": "FeatureCollection", "features": feats}

    total = sum(counts.values())

    # generate suggestions: prefer dataset-derived suggestions if we have provided turkana features
    turkana_feats = [f for f in provided if (
        f.properties or {}).get("recommendation") is not None]
    if turkana_feats:
        suggestions = _suggestions_from_features(
            turkana_feats, bbox=bbox, max_suggestions=req.max_suggestions)
        s_meta = {"source": "turkana_sample",
                  "area_km2": round(_area_size_approx_km2(bbox), 6)}
    else:
        suggestions, s_meta = _generate_suggestions(
            bbox, budget=0.0, max_suggestions=req.max_suggestions, prefs={})

    # convert suggestions to GeoJSON centroids
    suggestion_centroids = _suggestions_to_featurecollection(
        suggestions, bbox=bbox)

    # render hints (consistent with other endpoints)
    render_hints = {
        "amenities": {"marker_color": "#FF8C00", "marker_radius": 6},
        "infrastructure": {"marker_color": "#0077BE", "marker_radius": 6},
        "data": {"marker_color": "#4CAF50", "marker_radius": 5},
        "suggestions": {"marker_color": "#8A2BE2", "marker_radius": 7},
    }

    meta = {
        "counts": counts,
        "total": total,
        "render_hints": render_hints,
        "suggestion_meta": s_meta,
        "source_bbox": {"min_lon": bbox.min_lon, "min_lat": bbox.min_lat, "max_lon": bbox.max_lon, "max_lat": bbox.max_lat},
    }

    # convert GeoJSONFeatureCollection models in response type
    geo_collections: Dict[str, GeoJSONFeatureCollection] = {}
    for k, v in layers_out.items():
        fc = GeoJSONFeatureCollection(
            features=v["features"], bbox=v.get("bbox"))
        geo_collections[k] = fc

    suggestion_fc = GeoJSONFeatureCollection(features=suggestion_centroids.get(
        "features", []), bbox=suggestion_centroids.get("bbox"))

    return MapLoadResponse(layers=geo_collections, suggestions=suggestions, suggestion_centroids=suggestion_fc, meta=meta)
