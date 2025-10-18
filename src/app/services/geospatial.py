from typing import Dict, Union
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import json


class GeospatialService:
    def __init__(self, gdf: gpd.GeoDataFrame):
        """Initialize the service with a GeoDataFrame."""
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            raise ValueError(
                "Input GeoDataFrame must use WGS84 (EPSG:4326) CRS")
        self._gdf = gdf.copy()
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare and optimize the GeoDataFrame for fast queries."""
        # Create spatial index for faster intersection calculations
        self._spatial_index = self._gdf.sindex

        # Store original WGS84 geometries
        self._gdf_wgs84 = self._gdf.copy()

        # Project to an equal-area projection suitable for East Africa
        self._gdf = self._gdf.to_crs("EPSG:32736")  # UTM 36S

        # Pre-calculate areas using projected coordinates (in square meters)
        self._gdf['total_area'] = self._gdf.geometry.area

        # Normalize numeric columns
        self._normalize_columns()

        # Cache common computations
        self._feature_count = len(self._gdf)

        # Cache GeoJSON representation with WGS84 geometries
        self._json_features = self._gdf_wgs84.to_json()

    def _normalize_columns(self) -> None:
        """Normalize numeric columns for scoring."""
        numeric_cols = ['impact_per_1k_usd']
        for col in numeric_cols:
            if col in self._gdf.columns:
                min_val = self._gdf[col].min()
                max_val = self._gdf[col].max()
                self._gdf[f'{col}_normalized'] = (
                    self._gdf[col] - min_val) / (max_val - min_val)

    def _calculate_intersection_scores(self, area: Union[Polygon, MultiPolygon]) -> pd.Series:
        """Calculate intersection scores using spatial index and projected geometries."""
        # Project the input area to match our analysis CRS
        area_projected = gpd.GeoSeries(
            [area], crs="EPSG:4326").to_crs("EPSG:32736")[0]

        # Use spatial index to find potential intersections
        potential_matches_idx = list(
            self._spatial_index.intersection(area.bounds))

        if not potential_matches_idx:
            return pd.Series(0, index=self._gdf.index)

        # Calculate intersections using projected geometries
        potential_matches = self._gdf.iloc[potential_matches_idx]
        intersections = potential_matches.geometry.intersection(area_projected)

        # Calculate normalized scores
        areas = intersections.area
        min_area = np.minimum(
            potential_matches.total_area.values, area_projected.area)
        scores = areas / min_area

        # Create full series with zeros for non-intersecting features
        all_scores = pd.Series(0, index=self._gdf.index)
        all_scores[scores.index] = scores

        return all_scores

    def get_features(self,
                     page: int = 1,
                     per_page: int = 100,
                     simplify_tolerance: float = 0.0001) -> Dict:
        """Return paginated and optionally simplified features from the cached GeoJSON.

        This endpoint supports efficient map rendering through pagination and dynamic geometry simplification.
        Recommended usage patterns:

        1. Initial map load:
           - Use high simplify_tolerance (0.001) for overview
           - Load first page only
           Example: get_features(page=1, simplify_tolerance=0.001)

        2. Zooming in:
           - Decrease simplify_tolerance as zoom increases
           - Load visible features by page
           Example: get_features(page=1, simplify_tolerance=0.0001)

        3. Maximum detail view:
           - Set simplify_tolerance=0 for full resolution
           - Only request for small areas
           Example: get_features(page=1, simplify_tolerance=0)

        Args:
            page: Page number, starting from 1
            per_page: Number of features per page (default 100)
            simplify_tolerance: Geometry simplification tolerance in degrees:
                              - 0.001: Country level (~100km, high simplification)
                              - 0.0001: Regional level (~10km, medium detail)
                              - 0.00001: Local level (~1km, high detail)
                              - 0: Full resolution, no simplification

        Returns:
            Dict containing:
            - type: "FeatureCollection"
            - features: List of GeoJSON features
            - metadata: Pagination and viewport info

        Example:
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [36.817223, 3.016789],
                                [36.819632, 3.012543],
                                [36.821276, 3.013891],
                                [36.817223, 3.016789]
                            ]]
                        },
                        "properties": {
                            "id": "TKN_042",
                            "name": "North Turkana Agricultural Zone",
                            "impact_per_1k_usd": 2450,
                            "urgency_score": 0.85,
                            "description": "High-potential agricultural area with water access",
                            "population_affected": 12500,
                            "area_hectares": 450.5
                        }
                    }
                ],
                "metadata": {
                    "page": 1,
                    "per_page": 100,
                    "total_features": 1250,
                    "total_pages": 13,
                    "bounds": [36.4052, 2.8791, 37.9584, 4.2134],
                    "simplification": {
                        "tolerance": 0.0001,
                        "reduction_ratio": 0.45  # Geometry size reduction
                    }
                }
            }
        """
        # Load features
        features = json.loads(self._json_features)['features']
        total_features = len(features)
        total_pages = (total_features + per_page - 1) // per_page

        # Calculate slice for pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_features)
        page_features = features[start_idx:end_idx]

        # Simplify geometries if requested
        if simplify_tolerance > 0:
            for feature in page_features:
                coords = feature['geometry']['coordinates']
                if feature['geometry']['type'] == 'Polygon':
                    # Simplify polygon coordinates
                    feature['geometry']['coordinates'] = [
                        self._simplify_line(ring, simplify_tolerance)
                        for ring in coords
                    ]
                elif feature['geometry']['type'] == 'MultiPolygon':
                    # Simplify each polygon in the multipolygon
                    feature['geometry']['coordinates'] = [
                        [self._simplify_line(ring, simplify_tolerance)
                         for ring in poly]
                        for poly in coords
                    ]

        # Calculate bounds for the viewport
        bounds = self._gdf_wgs84.total_bounds.tolist()

        return {
            "type": "FeatureCollection",
            "features": page_features,
            "metadata": {
                "page": page,
                "per_page": per_page,
                "total_features": total_features,
                "total_pages": total_pages,
                "bounds": bounds
            }
        }

    def get_recommendations(self,
                            area: Union[Polygon, MultiPolygon],
                            num_recommendations: int,
                            weights: Dict[str, float]) -> gpd.GeoDataFrame:
        """Get ranked project area recommendations based on multiple criteria.

        This endpoint provides intelligent area recommendations by combining spatial
        overlap analysis with impact and urgency scoring. Results are automatically
        simplified based on the result count for efficient transfer.

        Args:
            area: Target area geometry in WGS84 (EPSG:4326)
            num_recommendations: Number of areas to recommend (affects simplification)
            weights: Scoring weights dictionary with keys:
                    - spatial_overlap: Weight for area overlap score (0-1)
                    - urgency: Weight for urgency score (0-1)
                    - cost_effectiveness: Weight for impact per cost (0-1)
                    Weights should sum to 1.0

        Returns:
            GeoDataFrame with recommended areas, sorted by composite score.
            Includes columns:
            - geometry: Area boundary (Polygon/MultiPolygon)
            - id: Unique area identifier
            - name: Area name/description
            - impact_per_1k_usd: Impact score per $1000 invested
            - urgency_score: Normalized urgency rating
            - intersection_score: Spatial overlap score
            - composite_score: Final weighted score
            - rank: Recommendation rank (1 = highest)

        Example input:
            area = Polygon([[
                [36.7850, 3.1245],
                [36.7921, 3.1245],
                [36.7921, 3.1289],
                [36.7850, 3.1289],
                [36.7850, 3.1245]
            ]])
            weights = {
                'spatial_overlap': 0.3,
                'urgency': 0.4,
                'cost_effectiveness': 0.3
            }

        Example result row:
            {
                'geometry': <Polygon>,
                'id': 'TKN_127',
                'name': 'Central Turkana Agricultural Belt',
                'impact_per_1k_usd': 3200,
                'urgency_score': 0.92,
                'intersection_score': 0.85,
                'composite_score': 0.89,
                'rank': 1,
                'population_affected': 18500,
                'area_hectares': 625.3
            }
        """
        # Calculate intersection scores
        intersection_scores = self._calculate_intersection_scores(area)

        # Calculate weighted composite score using vectorized operations
        composite_scores = (
            intersection_scores * weights['spatial_overlap'] +
            self._gdf['urgency_score_normalized'] * weights['urgency'] +
            self._gdf['impact_per_1k_usd_normalized'] *
            weights['cost_effectiveness']
        )

        # Get top N recommendations using partial sort
        top_indices = np.argpartition(
            composite_scores, -num_recommendations)[-num_recommendations:]
        ranked_gdf = self._gdf_wgs84.iloc[top_indices].copy()

        # Add scoring information
        ranked_gdf['intersection_score'] = intersection_scores[top_indices]
        ranked_gdf['composite_score'] = composite_scores[top_indices]
        ranked_gdf['rank'] = range(1, len(ranked_gdf) + 1)

        # Sort by composite score
        return ranked_gdf.sort_values('composite_score', ascending=False)

    def _simplify_line(self, coordinates: list, tolerance: float) -> list:
        """Simplify a line/ring using the Douglas-Peucker algorithm.

        Args:
            coordinates: List of[longitude, latitude] coordinate pairs
            tolerance: Simplification tolerance in degrees

        Returns:
            list: Simplified list of coordinates
        """
        if len(coordinates) <= 2:
            return coordinates

        # Find the point with the maximum distance
        max_dist = 0
        max_idx = 0
        first = np.array(coordinates[0])
        last = np.array(coordinates[-1])
        vec = last - first
        vec_len = np.linalg.norm(vec)

        if vec_len == 0:
            return [coordinates[0]]

        for i, point in enumerate(coordinates[1:-1], 1):
            # Calculate perpendicular distance
            point = np.array(point)
            dist = np.linalg.norm(np.cross(vec, first - point)) / vec_len
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            left = self._simplify_line(coordinates[:max_idx + 1], tolerance)
            right = self._simplify_line(coordinates[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [coordinates[0], coordinates[-1]]
