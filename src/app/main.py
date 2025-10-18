from fastapi import FastAPI, Request  # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import geopandas as gpd
from .core.config import Settings
from .services.geospatial import GeospatialService
from .api.v1.features import router as features_router

settings = Settings()

# Initialize the GeospatialService
geojson_path = Path(__file__).parent / \
    "turkana_recommendations_full_data.geojson"

# Read GeoJSON and ensure WGS84 CRS
gdf = gpd.read_file(str(geojson_path))
if gdf.crs is None:
    gdf.set_crs("EPSG:4326", inplace=True)
elif gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# Create global service instance
geospatial_service = GeospatialService(gdf)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Helion API for geospatial recommendations"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add service to app state


@app.on_event("startup")
async def startup_event():
    app.state.geospatial_service = geospatial_service

# include versioned routers under /api
app.include_router(features_router, prefix="/api/v1", tags=["features"])


@app.get("/", summary="App root")
def root():
    return {"app": settings.app_name, "version": settings.version}
