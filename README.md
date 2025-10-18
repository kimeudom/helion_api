# Helion API

A FastAPI-based geospatial recommendation service for agricultural and development projects in the Turkana region. This API provides efficient serving of GeoJSON data with dynamic simplification and intelligent area recommendations based on multiple criteria.


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kimeudom/helion_api.git
   cd helion_api
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

# Data Settings
### ! IMPORTANT
GEOJSON_PATH=src/app/turkana_recommendations_full_data.geojson
```

## Running the API

Start the development server:
```bash
fastapi dev src/app/main.py --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8000`

## API Usage Guide

### 1. Features Endpoint

`GET /features` - Returns paginated GeoJSON features with dynamic simplification.

#### Query Parameters
- `page` (int, default=1): Page number
- `per_page` (int, default=100): Features per page
- `simplify_tolerance` (float, default=0.0001): Geometry simplification level

#### Recommended Usage Patterns

1. **Initial Map Load** (Country Level):
```http
GET /features?page=1&simplify_tolerance=0.001
```

2. **Regional View**:
```http
GET /features?page=1&simplify_tolerance=0.0001
```

3. **Detailed Local View**:
```http
GET /features?page=1&simplify_tolerance=0
```

### 2. Recommendations Endpoint

`POST /recommendations` - Get ranked area recommendations.

#### Request Body
```json
{
    "area": {
        "type": "Polygon",
        "coordinates": [[[36.7850, 3.1245], [36.7921, 3.1245], 
                        [36.7921, 3.1289], [36.7850, 3.1289], 
                        [36.7850, 3.1245]]]
    },
    "num_recommendations": 5,
    "weights": {
        "spatial_overlap": 0.3,
        "urgency": 0.4,
        "cost_effectiveness": 0.3
    }
}
```

## Frontend Integration

### Efficient Map Loading

```javascript
// Initial load with high simplification
const overview = await api.getFeatures({
    page: 1,
    simplify_tolerance: 0.001  // Country level
});

// Handle zoom events
map.on('zoom', async (e) => {
    const zoom = map.getZoom();
    const tolerance = zoom < 8 ? 0.001 :   // Country
                     zoom < 12 ? 0.0001 :  // Regional
                     0;                    // Local
    
    const features = await api.getFeatures({
        page: 1,
        simplify_tolerance: tolerance
    });
});
```

### Recommendation System

```javascript
// Request recommendations
const recommendations = await api.getRecommendations({
    area: drawnPolygon,
    num_recommendations: 5,
    weights: {
        spatial_overlap: 0.3,
        urgency: 0.4,
        cost_effectiveness: 0.3
    }
});
```

## Performance Guidelines

1. **Geometry Simplification**:
   - Country level (zoom < 8): Use 0.001 (~100km simplification)
   - Regional level (zoom 8-12): Use 0.0001 (~10km simplification)
   - Local level (zoom > 12): Use 0 (full resolution)

2. **Data Loading**:
   - Use pagination (default 100 features per page)
   - Load data progressively as users pan/zoom
   - Consider visible map bounds when requesting data

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
