# Urban Green Space Equity Assessment System (GSF-DSS)

A [Streamlit](https://streamlit.io)-based web application for evaluating the match between urban green space supply and population demand, diagnosing underserved areas, and recommending new green space locations through multi-modal accessibility analysis.

## What It Does

The system works in four sequential stages:

1. **Supply** — compute each green patch's service capacity from its area and vegetation index.
2. **Demand** — generate a hexagonal population grid over the study area.
3. **Accessibility** — build walk / drive / bus road networks and compute per-cell accessibility.
4. **Diagnosis & siting** — flag underserved areas, quantify inequality (Gini), and recommend or simulate new green space locations.

## Installation

### Prerequisites

- Python 3.8+
- GDAL (a system dependency required by `rasterio` / `geopandas`)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run GSF-DSS.py
```

Then open the printed local URL (default `http://localhost:8501`) in a browser.

## Usage

### Stage 1 — Green Space Supply

1. Upload the green space boundary file (Shapefile ZIP).
2. Upload the NDVI raster (TIF).
3. Click **Process Supply Data** to compute each patch's area, mean NDVI, and supply index.

### Stage 2 — Demand & Population Grid

1. Select the study area via the administrative picker (Province → City → District).
2. Set the hexagonal grid size (100–500 m, default 200 m).
3. Click **Generate Grid & Extract Population**.

> **Note:** the population raster must be downloaded beforehand and placed in the project root (see [Data Preparation](#data-preparation)). If the file is missing, a placeholder population of `1` is used per cell.

### Stage 3 — Network & Accessibility

1. Upload an OSM road network file (PBF).
2. Click **Extract Road Network** — the system builds Walk / Drive / Bus graphs.
3. Set the time threshold (default 900 s) and click **Calculate Accessibility Scores**.

### Stage 4 — Diagnosis & Site Selection

1. Choose the analysis dimension (**Total** = Walk + Drive + Bus, or a single mode).
2. Adjust the **per-capita threshold percentile** (default P30) to flag underserved areas.
3. Review the fairness metrics (Gini, underserved population/grids) and the accessibility map.
4. Use **Model-based Siting (MCLP)** to auto-recommend new park locations, or enter coordinates for **Manual Siting**.
5. Compare before/after Gini coefficients and rescued population; optionally export results as Shapefile or CSV.

## Data Preparation

Green space vectors, NDVI rasters, and road network files are uploaded on demand through the web interface. **Only the population raster must be pre-downloaded** and placed in the project root directory.

| Data | Used in | How to obtain |
|------|---------|---------------|
| Green space boundaries (Shapefile ZIP) | Stage 1 — supply | Upload via web interface |
| NDVI raster (TIF) | Stage 1 — supply | Upload via web interface |
| Road network (OSM PBF) | Stage 3 — network | Upload via web interface |
| Population raster (TIF) | Stage 2 — demand grid | **Pre-download required** (see below) |

### Population raster download

Download the GeoTIFF from the [WorldPop 2020 China Population Dataset](https://www.worldpop.org/), rename it to `population_total_pop.tif`, and place it in the project root directory.

## Data Sources

| Data | Source |
|------|--------|
| Administrative boundaries | [Alibaba Cloud DataV.GeoAtlas](https://geo.datav.aliyun.com) (fetched via API at runtime — internet required) |
| Population raster | [WorldPop 2020 China Population Dataset](https://www.worldpop.org/) (manual download) |

## Project Structure

```
├── .streamlit/config.toml      # Streamlit config — raises upload limit for large rasters
├── GSF-DSS.py                  # Main application
├── population_total_pop.tif    # [Required] Population raster — download manually
├── requirements.txt            # Python dependency list
├── Fuzhou_NDVI_2020.tif        # [Sample] Fuzhou NDVI (2020)
├── Fuzhou_NDVI_2024.tif        # [Sample] Fuzhou NDVI (2024)
├── Fuzhou_Green_2020_Final.zip # [Sample] Fuzhou green space vectors (2020)
├── Fuzhou_Green_2024_Final.zip # [Sample] Fuzhou green space vectors (2024)
└── README.md
```

> `[Required]` files are necessary for the application to run. `[Sample]` files are experimental data from the Fuzhou case study, provided for reference only.

## Notes

- **Large raster uploads.** `.streamlit/config.toml` sets `maxUploadSize = 1024` (MB) so that ~500 MB NDVI rasters can be uploaded through the web interface.
- **Hard-coded UTM zone.** Accessibility and siting computations use `EPSG:32650` (UTM zone 50N), correct for Fuzhou and most of eastern China; change it in `GSF-DSS.py` for other regions.
- **Fuzhou-specific defaults.** When the selected city is Fuzhou (福州市), six core districts are pre-selected by default; other cities require manual district selection.
- **Internet required.** Administrative boundaries are fetched at runtime, so Stage 2 needs network access.
- **Bus model is a simplification.** It reuses drivable roads at a fixed 20 km/h and ignores real transit schedules.

## License

[Add your license here.]
