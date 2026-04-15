# migratory_birds_project
Building an AI/ML model to predict the springtime migration of a subset of birds to the Northeast, USA. 

## Data Ingestion From Google Drive (4GB Dataset)

This project includes a starter loader script for reading a large dataset from Google Drive in a memory-safe way for exploration and feature engineering.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data from Google Drive to local project storage

Use either a share URL or file ID.

```bash
python src/data_ingestion/google_drive_loader.py \
	--drive-url "https://drive.google.com/file/d/<YOUR_FILE_ID>/view?usp=sharing" \
	--destination "data/raw/bird_sightings.csv"
```

Or with file ID directly:

```bash
python src/data_ingestion/google_drive_loader.py \
	--file-id "<YOUR_FILE_ID>" \
	--destination "data/raw/bird_sightings.csv"
```

### 3. Read data in chunks for exploration

```python
from src.data_ingestion.google_drive_loader import iter_csv_chunks

for chunk in iter_csv_chunks(
		file_path="data/raw/bird_sightings.csv",
		chunksize=500_000,
	usecols=["species", "eventDate", "decimalLatitude", "decimalLongitude", "stateProvince"],
):
		print(chunk.head())
		break
```

### 4. If working in Google Colab

```python
from src.data_ingestion.google_drive_loader import mount_drive_if_colab, iter_csv_chunks

mount_drive_if_colab()
file_path = "/content/drive/MyDrive/path/to/bird_sightings.csv"

for chunk in iter_csv_chunks(file_path, chunksize=500_000):
		print(chunk.shape)
		break
```

### Notes for large-file workflows

- Prefer column selection with `usecols` during exploration.
- Keep raw files in `data/raw/` and derived datasets in `data/processed/`.
- Convert CSV to Parquet after initial profiling for faster iterative analysis.

### Optional: quick profile pass

```bash
python src/data_ingestion/quick_profile.py \
	--file-path data/raw/bird_sightings.csv \
	--species-column species \
	--date-column eventDate
```

## Build all-state migration timing features

Keep all states in the pipeline and create first-sighting timing features by species and year.

### 1. Build first sightings table (all US states)

```bash
python src/features/state_arrival_lag.py \
	--raw-file-path data/raw/bird_sightings.csv \
	--first-sightings-path data/processed/first_sightings_by_species_state_year.parquet \
	--country-code US
```

### 2. Compute state-to-state lag features

Example: if a species is first seen in South Carolina, estimate lag until first seen in Connecticut.

```bash
python src/features/state_arrival_lag.py \
	--raw-file-path data/raw/bird_sightings.csv \
	--first-sightings-path data/processed/first_sightings_by_species_state_year.parquet \
	--source-state "South Carolina" \
	--target-state "Connecticut" \
	--lag-output-path data/processed/lag_south_carolina_to_connecticut.parquet
```

This outputs per-species, per-year lag features including:
- `lag_days_source_to_target`
- `lag_weeks_source_to_target`

## Quick visualization and exploration for very large files

If the full file is too large to inspect directly, build a compact EDA bundle from a streamed subset.

Run:

```bash
python src/exploration/build_eda_assets.py \
	--file-path data/raw/bird_sightings.csv \
	--output-dir data/exploration \
	--max-rows 2000000 \
	--sample-rows 20000
```

This creates:
- `data/exploration/summary.json` with top species, states, years, and months
- `data/exploration/sample_preview.csv` for easy manual inspection
- plots in `data/exploration/plots/`:
	- `top_species.png`
	- `top_states.png`
	- `by_year.png`
	- `by_month.png`

You can start smaller for quick iteration:

```bash
python src/exploration/build_eda_assets.py \
	--file-path data/raw/bird_sightings.csv \
	--output-dir data/exploration_small \
	--max-rows 300000 \
	--sample-rows 5000
```

## Interactive exploration dashboard

Launch the app:

```bash
streamlit run src/exploration/dashboard_app.py
```

In the sidebar, you can:
- use an existing sample CSV (default: `data/exploration/sample_preview.csv`)
- or build/refresh a sample directly from the raw file in chunks

Interactive controls include:
- species multiselect
- state multiselect
- date-range slider

Dashboard views include:
- monthly sightings trend
- top species bar chart
- state-by-month heatmap
- map of sighting locations
- filtered table + CSV download
