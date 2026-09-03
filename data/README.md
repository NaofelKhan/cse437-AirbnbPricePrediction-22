# Data

## Source
New York City Airbnb Open Data (2019)
Kaggle: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
Original source: Inside Airbnb (http://insideairbnb.com/)

## Size
~48,895 rows, 16 columns

## How to obtain
1. Download `AB_NYC_2019.csv` from the Kaggle link above (requires free Kaggle account),
   or via Kaggle CLI:
   ```bash
   kaggle datasets download -d dgomonov/new-york-city-airbnb-open-data -p data/raw --unzip
   ```
2. Place the raw, untouched file in `data/raw/`.
3. Cleaned/transformed output from `notebooks/02_preprocessing.ipynb` is saved to `data/processed/`.
   Never edit files in `data/raw/` directly.

## Columns
`id`, `name`, `host_id`, `host_name`, `neighbourhood_group`, `neighbourhood`, `latitude`,
`longitude`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `last_review`,
`reviews_per_month`, `calculated_host_listings_count`, `availability_365`

## Known issues (confirm against your own audit in Notebook 01)
- `reviews_per_month` / `last_review` missing in ~20% of rows (listings with zero reviews)
- `price` contains implausible $0 listings and extreme high outliers
- `minimum_nights` contains extreme values (hundreds/thousands of nights)
- `price` is heavily right-skewed
