HouseTS is a large-scale, multimodal dataset for long-term house price forecasting and socioeconomic analysis. It contains monthly observations from 2012 to 2023, covering 6,000 ZIP codes across 30 major U.S. metropolitan areas. The dataset integrates four types of information:

🏠 **Housing Market Features:** From Zillow and Redfin, including sale prices, inventory, listings, days on market, and transaction metrics.

📊 **Socioeconomic Indicators:** From the American Community Survey (ACS), including income, population, labor force, poverty, rent, and commute times.

📌 **Points of Interest (POIs):** Monthly ZIP-level counts of amenities such as restaurants, schools, supermarkets, parks, and transit stations, collected via OpenStreetMap.

🛰️ **Satellite Imagery:** 1-meter resolution NAIP aerial images for a subset of ZIP codes, primarily in the Washington D.C.–Maryland–Virginia (DMV) area.

Each row represents one ZIP code in one month and includes 33 engineered features. The dataset is designed to support multivariate time-series forecasting, imputation, multimodal fusion, and urban modeling tasks.

**Use cases include:**

- Spatio-temporal housing price prediction  
- Socioeconomic modeling using census and amenity data  
- Multimodal learning with tabular and satellite inputs  
- Urban change detection through imagery and vision–language models