HouseTS
=======

*Large-scale, multimodal U.S. housing dataset + full benchmarking suite*

HouseTS offers monthly data from **2012‒2023** for ≈ 6 000 ZIP codes across 30 major metropolitan areas. Each (ZIP, month) record contains **33 engineered features** drawn from four complementary sources:

| Modality | Main sources | Examples of variables |
|----------|--------------|-----------------------|
| **Housing-market metrics** | Zillow Research, Redfin Data Center | Median sale/list price, inventory, new listings, days on market, transaction volumes |
| **Socioeconomic indicators** | U.S. Census Bureau **ACS 5-Year** | Income, population, labor-force size, poverty rate, rent burden, median commute time |
| **Points of Interest (POIs)** | OpenStreetMap via **ohsome API** | Monthly counts of restaurants, schools, supermarkets, parks, transit stations |
| **Aerial imagery** | USDA **NAIP** (1 m RGB) | Annual snapshots for a subset of ZIP codes in the Washington D.C.–Maryland–Virginia (DMV) region |

Typical research tasks
----------------------

* Spatio-temporal house-price prediction  
* Socioeconomic modeling that blends census and amenity data  
* Multimodal learning with tabular + satellite inputs  
* Urban-change detection through remote sensing and vision–language models  

Data access
-----------

| Purpose | Link |
|---------|------|
| **Raw & log-transformed CSVs** | <[https://virginiatech-my.sharepoint.com/:f:/g/personal/shengkun_vt_edu/EunsL7TsRDRMifm7MmVIbXsBGw5Mwg5JwuFsfXXAKHpvZQ?e=Z4tbU9](https://virginiatech-my.sharepoint.com/:f:/g/personal/shengkun_vt_edu/EunsL7TsRDRMifm7MmVIbXsBGw5Mwg5JwuFsfXXAKHpvZQ?e=WlxLmk)> |
| **Multimodal satellite & filled-missing data** | <https://www.kaggle.com/datasets/shengkunwang/housets-dataset> |

Baselines
---------

### Case study with DMV_Multi_Data

- Download `DMV_Multi_Data` from Kaggle  
- set  
  ```python
  from pathlib import Path
  DATA_ROOT = Path("/path/to/DMV_Multi_Data")
  ```

### Deep-learning (DNN)

We run our DNN baselines with the official [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

1. Copy the patched files in the `DNN/` directory so they overwrite the corresponding source files in the library.  
2. Download log dataset into the `dataset/` directory before training.

### Statistics & Machine-Learning Baselines

From the `StatML/` directory run:

```bash
python stat.py \
  --csv <path_to_csv> \
  --model all \
  --combos "(6,3),(6,6),(6,12),(12,3),(12,6),(12,12)" \
  --n_components 4
```

### Foundation-Model Experiments
All foundation-model runs can be started directly with the Python scripts inside the `Foundation/` folder.


### Citation
If you use **HouseTS** in your research, please cite:

```bibtex
@article{wang2025housets,
  title   = {HouseTS: A Large-Scale, Multimodal Spatiotemporal US Housing Dataset},
  author  = {Wang, Shengkun and Sun, Yanshen and Chen, Fanglan and Wang, Linhan and
             Ramakrishnan, Naren and Lu, Chang-Tien and Chen, Yinlin},
  journal = {arXiv e-prints},
  pages   = {arXiv--2506},
  year    = {2025}
}
