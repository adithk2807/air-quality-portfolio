# air-quality-portfolio
Statistical and Causal Inference portfolio

# Air Quality – Statistical and Causal Inference Portfolio

This repository contains the code to reproduce my portfolio analyses
for the course **Statistical and Causal Inference** (Items 6–9).

The project uses the PRSA Beijing Multi-Site Air Quality dataset (2013–2017) to study how meteorological variables, especially wind speed, influence PM2.5 pollution, and to perform both statistical and causal inference.

---

## Contents
 
  Main Python script. It:
  - load the PRSA Beijing air quality data,
  - regression analyses (Item 6),
  - compare linear and non-linear models (Items 7–8),
  - causal inference via covariate adjustment (Item 9),
  - causal inference via propensity scores (Item 10).

## Dataset

The dataset is **not** stored in this repository.  
Please download it separately from Kaggle:

- PRSA Beijing Multi-Site Air Quality (2013–2017):  
  https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set

After downloading and unzipping, you should have a folder containing the 12 CSV files
(one for each station).

Example folder on Windows:

```text
C:\Users\YOUR_NAME\Downloads\archive
    ├── PRSA_Data_Changping_20130301-20170228.csv
    ├── PRSA_Data_Dingling_20130301-20170228.csv
    ├── ...
