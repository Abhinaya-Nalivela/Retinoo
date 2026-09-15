# Data Directory Specification (SIH 26038)

This directory manages dataset partitions, raw downloads, annotations, splits, and sample validation assets.

## Directory Structure
- `raw/`: Raw downloaded archives for APTOS 2019, IDRiD, DRIVE, and Messidor-2.
- `processed/`: Standardized, cropped, and CLAHE-enhanced 512x512 fundus images.
- `annotations/`: Ground truth masks (vessels, optic disc, lesions) and CSV label records.
- `splits/`: Reproducible train/validation/test split CSV files preventing patient leakage.
- `sample_images/`: Synthetic and verified clinical cases for testing and instant local demonstration.

## Dataset Registry Sources
- **APTOS 2019**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- **IDRiD**: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- **DRIVE**: https://drive.grand-challenge.org/
- **Messidor-2**: https://www.adcis.net/en/third-party/messidor2/
