# Dataset Research & Acquisition Matrix

Status: **research performed 2026-08-17; downloads in progress.** Every source
below was verified against official documentation. Nothing was bypassed:
credentials-gated datasets are listed as gated and were **not** downloaded.

## Evaluated datasets

| Dataset | Official source | Access method | License / restrictions | Images | Patients | Format | Diseases | Label type | Download status |
|---|---|---|---|---|---|---|---|---|---|
| **NIH ChestX-ray14** | nihcc.app.box.com/v/ChestXray-NIHCC | Manual Box download | Research use; NIH distribution terms | 112,120 | 30,805 | PNG | 14 pathologies + No Finding | Multi-label CSV | **Excluded by project decision** |
| **CheXpert** | Stanford AIMI | Signed agreement + PhysioNet credentials | Research agreement | 224,316 | 65,240 | JPG | 14 findings | Multi-label CSV | Gated — not downloaded |
| **MIMIC-CXR** | PhysioNet | Credentialed access (CITI + DUA) | DUA required | 377,110 | 65,379 | JPG/DICOM | 14 findings | Extracted from reports | Gated — not downloaded |
| **PadChest** | BIMCV | Request to medical center | Research approval | 160,868 | 67,000+ | PNG/DICOM | 174 labels | Multi-label | Gated — not downloaded |
| **VinDr-CXR** | PhysioNet | Credentialed access | DUA required | 18,000 | 5,000+ | DICOM | 28 findings | Multi-label | Gated — not downloaded |
| **SIIM-ACR Pneumothorax** | Kaggle | Kaggle account + competition accept | Competition terms | 12,047 | 4,015 | DICOM | Pneumothorax | Segmentation | Gated — not downloaded |
| **RSNA Pneumonia Detection** | Kaggle | Kaggle account + competition accept | Competition terms | 26,684 | 26,684 | DICOM | Pneumonia (opacity) | BBox | Gated — not downloaded |
| **Kermany et al. (pneumonia)** | Mendeley Data / Kaggle | Account required | CC BY 4.0 | 5,863 | 5,863 | JPEG | Pneumonia / Normal | Binary folder | Gated — not downloaded |
| **COVID-19 Radiography Database** | Kaggle | Kaggle account | CC BY 4.0 | 21,165 | n/a | PNG | COVID-19 / Viral / Bacterial / Normal | Folder | Gated — not downloaded |
| **COVIDx (Figure1-COVID)** | GitHub | Direct git/zip download | CC BY-NC 4.0 | ~14,000 | ~13,500 | PNG | COVID-19 / Normal / Pneumonia | Metadata CSV | Candidate (not used: derived from the Cohen set below + augmentation) |
| **Cohen et al. covid-chestxray-dataset** | github.com/ieee8023 | Direct zip download | **CC BY 4.0** | ~900 (CXR subset ~600) | ~500 | PNG/JPG | COVID-19, viral/bacterial pneumonia, No Finding, SARS, MERS… | Metadata CSV | **DOWNLOADED — used** |
| **Shenzhen Hospital CXR (NLM)** | data.lhncbc.nlm.nih.gov | Direct per-file download | Public research use (NLM/NIH; do not redistribute) | **662** | 662 (1 image per patient) | PNG (~3k×3k) | Tuberculosis / Normal | Filename convention + clinical readings | **DOWNLOADED — used** |
| **Montgomery County CXR (NLM)** | data.lhncbc.nlm.nih.gov | Direct per-file download | Public research use (NLM/NIH) | 138 | 138 | PNG | Tuberculosis / Normal | Clinical readings (txt) | Partial — readings missing on NLM mirror; **excluded**, see note |

## Notes

- **Montgomery County set excluded:** a stable subset of its clinical-reading
  files returns HTTP 403 on the NLM mirror (verified per-file over multiple
  attempts); without complete labels the set cannot be used honestly. Shenzhen
  covers the same disease classes with complete, documented labels.
- **Shenzhen labels:** the NLM readme documents the filename convention
  `CHNCXR_#####_0/1.png` where `0` = normal, `1` = abnormal (TB); 336 TB +
  326 normal = 662 images. Clinical readings are downloaded alongside and
  cross-checked.
- **Cohen labels:** `metadata.csv` `finding` column is a hierarchical string
  (e.g. `COVID-19`, `Pneumonia/Viral/COVID-19`, `Pneumonia/Bacterial/
  Streptococcus`, `No Finding`). Only `modality == CXR` rows are used. Mapping
  to project classes is documented in `docs/label-mapping.md`.
- **Patient overlap:** Shenzhen and Cohen are disjoint institutions; patient
  IDs are namespaced by `dataset_source` during ingestion so a patient can
  never leak across partitions even if ID strings collide.
