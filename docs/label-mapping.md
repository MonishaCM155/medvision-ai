# Label Mapping (dataset → project classes)

Every mapping below is explicit. A mapping is only accepted when the source
dataset defines the label that way; nothing is inferred across diseases.

## Sources

| Source | Dataset | Labels provided | Format |
|---|---|---|---|
| NLM/LHNCBC | Shenzhen Hospital CXR set (662 images) | Per-image clinical reading (text) + readme-documented filename convention `CHNCXR_#####_0/1.png` (0 = normal, 1 = abnormal = pulmonary tuberculosis) | PNG ~3k×3k |
| Cohen et al. (CC BY 4.0) | covid-chestxray-dataset (950 rows; X-ray modality only) | `metadata.csv` `finding` column (hierarchical, e.g. `Pneumonia/Viral/COVID-19`), `modality` (`X-ray`/`CT`), `patientid` | PNG/JPG |

## Mappings

### Shenzhen → project labels

| Source label | Project label | Justification |
|---|---|---|
| `CHNCXR_#####_0` (filename) | `No Finding` | NLM readme: "0 represents the normal lung" |
| `CHNCXR_#####_1` (filename) | `Tuberculosis` | NLM readme: "1 represents the abnormal lung"; 336 cases with manifestation of tuberculosis |
| Clinical reading last line (`normal` / `ptb` / `stb` / `natb` / `atb` / `tb` / `tuberculosis ...`) | cross-check | Validated: 662/662 readings agree with the filename convention (0 disagreements; 1 case reads "right upper pneumonia" but is `_1` = abnormal per the documented convention) |

### Cohen et al. → project labels (X-ray modality only)

| Source finding | Project label | Justification |
|---|---|---|
| `No Finding` | `No Finding` | Literal label |
| `Pneumonia/Viral/COVID-19` | `COVID-19` | The dataset defines COVID-19 as a pneumonia subtype; for the detection task COVID-19 supersedes the generic `Pneumonia` label (documented decision — a COVID case is labeled COVID-19, not both) |
| `Pneumonia`, `Pneumonia/Bacterial/*`, `Pneumonia/Viral/*` (non-COVID), `Pneumonia/Fungal/*`, `Pneumonia/Lipoid`, `Pneumonia/Aspiration` | `Pneumonia` | Literal pneumonia findings |
| `todo`, `Unknown`, `Pneumonia/Viral/SARS`, `Pneumonia/Viral/MERS-CoV`, `Tuberculosis` | — (excluded) | `todo`/`Unknown` carry no usable label; SARS/MERS are coronaviruses but not COVID-19 and not in the target classes; the 18 Cohen `Tuberculosis` rows are excluded to keep TB provenance exclusively with Shenzhen |
| `modality == CT` | — (excluded) | Project is chest **X-ray** only |

## Class availability for the current data mix

| Project class | Trained? | Source(s) |
|---|---|---|
| `No Finding` | ✅ | Shenzhen normal (326) + Cohen `No Finding` (18 after file checks) |
| `Pneumonia` | ✅ | Cohen non-COVID pneumonia (216) |
| `COVID-19` | ✅ | Cohen COVID-19 (504) |
| `Tuberculosis` | ✅ | Shenzhen (336) |
| `Atelectasis` | ❌ unavailable | No accessible dataset with this label |
| `Pleural Effusion` | ❌ unavailable | Gated datasets only |
| `Edema` | ❌ unavailable | Gated datasets only |
| `Cardiomegaly` | ❌ unavailable | Gated datasets only |
| `Lung Opacity` | ❌ unavailable | Gated datasets only |
| `Pneumothorax` | ❌ unavailable | Kaggle-gated |

Patient identity is namespaced by source (`shenzhen_…`, `covid_…`) and the
patient-level split guarantees no patient appears in more than one partition.
