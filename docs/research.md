# MedVision AI — Research Notes

This document collects the research context behind the MedVision AI prototype:
the models, the datasets, the explainability methods, the evaluation metrics,
and the ethical constraints that shape the design. Citations are real,
peer-reviewed or widely-cited primary sources; no citation is invented.

> **Scope note:** MedVision AI is a research/educational prototype. It is not
> a medical device, has no regulatory approval, and must never be used for
> clinical diagnosis or patient triage.

---

## 1. Related work — chest X-ray AI

- **CheXNet** (Rajpurkar et al., 2017) — DenseNet-121 trained on the NIH
  ChestX-ray14 dataset for 14-class chest X-ray classification, reporting
  performance at or above practicing radiologists on several pathologies.
  The DenseNet-121 backbone and the 0.841 AUROC figure referenced in the UI
  come from this paper's published benchmark — they are literature values,
  not measurements of this build. arXiv:1711.05225.
- **CheXpert** (Irvin et al., 2019) — a large chest radiograph dataset with
  automated label extraction and uncertainty handling; a common benchmark for
  multi-label CXR classification. arXiv:1901.07031.
- **MIMIC-CXR** (Johnson et al., 2019) — a large public dataset of chest
  radiographs with free-text reports, subject to a data-use agreement.
  Scientific Data 6:317. DOI:10.1038/s41597-019-0322-0.
- **NIH ChestX-ray14** (Wang et al., 2017) — 112,120 frontal-view X-rays from
  30,805 patients with 14 disease labels. The training pipeline in
  `training/` is built around its CSV layout. CVPR 2017.
- **VinDr-CXR / RSNA Pneumonia Challenge** — additional public benchmarks used
  by the community for localization and pneumonia detection respectively.

## 2. DenseNet-121 as backbone

DenseNet (Huang et al., 2017, CVPR) connects each layer to every other layer
in a feed-forward fashion, which mitigates vanishing gradients and enables
parameter-efficient feature reuse. DenseNet-121's final convolutional block
(`denseblock4`) provides high-level spatial features that are the standard
target layer for Grad-CAM attribution.

## 3. Multi-label classification setup

Chest pathologies co-occur, so the model head is a **sigmoid multi-label
classifier**: each class receives an independent probability in (0, 1),
trained with **BCEWithLogitsLoss** and, optionally, inverse-frequency positive
weights. Evaluation therefore centers on **AUROC / AUPRC** (threshold-free)
plus precision / recall / F1 / sensitivity / specificity at decision
thresholds — accuracy alone is a poor metric for imbalanced multi-label
medical data. This follows the conventions established by CheXNet and the
CheXpert benchmark.

## 4. Explainability

- **Grad-CAM** (Selvaraju et al., 2017, ICCV) — the gradient of a class score
  with respect to the final convolutional feature maps is globally averaged to
  weight each feature channel; the weighted sum is ReLU'd to produce a spatial
  activation map that localizes the evidence for a class.
- **Grad-CAM++** (Chattopadhay et al., 2018, WACV) — uses higher-order
  gradients to weight the feature maps, improving localization of multiple
  instances of a class.
- **Class-agnostic feature activation** — when only a pretrained (ImageNet)
  backbone is available (no fine-tuned head), disease-specific attribution is
  not possible. The engine returns the mean of ReLU'd final feature maps and
  labels it explicitly as backbone attention, not a disease localization.

The heatmaps in the UI are real engine outputs (data-URL PNGs) when the
engine is online; the remaining methods in the explainability explorer are
clearly-marked simulated previews.

## 5. Uncertainty and calibration

- **Temperature scaling** (Guo et al., 2017, ICML "On Calibration of Modern
  Neural Networks") — a single scalar divides the logits to flatten
  overconfident probabilities. The engine applies it (default temperature
  1.0 = identity) and reports raw vs calibrated confidence.
- **Monte-Carlo dropout** (Gal & Ghahramani, 2016, ICML) — stochastic forward
  passes at inference approximate model uncertainty; used when a fine-tuned
  head is loaded. Otherwise an honest margin + quality proxy is reported.

## 6. Medical AI limitations (why this is a prototype)

- Chest X-ray findings are ambiguous: overlap between consolidation,
  atelectasis and effusion is common, so high-probability outputs are still
  not diagnoses.
- Dataset bias: public datasets inherit label noise, demographic skew, and
  subtle technical artifacts; models trained on them do not transfer blindly
  to new scanners or patient populations (Zech et al., 2018, PLoS Medicine —
  "Confounding variables can degrade generalization of performance").
- Explainability maps show *where the network attended*, not *where the
  pathology is*; they require radiologist interpretation.
- No model in this repository has been clinically validated; there is no
  FDA/CE clearance, and no clinical deployment is implied.

## 7. Dataset considerations

- NIH ChestX-ray14, CheXpert and MIMIC-CXR have specific licenses and access
  requirements (MIMIC requires a signed DUA). This repository ships **no
  dataset**; the ingestion/loading code expects the user to obtain data
  legitimately.
- The bundled "sample studies" are procedurally-rendered synthetic images
  clearly labelled `PAT-DEMO-*`; they are not real patients and are not
  training data.

## 8. Ethical considerations

- This system must always present itself as a decision-support / research
  visualization, never as a radiologist replacement.
- No patient-identifiable information is stored; the patient registry is
  synthetic/demo data.
- Predictions, heatmaps and reports are gated by server-side image validation
  and are always accompanied by disclaimers.
- Anyone extending the system toward clinical use must perform formal
  validation, obtain regulatory clearance, and follow applicable data
  protection law — claims of HIPAA/GDPR compliance are only valid once the
  system is actually implemented and verified to that standard.
