# Speaker-Independent Parkinson's Disease Detection from Speech Using Self-Supervised Representations

M.Tech Project (MTP-I) · Department of Electrical Engineering, IIT Kharagpur · November 2026
Student: Rwik Dey · Supervisor: Prof. Anirban Mukherjee

## Overview

Speech-based detection of Parkinson's disease (PD) is often reported at 90–99% accuracy. Many of these results, however, let recordings of the same person appear in both the training and the test data, so a model can succeed by recognising the speaker rather than the disease. This project re-evaluates PD detection with strictly **speaker-independent** testing, and builds a detector that holds up under that protocol.

The work has three parts:

1. **Evaluation audit.** Published pipelines are reproduced and re-tested with every speaker kept on one side of the split. A non-speech control checks whether a dataset's recording conditions alone reveal the label.
2. **Representation study.** Seven frozen self-supervised speech encoders (XLS-R, WavLM, HuBERT, wav2vec 2.0, MMS, Whisper) are compared against clinical acoustic features and end-to-end fine-tuning.
3. **Proposed models.** Two architectures, NORM-PD and STRIDE-PD, are tested component by component against matched controls. The final model is **CF-probe**: a linear probe on an XLS-R layer chosen inside each fold, combined with a cross-fitted speaker-level decision rule.

## Key results

All results are speaker-level and speaker-independent (leave-one-speaker-out on MDVR-KCL, 37 speakers).

| System | Balanced accuracy | AUC |
|---|---|---|
| eGeMAPS + SVM (clinical-feature baseline) | 0.851 | 0.893 |
| Fine-tuned WavLM base+ | 0.847 ± 0.005 | 0.882 |
| Frozen XLS-R, layer chosen in-fold | 0.875 | 0.926 |
| **CF-probe (proposed)** | **0.903 ± 0.020** | **0.912** |

- **Specificity 1.00:** CF-probe flags no healthy speaker. Its three errors are mild patients (Hoehn and Yahr stage 2) with near-normal speech.
- **Robust to device change:** under a simulated change of recording device it keeps 0.894 balanced accuracy.
- **Spontaneous speech:** on dialogue, a layer-attention head performs best (AUC 0.942).
- **Leakage:** with random chunk splits, the same clinical features score 1.000. Under speaker-independent testing they score 0.78–0.85, so much of the published accuracy reflects speaker identity.
- **Recording confound:** on the Italian PVS corpus, a model trained only on the pauses and background between words reaches 0.979 balanced accuracy, as high as one trained on speech. English MDVR-KCL passes the same check (0.63).

## Datasets

| Dataset | Language | Speakers | Use |
|---|---|---|---|
| [MDVR-KCL](https://zenodo.org/records/2867216) | English | 37 (16 PD, 21 HC) | Primary benchmark: read passage and dialogue |
| Figshare sustained vowels (Iyer et al., 2023) | – | 81 (40 PD, 41 HC) | Second recording condition |
| [Italian Parkinson's Voice and Speech](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech) | Italian | 45 with a read passage | Second language, confound check |
| UCI Parkinson's (Sakar et al., 2019) | Turkish | 252 | Reproduction of the heatmap ResNet base paper |
| [BenSParX](https://github.com/riadEDU/BenSParX) | Bengali | 120 | Evaluation-validity case study |

The datasets are not redistributed here; download them from their original sources under their own licences.

## Evaluation protocol

- **Speaker-independent splits:** leave-one-speaker-out, or speaker-grouped folds; no speaker appears in both training and test data.
- **Speaker-level decisions:** 4-second chunk scores are averaged into one probability per speaker.
- **In-fold tuning only:** scaling, layer choice and thresholds are fitted on training speakers.
- **Reporting:** balanced accuracy and AUC, with bootstrap confidence intervals over speakers, several random seeds for neural models, and McNemar and paired-bootstrap tests for comparisons.
- **Matched controls:** every proposed component is compared against a control that removes only its mechanism.

## Reproducing the experiments

Experiments were run in Google Colab on an NVIDIA T4 GPU. Encoder outputs for every 4-second chunk were cached once, so all experiments reuse identical embeddings. Main dependencies are Python 3, PyTorch, Hugging Face Transformers, scikit-learn, openSMILE and Praat-Parselmouth.

## Citation

```bibtex
@mastersthesis{dey2026pdspeech,
  author = {Dey, Rwik},
  title  = {Speaker-Independent {Parkinson}'s Disease Detection from Speech Using Self-Supervised Representations},
  school = {Indian Institute of Technology Kharagpur},
  type   = {M.Tech Project Report},
  year   = {2026}
}
```
