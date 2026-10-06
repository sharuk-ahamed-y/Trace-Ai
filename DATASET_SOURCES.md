# Dataset Source Strategy & Compliance Policy

## 1. Research Objectives & Legal Compliance

For the Image Provenance Analysis research project, we require a dataset containing controlled manipulation histories, ground-truth provenance directed graphs, parent-child edge relationships, transformation metadata, and distractor images.

To ensure strict legal and ethical compliance in academic research:
1. We **do not** bypass access controls or registration requirements of gated research datasets.
2. We **do not** misbrand synthetic benchmark data as official benchmark datasets (e.g. NIST MFC-Prov).
3. We construct a transparent, research-grade synthetic provenance dataset derived from legally usable source images, named **Synthetic Image Provenance Graph Dataset (SIPGD)** (or **Image Provenance Research Dataset - IPRD**).

---

## 2. Public & Permissible Source Image Repositories

The following sources provide legally usable images for research and dataset construction:

| Repository / Source | License / Terms | Intended Usage in SIPGD | Download Mechanism |
| :--- | :--- | :--- | :--- |
| **Wikimedia Commons** | CC0 / Public Domain / CC-BY | Source root images for provenance trees | Direct HTTPS / Wikimedia API |
| **Unsplash / Pexels** | Unsplash License / Pexels License (Free for commercial/non-commercial use) | Source root images | Public API / Direct URLs |
| **COCO (Common Objects in Context)** | CC-BY 4.0 | Diverse object/scene source images | Direct download (`http://images.cocodataset.org/`) |
| **Procedural Synthetic Sources** | Public Domain (Generated) | Scalable high-variance root images | Local generator (`create_sample_images.py`) |

---

## 3. Restricted & Gated Dataset Audit

An audit of restricted academic datasets was conducted. Access procedures are recorded below in accordance with research ethics:

### NIST Media Forensics Challenge (MFC-Prov / OpenMFC)
- **Status**: Restricted Access / Registration Required.
- **Official Source**: [https://mfc.nist.gov/](https://mfc.nist.gov/)
- **Access Terms**: Requires formal data usage agreement (DUA) submission to NIST.
- **Compliance Policy**: Access restrictions are strictly respected. No unauthorized mirrors or bypassed access methods are used.

### Reddit Photoshop Battles (PS-Battles)
- **Status**: Public annotations; image URLs require scraping/hydration.
- **Official Source**: Academic repositories / GitHub (e.g., PS-Battles dataset release).
- **Compliance Policy**: Used where raw image access is legally permitted; otherwise supplemented by synthetic controlled pipelines.

---

## 4. Synthetic Image Provenance Graph Dataset (SIPGD) Strategy

Due to the lack of freely redistributable, unencumbered ground-truth provenance graphs with full transformation parameter tracking, we generate **SIPGD**:
- **Root Generation**: Derived from verified permissible source images (procedural + public domain images).
- **Multi-Level Graphs**: Synthesized with realistic multi-step edits (crop, resize, compression, blur, brightness, contrast, sharpening, rotation, saturation, noise, perspective, degradation).
- **Distractors**: Unrelated image identities mixed at ratios (1:1, 1:2, 1:5, 1:10) to benchmark graph pruning and MST reconstruction.
- **Provenance Ground Truth**: Fully tracked edge parameters, generation numbers, root IDs, and split by original source identity (0% data leakage).
