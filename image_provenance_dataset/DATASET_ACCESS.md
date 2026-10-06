# DATASET_ACCESS.md — External Provenance Dataset Audit & Access Record

This document records the access status and formal acquisition guidelines for established image provenance datasets in the computer vision and multimedia forensics research community.

---

## 1. NIST MFC-Prov (Media Forensics Challenge — Provenance Dataset)

- **Official Portal**: [https://mfc.nist.gov/](https://mfc.nist.gov/)
- **Provider**: National Institute of Standards and Technology (NIST)
- **Access Protocol**:
  1. Access requires formal registration and execution of a NIST Data Use Agreement (DUA).
  2. Researchers must request access through an institutional email address and obtain approval from the NIST MFC evaluation team.
- **Compliance Record**:
  - Direct public downloading without prior authorization is prohibited by NIST terms.
  - We **DO NOT** attempt to bypass, mirror, or unauthorizedly fetch MFC-Prov data.
  - All experiments in this repository requiring redistributable data use our self-contained **Synthetic Image Provenance Graph Dataset (SIPGD)**.

---

## 2. PS-Battles (Reddit Photoshop Battles Dataset)

- **Official Repository**: [https://github.com/Cross-Media-Data-Intelligence/PS-Battles](https://github.com/Cross-Media-Data-Intelligence/PS-Battles)
- **Provider**: Academic release (Heller et al.)
- **Access Protocol**:
  - Annotations and graph structures are published openly.
  - Original images are hosted on Imgur/Reddit and must be downloaded via web requests respecting domain rate limits and robot.txt terms.
- **Compliance Record**:
  - Where image links are expired or restricted, SIPGD provides a fully reproducible alternative with 100% verified parent-child ground truth.

---

## 3. NIMBLE / Nimble Challenge Datasets

- **Official Portal**: Defense Advanced Research Projects Agency (DARPA) MediFor Program
- **Access Protocol**:
  - Controlled release subject to DARPA/NIST distribution guidelines.
- **Compliance Record**:
  - Requires approved participation in MediFor / SemaFor programs. Recorded as gated research data.

---

## 4. Distinction of Our Dataset

The dataset generated in this repository is explicitly named **Synthetic Image Provenance Graph Dataset (SIPGD)** (or **Image Provenance Research Dataset - IPRD**).

- It is **NOT** MFC-Prov or any official NIST challenge dataset.
- It is a synthetic provenance graph dataset generated from real and procedural source images with fully tracked multi-generational edit operations.
- This distinction ensures transparency for paper submission and peer review.
