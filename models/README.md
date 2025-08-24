# 📂 Models

This directory contains serialized model artifacts used for **inference**.

## Contents

- **`pipeline_v2_300k.joblib`**  
  A `scikit-learn` pipeline that includes preprocessing steps and a trained classifier.

## Notes

- Ensure that package versions in **`requirements.txt`** remain aligned with the artifact’s training environment.
- This is especially important for **scikit-learn**, as version mismatches may cause deserialization issues.
