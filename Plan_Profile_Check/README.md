# Streamlit Cropped Plan/Profile Checker

This folder is self-contained for Git/Streamlit deployment.

It includes:

```text
app.py
02_Tool/pp_checker/
02_Tool/pp_checker/standards/irc_73_2023.py
02_Tool/pp_checker/standards/irc_sp84_2019.py
04_CV_Work/ml_prototype/
04_CV_Work/plan_profile_curve_table_results/runs/retrain_v2_yolo11n_1280/weights/best.pt
```

The app runs the cropped-variant checker and creates `public_report.html`,
which hides detailed content under:

```text
Hybrid ML corroboration (engineering results preserved)
ML visual candidates by drawing page
```

## Streamlit Cloud

Use this app file:

```text
app.py
```

## Local Run

From this folder:

```powershell
streamlit run app.py
```
