# Agent instructions for SMS Spam Detection project

## Environment
- Python 3.15 at: `C:\Users\Acer\AppData\Local\Python\pythoncore-3.15-64\python.exe` (not on PATH)
- Run commands from: `C:\Users\Acer\Downloads\SMS-Spam-Detection-using-Machine-Learning-main\SMS-Spam-Detection-using-Machine-Learning-main`

## Commands
- Install deps: `python -m pip install -r requirements.txt`
- Train model: `python train_pipeline.py` (produces `spam_pipeline.joblib`)
- Run tests: `python -m pytest test_spam.py -v`
- Streamlit UI: `streamlit run app.py`
- FastAPI API: `uvicorn backend:app --reload`

## Project files
- `train_pipeline.py` — end-to-end training (steps 1-8), saves `spam_pipeline.joblib`
- `preprocess.py` — `clean_text()` used by the pipeline (importable at load time)
- `app.py` — Streamlit app
- `backend.py` — FastAPI API
- `step1_explore.py` … `step8_custom.py` — original step skeletons
- `test_spam.py` — pytest suite (3 tests)
- `spam.csv` — UCI SMS Spam Collection (5,574 messages)
- `spam_pipeline.joblib` — trained pipeline (Multinomial NB)

## Notes
- `app.py` / `backend.py` resolve `spam_pipeline.joblib` relative to the script file, so they work from any directory.
- Best model: Multinomial NB (spam F1 ~89%), chosen because it supports `predict_proba` for confidence scores.
