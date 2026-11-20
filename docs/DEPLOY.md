# Deployment guide

I cannot log in to your accounts, so these are exact steps for you to run. Everything below was prepared in the
repo; the Streamlit app, API and tests were verified in a clean virtual environment built only from
`requirements.txt`. **The Dockerfile and the cloud deployments themselves have NOT been run by me.**

## 0. Check locally first
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## 1. Push to GitHub (~5 min)
1. Create an empty repo on github.com (e.g. `agrisense`), without a README.
2. From the project folder:
```bash
git init
git add .
git commit -m "AgriSense: crop yield prediction and risk analysis"
git branch -M main
git remote add origin https://github.com/<your-username>/agrisense.git
git push -u origin main
```
Notes: the largest tracked file is the 34 MB district boundary GeoJSON (GitHub's limit is 100 MB, so no Git LFS needed).
`data/processed/agrisense.db`, `model_table.parquet` and `exports/yield_history.csv` are git-ignored because
`make data features` and `python -m src.exports` rebuild them; the app does not need them.

## 2. Streamlit Community Cloud (free, ~5 min)
1. Go to share.streamlit.io and sign in with GitHub.
2. **New app** → pick your repo, branch `main`, main file path **`app/app.py`**.
3. **Advanced settings → Python version: 3.12** (the pinned versions in `requirements.txt` were tested on 3.12).
4. Deploy. First build takes a few minutes; you get a URL like `https://<name>.streamlit.app`.
5. If the build fails on a pinned version, loosen that single pin in `requirements.txt` and push again.

## 3. Hugging Face Spaces (Docker, alternative)
1. huggingface.co → New Space → SDK **Docker** (blank).
2. Clone the Space repo, copy the project files into it, and put this header at the very top of the Space's `README.md`:
```
---
title: AgriSense
emoji: 🌾
sdk: docker
app_port: 8501
---
```
3. `git push` to the Space. Spaces run containers as a non-root user; if the build complains about permissions
   or write access, add `RUN useradd -m -u 1000 user` and `USER user` to the Dockerfile (untested).

## 4. Docker locally (untested by me: no Docker in my environment)
```bash
docker build -t agrisense .
docker run -p 8501:8501 agrisense                                              # app
docker run -p 8000:8000 agrisense uvicorn api.main:app --host 0.0.0.0 --port 8000   # API
```

## 5. Optional: host the API (Render / Railway / Fly.io)
Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, install command: `pip install -r requirements.txt`.

## 6. After deploying
- Put the live URL at the top of `README.md` and in your GitHub repo's "About" box.
- Add 2-3 screenshots (`reports/figures/risk_map.png`, an app screenshot) to the README.
- Turn on the GitHub Actions workflow (already in `.github/workflows/ci.yml`) and add the badge.
