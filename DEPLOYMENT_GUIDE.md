# 🚀 Deployment Guide

## Option 1: Railway (EASIEST - Recommended) ⭐

Railway can deploy both your Streamlit frontend and FastAPI backend easily.

### Steps:

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Prepare Files**
   - Already done! You have `requirements.txt` and the code is ready.

3. **Create `railway.json` configuration** (I'll create this)

4. **Deploy to Railway:**
   - Push your code to GitHub
   - In Railway, click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect and deploy

5. **Set Environment Variables:**
   - In Railway, go to your service → Variables
   - Add: `PORT=8000` (for API)
   - Add: `STREAMLIT_SERVER_PORT=8501` (for Streamlit)

---

## Option 2: Streamlit Community Cloud (EASIEST for Streamlit only)

**Note:** This works if you make the API optional or embed the logic directly in Streamlit.

### Steps:

1. **Create Streamlit account**
   - Go to https://share.streamlit.io
   - Sign up with GitHub

2. **Create `.streamlit/config.toml`** (I'll create this)

3. **Update API_URL to be optional** (I'll help with this)

4. **Deploy:**
   - Push to GitHub
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repo and set main file to `src/ui/dashboard.py`

---

## Option 3: Hugging Face Spaces (Also Easy)

### Steps:

1. **Create Hugging Face account**
   - Go to https://huggingface.co
   - Sign up

2. **Create Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" SDK
   - Name it (e.g., "cryptoguard-mlops")

3. **Upload files:**
   - Upload your code via Git or web interface

---

## Recommendation: **Railway** 🏆

Railway is easiest because:
- ✅ Handles both Streamlit + FastAPI automatically
- ✅ No complex configuration needed
- ✅ Free tier available
- ✅ Auto-deploys on git push
- ✅ Built-in environment variables

Let me create the necessary configuration files for Railway deployment!

