# 🚀 Quick Deployment Guide

## ⭐ EASIEST: Railway (Recommended)

### Why Railway?
- ✅ Deploys both Streamlit + FastAPI automatically
- ✅ Free tier available
- ✅ Auto-deploys on git push
- ✅ No complex setup needed

### Steps:

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Streamlit and deploys! 🎉

3. **Get your URL:**
   - Railway provides a URL like: `https://your-app.railway.app`
   - Share this URL!

---

## Alternative: Streamlit Community Cloud

### Steps:

1. **Push to GitHub** (same as above)

2. **Deploy:**
   - Go to https://share.streamlit.io
   - Sign up with GitHub
   - Click "New app"
   - Select repo: `YOUR_USERNAME/CryptoGuard-MLOps`
   - Main file: `src/ui/dashboard.py`
   - Click "Deploy"

3. **Note:** API features won't work unless you also deploy the FastAPI backend separately.

---

## Alternative: Hugging Face Spaces

1. **Create Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" SDK
   - Name: `cryptoguard-mlops`

2. **Upload code:**
   - Use Git: `git clone https://huggingface.co/spaces/YOUR_USERNAME/cryptoguard-mlops`
   - Copy your files
   - Push back

---

## Environment Variables (if needed)

For Railway, you can set these in the dashboard:
- `API_URL` - If deploying API separately
- `PORT` - Auto-set by Railway

---

## Troubleshooting

**Issue: "Module not found"**
- Make sure all dependencies are in `requirements.txt`
- Railway will install them automatically

**Issue: "Port already in use"**
- Railway sets `PORT` automatically
- The `Procfile` handles this

**Issue: "API not connecting"**
- Check if `API_URL` environment variable is set correctly
- Or deploy API as separate service on Railway

---

## Recommendation

**Use Railway** - It's the easiest for full-stack ML apps like yours! 🏆

