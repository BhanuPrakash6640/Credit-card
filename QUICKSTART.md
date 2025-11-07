# ⚡ Quick Start Guide - Fraud Detection AI

Get up and running in **under 5 minutes**!

---

## 🎯 Fastest Path to Running

### Option 1: Use Sample Data (No Dataset Required) ⭐

```bash
# 1. Setup environment (one-time)
# Windows:
.\setup.ps1
# Linux/Mac:
chmod +x setup.sh && ./setup.sh

# 2. Run the app
# Windows:
.\scripts\run.ps1
# Linux/Mac:
./scripts/run.sh

# 3. Open browser to http://localhost:8501
# 4. Click "Use Sample Data" button
# 5. Explore the results!
```

**Total time: 3 minutes** ⚡

---

### Option 2: Docker (Even Faster!) 🐳

```bash
# One command to rule them all:
docker-compose up --build

# Open browser to http://localhost:8501
# Click "Use Sample Data"
```

**Total time: 2 minutes** 🚀

---

## 📊 To Train Your Own Model

### Prerequisites
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root.

### Train Model

```bash
# Windows:
.\scripts\train.ps1

# Linux/Mac:
./scripts/train.sh

# Or directly:
python src/train.py
```

**Training time: 2-3 minutes**

This will:
- ✅ Engineer 40+ features
- ✅ Apply SMOTE balancing
- ✅ Train Random Forest (300 trees)
- ✅ Optimize threshold
- ✅ Generate metrics & plots
- ✅ Save model to `models/`

---

## 🎨 What You'll See

### Dashboard Features
1. **Upload CSV** or click **"Use Sample Data"**
2. View **Fraud Alert** with count of suspicious transactions
3. Explore **4 tabs**:
   - 🚨 Flagged Transactions (sorted by risk)
   - 📊 Visualizations (charts & analytics)
   - 📋 All Transactions (filterable table)
   - 💾 Export (download CSV results)

### Navigation Pages
- **🏠 Dashboard**: Main fraud detection interface
- **🔍 Explainability**: See why transactions are flagged
- **📈 Model Metrics**: Performance dashboards & curves
- **ℹ️ About**: Project information

---

## 🔧 Troubleshooting

### "Model not found"
**Solution**: Use sample data (no model needed) OR train the model first
```bash
python src/train.py
```

### "Port 8501 already in use"
**Solution**: Use a different port
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### "Module not found"
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

### Docker issues
**Solution**: Check Docker is running
```bash
docker --version
docker-compose --version
```

---

## 📖 Next Steps

### 1. Explore the Documentation
- **README.md**: Complete project overview
- **docs/architecture.md**: System design & ML pipeline
- **docs/deployment.md**: Deploy to cloud platforms
- **docs/slides.md**: Presentation material
- **docs/demo_script.md**: 2-minute demo walkthrough

### 2. Customize the Model
Edit `src/train.py` to:
- Change Random Forest parameters
- Add new features
- Adjust SMOTE settings
- Modify threshold optimization

### 3. Deploy to Cloud
See `docs/deployment.md` for:
- Streamlit Cloud (1-click deploy)
- Render / Railway
- HuggingFace Spaces
- AWS deployment

---

## 🎯 Demo Checklist

Before presenting:
- [ ] Run `streamlit run app/streamlit_app.py`
- [ ] Test "Use Sample Data" button
- [ ] Verify all tabs load correctly
- [ ] Check visualizations appear
- [ ] Test export functionality
- [ ] Review `docs/demo_script.md`

---

## 💡 Key Commands Cheat Sheet

```bash
# Setup
.\setup.ps1                          # Windows setup
./setup.sh                           # Linux/Mac setup

# Run app
.\scripts\run.ps1                    # Windows
./scripts/run.sh                     # Linux/Mac
streamlit run app/streamlit_app.py  # Direct

# Train model
.\scripts\train.ps1                  # Windows
./scripts/train.sh                   # Linux/Mac
python src/train.py                 # Direct

# Docker
docker-compose up                    # Start
docker-compose up -d                 # Start detached
docker-compose logs -f               # View logs
docker-compose down                  # Stop

# Generate sample data
python scripts/generate_sample_data.py
```

---

## 🌐 URLs After Running

- **App**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health

---

## 📞 Getting Help

1. Check `docs/` folder for detailed documentation
2. Review `PROJECT_SUMMARY.md` for complete feature list
3. See `README.md` for comprehensive guide

---

## 🎉 You're Ready!

The system is production-ready and demo-ready. Just run it and impress! 🚀

**Happy fraud detecting!** 🛡️
