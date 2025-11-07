# Deployment Guide: Fraud Detection AI

Complete guide for deploying the fraud detection system to various platforms.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Streamlit Cloud](#streamlit-cloud)
5. [Render](#render)
6. [Railway](#railway)
7. [HuggingFace Spaces](#huggingface-spaces)
8. [AWS Deployment](#aws-deployment)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Optional
- Docker and Docker Compose (for containerized deployment)
- GitHub account (for cloud deployments)
- Cloud platform account (Streamlit Cloud, Render, etc.)

---

## Local Deployment

### Option 1: Standard Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd fraud_detection_hackathon_pack

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Train the model
python src/train.py

# 6. Run the application
streamlit run app/streamlit_app.py
```

### Option 2: Using Scripts

**Windows:**
```powershell
# Train model
.\scripts\train.ps1

# Run application
.\scripts\run.ps1
```

**Linux/Mac:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train model
./scripts/train.sh

# Run application
./scripts/run.sh
```

### Accessing the Application

- Open browser to: `http://localhost:8501`
- The app will auto-reload on code changes

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/assets:/app/assets \
  fraud-detection:latest

# Run with custom port
docker run -p 8080:8501 fraud-detection:latest
```

### Docker Environment Variables

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  fraud-detection:latest
```

---

## Streamlit Cloud

### Step-by-Step Guide

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Configure:
     - **Main file path**: `app/streamlit_app.py`
     - **Python version**: 3.10
   - Click "Deploy"

3. **Configuration** (Optional)
   - Create `.streamlit/config.toml`:
     ```toml
     [theme]
     primaryColor = "#FF6B6B"
     backgroundColor = "#FFFFFF"
     secondaryBackgroundColor = "#F0F2F6"
     textColor = "#262730"
     font = "sans serif"
     
     [server]
     maxUploadSize = 200
     enableCORS = false
     enableXsrfProtection = true
     ```

4. **Secrets Management** (if needed)
   - In Streamlit Cloud dashboard, go to "Advanced settings"
   - Add secrets in TOML format:
     ```toml
     [secrets]
     api_key = "your-api-key"
     ```

### Update Deployment

```bash
# Push changes
git add .
git commit -m "Update feature"
git push

# Streamlit Cloud auto-deploys on push
```

---

## Render

### Deployment Steps

1. **Create `render.yaml`** (in project root):
   ```yaml
   services:
     - type: web
       name: fraud-detection
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
       envVars:
         - key: PYTHON_VERSION
           value: 3.10.0
   ```

2. **Deploy via Render Dashboard**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Configure:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - Click "Create Web Service"

3. **Custom Domain** (Optional)
   - In Render dashboard, go to "Settings"
   - Add custom domain
   - Update DNS records

---

## Railway

### Deployment Steps

1. **Install Railway CLI** (Optional)
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy via Dashboard**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway auto-detects Python app

3. **Configure Settings**
   - Add environment variable:
     ```
     PORT = 8501
     ```
   - Custom start command (if needed):
     ```
     streamlit run app/streamlit_app.py --server.port=$PORT
     ```

4. **Deploy via CLI** (Alternative)
   ```bash
   railway init
   railway up
   ```

---

## HuggingFace Spaces

### Deployment Steps

1. **Create Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select SDK: **Streamlit**
   - Name your space

2. **Configure Space**
   - Create `README.md` header:
     ```yaml
     ---
     title: Fraud Detection AI
     emoji: 🛡️
     colorFrom: blue
     colorTo: red
     sdk: streamlit
     sdk_version: 1.26.0
     app_file: app/streamlit_app.py
     pinned: false
     ---
     ```

3. **Upload Files**
   - Option 1: Git push
     ```bash
     git remote add hf https://huggingface.co/spaces/<username>/<space-name>
     git push hf main
     ```
   - Option 2: Web upload
     - Drag and drop files in browser

4. **Requirements**
   - Ensure `requirements.txt` is in root
   - HuggingFace automatically installs dependencies

---

## AWS Deployment

### Option 1: EC2 Instance

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx -y

# 4. Clone repository
git clone <your-repo-url>
cd fraud_detection_hackathon_pack

# 5. Setup application
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Run with systemd service
sudo nano /etc/systemd/system/fraud-detection.service
```

**Service file**:
```ini
[Unit]
Description=Fraud Detection Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fraud_detection_hackathon_pack
Environment="PATH=/home/ubuntu/fraud_detection_hackathon_pack/venv/bin"
ExecStart=/home/ubuntu/fraud_detection_hackathon_pack/venv/bin/streamlit run app/streamlit_app.py --server.port=8501

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl daemon-reload
sudo systemctl start fraud-detection
sudo systemctl enable fraud-detection
```

### Option 2: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 fraud-detection

# Create environment
eb create fraud-detection-env

# Deploy
eb deploy

# Open in browser
eb open
```

### Option 3: AWS ECS (Docker)

1. Push Docker image to ECR
2. Create ECS task definition
3. Create ECS service
4. Configure load balancer

---

## Environment Variables

### Common Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false

# Application Configuration
MODEL_PATH=models/rf_fraud_model.joblib
ASSETS_PATH=assets/
LOG_LEVEL=INFO
```

### Setting Variables

**Local (.env file)**:
```bash
# Create .env file
echo "MODEL_PATH=models/rf_fraud_model.joblib" > .env
```

**Docker**:
```bash
docker run -e MODEL_PATH=/app/models/rf_fraud_model.joblib fraud-detection
```

**Cloud Platforms**:
- Use platform's environment variable settings in dashboard

---

## Custom Domain Setup

### Streamlit Cloud
- Contact Streamlit support for custom domain

### Render/Railway
1. Go to Settings → Custom Domain
2. Add your domain
3. Update DNS records:
   ```
   Type: CNAME
   Name: www
   Value: <your-app>.onrender.com
   ```

### AWS Route 53
1. Create hosted zone
2. Add A record pointing to load balancer
3. Configure SSL certificate (AWS Certificate Manager)

---

## SSL/HTTPS Configuration

### Streamlit Cloud
- Automatic HTTPS (included)

### Render/Railway
- Automatic HTTPS (included)

### Self-Hosted (Nginx)

```nginx
# /etc/nginx/sites-available/fraud-detection
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Performance Optimization

### Production Settings

**config.toml**:
```toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[runner]
magicEnabled = false
fastReruns = true

[client]
showErrorDetails = false
```

### Caching

Use Streamlit caching effectively:
```python
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

@st.cache_data
def load_data(file):
    return pd.read_csv(file)
```

### Resource Limits

**Docker**:
```bash
docker run --memory="2g" --cpus="2" fraud-detection
```

---

## Monitoring & Logging

### Application Logs

**View logs**:
```bash
# Docker
docker logs fraud_detection_app

# Systemd
sudo journalctl -u fraud-detection -f

# Streamlit Cloud
# Check dashboard → Manage app → Logs
```

### Health Checks

**Docker Compose** (already configured):
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Troubleshooting

### Common Issues

**1. Model not found**
```bash
# Solution: Train the model first
python src/train.py
```

**2. Port already in use**
```bash
# Solution: Use different port
streamlit run app/streamlit_app.py --server.port=8502
```

**3. Dependencies not installing**
```bash
# Solution: Upgrade pip
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Memory errors**
```bash
# Solution: Increase Docker memory or use smaller batch sizes
docker run --memory="4g" fraud-detection
```

**5. Streamlit not accessible externally**
```bash
# Solution: Bind to 0.0.0.0
streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

### Debug Mode

```bash
# Enable debug logging
streamlit run app/streamlit_app.py --logger.level=debug
```

---

## Scaling Strategies

### Horizontal Scaling

**Load Balancer + Multiple Instances**:
```yaml
# docker-compose-scaled.yml
version: '3.8'
services:
  fraud-detection:
    build: .
    deploy:
      replicas: 3
    ports:
      - "8501-8503:8501"
```

### Vertical Scaling

- Increase container resources
- Use more powerful instance type
- Optimize model (reduce tree count, feature selection)

---

## Backup & Recovery

### Backup Critical Files

```bash
# Models
cp models/*.joblib backups/

# Configuration
cp .streamlit/config.toml backups/

# Assets
tar -czf assets_backup.tar.gz assets/
```

### Automated Backups (Cron)

```bash
# Add to crontab
0 2 * * * /path/to/backup_script.sh
```

---

## Security Best Practices

1. **Never commit secrets**
   - Use `.env` files (add to `.gitignore`)
   - Use platform secret managers

2. **Update dependencies regularly**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Use HTTPS in production**
   - Enable SSL certificates
   - Redirect HTTP to HTTPS

4. **Implement rate limiting**
   - Use Nginx/CloudFlare
   - Add application-level limits

5. **Validate inputs**
   - Check file types
   - Limit file sizes
   - Sanitize user inputs

---

## Next Steps

After deployment:
1. Test all features thoroughly
2. Monitor performance and logs
3. Set up alerts for errors
4. Configure auto-scaling (if needed)
5. Plan for model updates/retraining

---

## Support

For deployment issues:
- Check [Streamlit documentation](https://docs.streamlit.io)
- Visit platform-specific help centers
- Open GitHub issue for bugs

---

**Good luck with your deployment! 🚀**
