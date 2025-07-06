# Customer Segmentation & AI-Powered Campaign Dashboard

This project demonstrates advanced customer segmentation using clustering, with an interactive Streamlit dashboard for business insights and AI-powered ad campaign generation.

---

## Project Structure

- `customer-segmentation-clustering.ipynb` — Jupyter notebook for data exploration, feature engineering, and clustering.
- `run_clustering.py` — Script to preprocess data and generate clusters for the dashboard.
- `streamlit_app.py` — Streamlit dashboard for interactive segmentation, niche analysis, and campaign generation.
- `marketing_campaign.csv` — Raw marketing campaign dataset.
- `marketing_campaign_with_clusters.csv` — Dataset with cluster labels (generated).
- `requirements.txt` — All Python dependencies.
- `.streamlit/secrets.toml` — Secure API key storage (see below).
- `Epsilon Hackathon Default Image.png` — Fallback image for ad campaigns.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (for AI features)

- Copy `secrets_template.toml` to `.streamlit/secrets.toml`
- Get your API keys:
  - [Groq API key](https://console.groq.com/)
  - [Cloudflare Workers AI key & Account ID](https://developers.cloudflare.com/workers-ai/)
- Paste your keys into `.streamlit/secrets.toml`

### 3. Run Clustering

```bash
python run_clustering.py
```

### 4. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

Or, on Windows, use the one-click demo:

```bash
run_demo.bat
```

---

## Features

- **Customer Segmentation:** Visualize clusters, explore feature distributions, and identify niche segments.
- **Niche Analysis:** Deep-dive into the smallest, most unique customer group for micro-campaigns.
- **AI Ad Campaigns:** Instantly generate tailored ad copy and images for your niche segment using Groq and Cloudflare Workers AI.
- **Channel Strategy:** Get recommended marketing channels for your niche audience.
- **Secure API Key Management:** All secrets are managed via Streamlit's secrets system and never committed to version control.
- **Fallback Image:** If AI image generation fails, a default campaign image is shown.

---

## Data Flow

```
marketing_campaign.csv → run_clustering.py → marketing_campaign_with_clusters.csv → streamlit_app.py
```

---

## Troubleshooting

- **No clustering results?** Run `python run_clustering.py` first.
- **Dependency errors?** Run `pip install -r requirements.txt`.
- **API key errors?** Ensure `.streamlit/secrets.toml` is present and filled out.
- **No AI features?** The dashboard works without API keys, but ad/image generation will be disabled.

---

## Security

- API keys are stored in `.streamlit/secrets.toml` (excluded from git).
- Never commit your API keys or sensitive data.
- See `secrets_template.toml` for setup instructions.

---

## Credits

Built for the Epsilon Hackathon.

Data: [Marketing Campaign Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

AI: Groq, Cloudflare Workers AI
