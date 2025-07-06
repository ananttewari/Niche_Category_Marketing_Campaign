@echo off
echo Starting Customer Segmentation Demo...
echo.

echo Step 1: Running customer clustering...
python run_clustering.py
echo.

echo Step 2: Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the app when done.
echo.
streamlit run streamlit_app.py 