@echo off
echo 🚀 AGI Trading System - Environment Setup
echo ========================================

echo 📦 Installing Python packages...
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo 📚 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"

echo 📁 Creating database directories...
if not exist "database" mkdir database
if not exist "logs" mkdir logs  
if not exist "models" mkdir models
if not exist "models\neural_models" mkdir models\neural_models
if not exist "models\pattern_models" mkdir models\pattern_models
if not exist "models\saved_models" mkdir models\saved_models

echo 📄 Creating log files...
echo. > logs\system.log
echo. > logs\trading.log
echo. > logs\errors.log
echo. > logs\performance.log

echo 🗃️ Initializing database files...
echo. > database\price_data.db
echo. > database\signal_history.db
echo. > database\system_metrics.db

echo ✅ Environment setup complete!
pause