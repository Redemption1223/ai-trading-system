@echo off
echo 📦 AGI Trading System - Dependency Installation
echo ==============================================

echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

echo 📚 Installing requirements...
pip install -r requirements.txt

echo 🧠 Installing additional ML packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ✅ Dependency installation complete!
pause