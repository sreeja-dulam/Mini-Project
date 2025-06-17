# SATELLITE IMAGE PREPROCESSING AND WATER BODY DETECTION TOOL

This project offers an end-to-end Python-based system for water body detection and satellite image preprocessing. It automatically tiles high-resolution satellite images, creates NDWI-based binary water body masks, and conducts real-time segmentation with a U-Net model. A Streamlit interface allows for simple folder upload, processing, and result visualization.

## 🚀 Features
- **Automated Preprocessing**: Converts large satellite images into 256x256 tiles and applies NDWI for water body detection.  
- **Real-Time Water Body Prediction**: Uses a trained lightweight U-Net model to calculate water body coverage percentage.  
- **Streamlit Web Interface**: Allows users to upload zipped image folders and view/download results interactively.  
- **Ready Dataset Creation**: Outputs training-ready datasets with minimal effort.

## 🛠️ Installation
Ensure the following are already installed:
- Visual Studio Code (VS Code)  
- Python 3.10+  

⚠️ **Note**: The virtual environment (`venv/`) is already included in the project.

### 🔧 Steps
1. Open the project folder in VS Code  
2. Activate the virtual environment:

   **For Windows:**
   ```bash
   venv\Scripts\activate
