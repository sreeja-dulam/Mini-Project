# SATELLITE IMAGE PREPROCESSING AND WATER BODY DETECTION TOOL

This project offers an end-to-end Python-based system for water body detection and satellite image preprocessing. It automatically tiles high-resolution satellite images, creates NDWI-based binary water body masks, and conducts real-time segmentation with a U-Net model. A Streamlit interface allows for simple folder upload, processing, and result visualization.

## Features
- **Automated Preprocessing**: Converts large satellite images into 256x256 tiles and applies NDWI for water body detection.  
- **Real-Time Water Body Prediction**: Uses a trained lightweight U-Net model to calculate water body coverage percentage.  
- **Streamlit Web Interface**: Allows users to upload zipped image folders and view/download results interactively.  
- **Ready Dataset Creation**: Outputs training-ready datasets with minimal effort.

## Installation
Ensure the following are already installed:
- Visual Studio Code (VS Code)  
- Python 3.10+  

**Note**: The virtual environment (`venv/`) is already included in the project.

### Steps
1. Open the project folder in VS Code  
2. Activate the virtual environment:
   **For Windows:**
   ```bash
   venv\Scripts\activate
   ```
  **For Linux/macOS:**
   ```bash
    source venv/bin/activate
   ```
3. How to Run the Tool
Run the following command in the terminal:
```bash
streamlit run app.py
```
This opens the app at: [http://localhost:8501](http://localhost:8501)

**For preprocessing**: Upload a `.zip` file containing multiple `.tiff` satellite images.  
**For prediction**: Upload a single `.tiff` image.

---

## Outputs

- A downloadable ZIP containing preprocessed image–mask pairs.  
- Real-time display of water body segmentation with calculated coverage percentage.

---

## Usage & Future Enhancements

This tool currently works as a prototype on a small dataset (30 `.tiff` images). Planned future upgrades include:

- Integration with real-time satellite data sources  
- Time-based change detection of water bodies  
- Support for large-scale datasets with GPU and cloud-based processing
