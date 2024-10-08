# Multilingual PDF RAG System

## Background

Needs to develop a Retrieval-Augmented Generation (RAG) system capable of processing multilingual PDFs, extracting information, and providing summaries and answers to questions based on the content. The system should handle various languages including Hindi, English, Bengali, and Chinese, and be able to process both scanned and digital PDFs.

## Requirements

Before running the application, you need to set up a Python virtual environment and install the required dependencies.

### Setup Instructions

1. **Create a Virtual Environment**

   Navigate to your project directory in the terminal and create a virtual environment by running:

   ```venv\Scripts\activate```

2. **Install Dependencies**

    ```pip install -r requirements.txt```

3. **Usage**

    ```python app.py```


## Additional Information

### Tesseract OCR Configuration

To utilize the Optical Character Recognition (OCR) capabilities of this system, you must configure the Tesseract OCR executable path. In the code, this is done with the following line:

```pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'```


