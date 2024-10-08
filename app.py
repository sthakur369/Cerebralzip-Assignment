import os
import pytesseract
import fitz  # PyMuPDF
import numpy as np
import cv2
import re
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count
import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import re
import pandas as pd

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pd.set_option('display.max_colwidth', None)  # To display full extracted text

# Function to convert PDF page to image


def pdf_page_to_image(page):
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return img

# Function to preprocess the image


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

# Function to extract text from a single PDF


def extract_text_from_pdf(pdf_path, lang='eng'):
    print(f"Processing {pdf_path} with language: {lang}")

    try:
        doc = fitz.open(pdf_path)
        text = ''

        # Loop through each page and convert to image
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            img = pdf_page_to_image(page)

            # Preprocess the image
            preprocessed_img = preprocess_image(img)

            # Perform OCR on the preprocessed image with custom configuration
            custom_config = r'--oem 3 --psm 6'
            page_text = pytesseract.image_to_string(
                preprocessed_img, lang=lang, config=custom_config)

            # Clean the extracted text
            cleaned_text = clean_text(page_text)
            text += cleaned_text + "\n"  # Accumulate text from all pages

        return text.strip() if text else 'No text found'

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Function to clean the extracted text


def clean_text(text):
    # Remove unwanted characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Wrapper function to process each PDF


def process_pdf(pdf_path, lang='eng'):
    text = extract_text_from_pdf(pdf_path, lang=lang)
    return {
        'folder': os.path.basename(os.path.dirname(pdf_path)),
        'filename': os.path.basename(pdf_path),
        'extracted_text': text if text else 'No text found'
    }

# Function to extract text from all PDFs in a folder


def extract_texts_from_folder(folder_path, lang='eng'):
    pdf_paths = [os.path.join(root, file) for root, _, files in os.walk(
        folder_path) for file in files if file.endswith('.pdf')]
    if not pdf_paths:
        print(f"No PDF files found in {folder_path}.")
        return []

    with Pool(min(cpu_count() // 2, len(pdf_paths))) as pool:
        results = pool.starmap(
            process_pdf, [(pdf_path, lang) for pdf_path in pdf_paths])
    return results


if __name__ == '__main__':
    start_time = time.time()
    language_codes = {'bn': 'ben', 'en': 'eng', 'ur': 'urd', 'zh': 'chi_sim'}
    all_data = []

    # Loop through each language folder and process the PDFs in that folder
    for lang, code in language_codes.items():
        folder_path = os.path.join('pdfs', lang)
        print(
            f"Processing PDFs in folder: {folder_path} using language: {code}")
        extracted_data = extract_texts_from_folder(folder_path, lang=code)
        all_data.extend(extracted_data)

        # Optional: Clear memory after each folder
        del extracted_data
        gc.collect()

    # Convert results to DataFrame and display
    df = pd.DataFrame(all_data)
    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
    print(df.shape)

    df.to_csv('extracted_texts.csv', index=False)



model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load the extracted text data from CSV
documents = df['extracted_text'].tolist()

# Step 2: Generate vector embeddings for the documents
document_embeddings = model.encode(documents)

# Normalize the embeddings for cosine similarity
document_embeddings = document_embeddings / \
    np.linalg.norm(document_embeddings, axis=1, keepdims=True)

# Step 3: Create a FAISS index for efficient search
num_vectors = len(document_embeddings)
dim = len(document_embeddings[0])  # Embedding size
faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity

# Add document embeddings to the FAISS index
faiss_index.add(np.array(document_embeddings, dtype=np.float32))


# Step 4: Question-Answering and Summarization Pipelines
qa_pipeline = pipeline('question-answering', model='deepset/xlm-roberta-large-squad2')  # QA pipeline for answering questions
summarizer = pipeline('summarization', model='facebook/mbart-large-50')  # Summarization pipeline for generating summaries



# Function to retrieve relevant documents using FAISS and answer the question
def retrieve_and_answer(query, documents, top_k=5):
    # Encode the query and normalize it
    query_vector = model.encode([query])
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Search for the most relevant documents in FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Extract relevant text from the documents
    relevant_text = []
    context_window_size = 5

    for i, index in enumerate(indices[0]):
        distance = distances[0][i]
        truncated_text = documents[index][:1000]  # Display only first 1000 characters
        relevant_text.append(truncated_text)

    # Combine the relevant text for the QA model
    combined_text = ' '.join(relevant_text)

    # Use QA pipeline to answer the question based on the relevant documents
    if combined_text:
        answer = qa_pipeline({'question': query, 'context': combined_text})
        return answer['answer'], combined_text  # Return both the answer and the relevant text
    else:
        return "No relevant text found for the query.", ""



def chunk_text(text, max_length=1024):
    # Split the text into chunks that fit the model's input length
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))  # Add the current chunk to the list
            current_chunk = [word]  # Start a new chunk

    if current_chunk:  # Add the last chunk
        chunks.append(' '.join(current_chunk))

    return chunks


def clean_text(text):
    # Remove unwanted characters and excess whitespace
    cleaned_text = re.sub(r'[^\w\s,.?!]+', '', text)  # Retain only words and common punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    return cleaned_text.strip()

def generate_summary(relevant_text, max_length=130, min_length=30):
    if relevant_text:
        cleaned_text = clean_text(relevant_text)  # Clean the text
        chunks = chunk_text(cleaned_text)  # Split the text into manageable chunks
        summaries = []

        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return ' '.join(summaries)  # Combine all summaries
    else:
        return "No text available for summarization."
    

# Example usage
query = 'The main lesson of The Alchemist?'

# Retrieve the answer and relevant document text
answer, relevant_text = retrieve_and_answer(query, documents)

# Generate summary of the relevant text
summary = generate_summary(relevant_text)

# Output the results
print(f"Question: {query}")
print(f"Answer: {answer}")
print(f"Summary: {summary}")

