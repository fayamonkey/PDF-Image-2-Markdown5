import streamlit as st
import fitz  # PyMuPDF
import easyocr
import zipfile
from io import BytesIO
import tempfile
import os
import logging
from PIL import Image
import io
import time
import requests
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_PDF_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 10
MAX_PAGE_PIXELS = 1e7  # ~10000x1000 pixels
OCR_TIMEOUT_SECONDS = 30
MODEL_DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout for model download

# Create cache directory
CACHE_DIR = Path(tempfile.gettempdir()) / "easyocr_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["EASYOCR_MODULE_PATH"] = str(CACHE_DIR)

# Model files to check
MODEL_FILES = [
    'craft_mlt_25k.pth',
    'english_g2.pth'
]

def download_with_progress():
    """Initialize EasyOCR with progress tracking"""
    try:
        # Create a temporary reader just to trigger downloads
        with st.spinner("⏳ Downloading OCR models (this may take a few minutes)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def check_files():
                for i, file in enumerate(MODEL_FILES):
                    progress = (i + 1) / len(MODEL_FILES)
                    status_text.text(f"Checking model file: {file}")
                    progress_bar.progress(progress)
                
                # Initialize reader to trigger downloads
                reader = easyocr.Reader(['en'], 
                                      model_storage_directory=str(CACHE_DIR),
                                      download_enabled=True,
                                      verbose=False)
                return reader
            
            # Use a queue to get the result from the thread
            q = queue.Queue()
            download_thread = threading.Thread(target=lambda: q.put(check_files()))
            download_thread.start()
            download_thread.join(timeout=MODEL_DOWNLOAD_TIMEOUT)
            
            if download_thread.is_alive():
                raise TimeoutError("Model download timed out")
            
            reader = q.get_nowait()
            progress_bar.progress(1.0)
            status_text.text("✅ OCR models downloaded successfully!")
            return reader
            
    except Exception as e:
        error_msg = f"Failed to initialize EasyOCR: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("The app will continue without OCR capabilities. Text extraction will still work.")
        return None
    finally:
        # Clean up progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

# Initialize EasyOCR reader with proper error handling
@st.cache_resource
def get_reader():
    try:
        # First check if models are already downloaded
        all_models_exist = all((CACHE_DIR / model_file).exists() for model_file in MODEL_FILES)
        
        if not all_models_exist:
            return download_with_progress()
        else:
            # Models exist, just initialize the reader
            with st.spinner("Initializing OCR engine..."):
                reader = easyocr.Reader(['en'], 
                                      model_storage_directory=str(CACHE_DIR),
                                      download_enabled=False,
                                      verbose=False)
                return reader
                
    except Exception as e:
        error_msg = f"Failed to initialize EasyOCR: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("The app will continue without OCR capabilities. Text extraction will still work.")
        return None

# Initialize reader with status message
st.info("⚙️ Initializing OCR engine...")
reader = get_reader()
if reader:
    st.success("✅ OCR engine initialized successfully!")
else:
    st.warning("⚠️ OCR engine initialization failed. Only text extraction will be available.")

def validate_pdf_size(pdf_bytes):
    """Validate PDF file size"""
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(f"PDF file too large (max {MAX_PDF_SIZE_MB}MB allowed, got {size_mb:.1f}MB)")

def validate_image(image_bytes):
    """Validate image size and format"""
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Image too large (max {MAX_IMAGE_SIZE_MB}MB allowed)")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.format in ['PNG', 'JPEG', 'TIFF']
    except Exception:
        return False

def process_pdf(pdf_bytes):
    try:
        # Initialize variables to track processing
        total_pages = 0
        processed_pages = 0
        errors = []
        warnings = []
        
        # Create a PDF document object
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_document)
        
        # Create markdown content
        markdown_content = []
        markdown_content.append("# PDF Conversion Results\n")
        
        # Process each page
        for page_num in range(total_pages):
            try:
                with st.spinner(f"Processing page {page_num + 1} of {total_pages}..."):
                    page = pdf_document[page_num]
                    markdown_content.append(f"\n## Page {page_num + 1}\n")
                    
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        markdown_content.append(f"\n### Text Content\n\n{text}\n")
                    
                    # Extract images if OCR is available
                    if reader is not None:
                        image_list = page.get_images()
                        if image_list:
                            markdown_content.append("\n### Images and OCR Text\n")
                            
                            for img_index, img in enumerate(image_list):
                                try:
                                    xref = img[0]
                                    base_image = pdf_document.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    
                                    # Create a temporary file for the image
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{base_image['ext']}") as temp_img:
                                        temp_img.write(image_bytes)
                                        temp_img_path = temp_img.name
                                    
                                    try:
                                        # Perform OCR on the temporary file
                                        ocr_results = reader.readtext(temp_img_path)
                                        
                                        if ocr_results:
                                            markdown_content.append(f"\n#### Image {img_index + 1} OCR Results:\n")
                                            for detection in ocr_results:
                                                text = detection[1]
                                                markdown_content.append(f"- {text}\n")
                                        else:
                                            warnings.append(f"No text found in image {img_index + 1} on page {page_num + 1}")
                                    finally:
                                        # Clean up the temporary file
                                        try:
                                            os.unlink(temp_img_path)
                                        except Exception:
                                            pass
                                            
                                except Exception as e:
                                    error_msg = f"Error processing image {img_index + 1} on page {page_num + 1}: {str(e)}"
                                    errors.append(error_msg)
                                    logger.error(error_msg)
                    else:
                        warnings.append(f"OCR processing skipped for page {page_num + 1} - OCR engine not available")
                
                processed_pages += 1
                # Update progress
                progress = processed_pages / total_pages
                st.progress(progress)
                
            except Exception as e:
                error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Add processing summary
        markdown_content.append("\n## Processing Summary\n")
        markdown_content.append(f"- Total pages: {total_pages}\n")
        markdown_content.append(f"- Successfully processed pages: {processed_pages}\n")
        
        if warnings:
            markdown_content.append("\n### Warnings:\n")
            for warning in warnings:
                markdown_content.append(f"- {warning}\n")
        
        if errors:
            markdown_content.append("\n### Errors:\n")
            for error in errors:
                markdown_content.append(f"- {error}\n")
        
        return "\n".join(markdown_content)
        
    except Exception as e:
        error_msg = f"Failed to process PDF: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

# Streamlit UI
st.title("📄 PDF to Markdown Converter with OCR")
st.write("Extract text and OCR images from PDF files to Markdown")

# Add file size warning
st.info(f"Maximum PDF size: {MAX_PDF_SIZE_MB}MB per file")

uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("✨ Process Files") and uploaded_files:
    zip_buffer = BytesIO()
    total_files = len(uploaded_files)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processing {uploaded_file.name} ({i+1}/{total_files})..."):
                try:
                    md_content = process_pdf(uploaded_file.getvalue())
                    if md_content:
                        filename = f"{os.path.splitext(uploaded_file.name)[0]}.md"
                        zipf.writestr(filename, md_content.encode('utf-8'))
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    logger.error(f"Error processing {uploaded_file.name}: {str(e)}", exc_info=True)
    
    if zip_buffer.getvalue():  # Only show download if we have processed files
        st.success("✅ Processing complete!")
        st.download_button(
            label="📥 Download Markdown Files",
            data=zip_buffer.getvalue(),
            file_name="processed_documents.zip",
            mime="application/zip"
        )
