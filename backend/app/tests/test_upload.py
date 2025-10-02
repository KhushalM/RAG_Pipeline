import requests
import os

def test_multiple_pdf_upload():
    """Test the PDF upload endpoint with multiple PDFs"""
    
    # Define multiple test PDFs
    pdf_files_dir = "/Users/khushal/Documents/GitHub/RAG_Pipeline/backend/pdf_files"
    test_pdfs = [
        os.path.join(pdf_files_dir, "ST.pdf"),
        os.path.join(pdf_files_dir, "PG.pdf")
    ]
    
    # Check which PDFs exist
    existing_pdfs = []
    for pdf_path in test_pdfs:
        if os.path.exists(pdf_path):
            existing_pdfs.append(pdf_path)
        else:
            print(f"⚠️  PDF not found: {pdf_path}")
    
    if not existing_pdfs:
        print("❌ No test PDFs found!")
        print(f"Please add PDFs to {pdf_files_dir}")
        return
    
    print(f"📄 Found {len(existing_pdfs)} PDFs to upload:")
    for pdf in existing_pdfs:
        print(f"  - {os.path.basename(pdf)}")
    
    url = "http://localhost:8000/pdf_upload"
    
    # Prepare files for upload
    files = []
    file_handles = []
    
    for pdf_path in existing_pdfs:
        file_handle = open(pdf_path, "rb")
        file_handles.append(file_handle)
        files.append(("pdf_files", (os.path.basename(pdf_path), file_handle, "application/pdf")))
    
    try:
        print("🚀 Uploading multiple PDFs...")
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multiple PDF upload successful!")
            print(f"📊 Upload Results:")
            print(f"  - Total chunks: {result.get('number_of_chunks', 'N/A')}")
            print(f"  - Per file chunks: {result.get('per_file_chunks', {})}")
            
            chunk_info = result.get('chunk_info', {})
            if chunk_info:
                print(f"  - Average chunk size: {chunk_info.get('average_size', 'N/A')} chars")
                print(f"  - Min chunk size: {chunk_info.get('min_size', 'N/A')} chars")
                print(f"  - Max chunk size: {chunk_info.get('max_size', 'N/A')} chars")
        else:
            print("❌ Multiple PDF upload failed!")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close all file handles
        for file_handle in file_handles:
            file_handle.close()

def test_single_pdf_upload():
    """Test the PDF upload endpoint with a single PDF (original test)"""
    
    test_pdf_path = "/Users/khushal/Documents/GitHub/RAG_Pipeline/backend/pdf_files/ST.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"Please add a test PDF at {test_pdf_path}")
        return
    
    url = "http://localhost:8000/pdf_upload"
    
    with open(test_pdf_path, "rb") as f:
        files = {"pdf_files": ("ST.pdf", f, "application/pdf")}
        
        try:
            print("🚀 Uploading single PDF...")
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Single PDF upload successful!")
                print(f"Response: {result}")
            else:
                print("❌ Single PDF upload failed!")
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Server not running. Start with: uvicorn app.main:app --reload")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_health_check():
    """Test the health check endpoint"""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        print(f"Health Check Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed!")
            print(f"📊 DB Stats:")
            print(f"  - Documents count: {result.get('documents_count', 'N/A')}")
            print(f"  - DB stats: {result.get('db_stats', {})}")
        else:
            print("❌ Health check failed!")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=== RAG Pipeline Upload Tests ===\n")
    
    # Test health check first
    print("1. Testing health check...")
    test_health_check()
    print()
    
    # Test single PDF upload
    print("2. Testing single PDF upload...")
    test_single_pdf_upload()
    print()
    
    # Test multiple PDF upload
    print("3. Testing multiple PDF upload...")
    test_multiple_pdf_upload()
    print()
    
    print("=== Tests Complete ===")