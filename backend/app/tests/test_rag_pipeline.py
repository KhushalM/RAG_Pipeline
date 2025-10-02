import requests
import os
import time
import json
from typing import Dict, Any


class RAGPipelineTest:
    """Complete RAG pipeline test including upload and query functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.pdf_files_dir = "/Users/khushal/Documents/GitHub/RAG_Pipeline/backend/pdf_files"
    
    def test_pdf_upload(self) -> bool:
        """Test PDF upload functionality"""
        print("\n📄 Testing PDF Upload...")
        
        # Find available PDFs
        test_pdfs = []
        if os.path.exists(self.pdf_files_dir):
            for file in os.listdir(self.pdf_files_dir):
                if file.endswith('.pdf'):
                    test_pdfs.append(os.path.join(self.pdf_files_dir, file))
        
        if not test_pdfs:
            print(f"❌ No PDFs found in {self.pdf_files_dir}")
            return False
        
        print(f"Found {len(test_pdfs)} PDFs:")
        for pdf in test_pdfs:
            print(f"  - {os.path.basename(pdf)}")
        
        # Prepare files for upload
        files = []
        for pdf_path in test_pdfs:
            files.append(('pdf_files', (os.path.basename(pdf_path), open(pdf_path, 'rb'), 'application/pdf')))
        
        try:
            response = requests.post(f"{self.base_url}/pdf_upload", files=files)
            
            # Close file handles
            for _, file_tuple in files:
                file_tuple[1].close()
            
            if response.status_code == 200:
                upload_data = response.json()
                print("✅ PDF upload successful")
                print(f"   Total chunks: {upload_data.get('number_of_chunks', 0)}")
                print(f"   Per file chunks: {upload_data.get('per_file_chunks', {})}")
                return True
            else:
                print(f"❌ PDF upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ PDF upload failed: {e}")
            return False
    
    def test_query_processing(self, queries: list = None) -> bool:
        """Test query processing functionality"""
        print("\n🔍 Testing Query Processing...")
        
        if queries is None:
            queries = [
                "After how many months did Steve Jobs drop out and from which university?",
            ]
        
        success_count = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            
            # Test different retrieval modes
            retrieval_modes = ["hybrid", "semantic", "lexical"]
            
            for mode in retrieval_modes:
                success = self._test_single_query(query, mode)
                if success:
                    success_count += 1
        
        total_tests = len(queries) * len(retrieval_modes)
        print(f"\n📊 Query Test Results: {success_count}/{total_tests} successful")
        
        return success_count > 0
    
    def _test_single_query(self, query: str, retrieval_mode: str = "hybrid") -> bool:
        """Test a single query with specified retrieval mode"""
        
        payload = {
            "query": query,
            "max_context_chunks": 3,
            "retrieval_mode": retrieval_mode,
            "temperature": 0.7,
            "max_tokens": 500,
        }
        
        try:
            response = requests.post(f"{self.base_url}/query_processing", json=payload)
            
            if response.status_code == 200:
                query_data = response.json()
                print(f"  ✅ {retrieval_mode} mode: Success")
                print(f"     Processing time: {query_data.get('processing_time', 0):.2f}s")
                print(f"     Sources: {len(query_data.get('sources', []))}")
                print(f"     Sources: {query_data.get('sources', [])}")
                print(f"     Answer length: {len(query_data.get('answer', ''))}")
                
                answer = query_data.get('answer', '')
                if answer:
                    preview = answer[0:500] + "..." if len(answer) > 100 else answer
                    print(f"     Answer preview: {preview}")
                
                return True
            else:
                print(f"  ❌ {retrieval_mode} mode failed: {response.status_code}")
                print(f"     Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ {retrieval_mode} mode failed: {e}")
            return False
    
    def run_full_pipeline_test(self) -> bool:
        """Run the complete RAG pipeline test"""
        print("🚀 Starting Full RAG Pipeline Test")
        print("=" * 50)
        
        # Step 1: Upload PDFs
        if not self.test_pdf_upload():
            print("❌ Pipeline test failed at PDF upload")
            return False
        
        # Step 2: Test queries
        if not self.test_query_processing():
            print("❌ Pipeline test failed at query processing")
            return False
        
        
        print("\n" + "=" * 50)
        print("🎉 Full RAG Pipeline Test PASSED!")
        return True


def run_pipeline_test():
    """Main function to run the pipeline test"""
    tester = RAGPipelineTest()   
    success = tester.run_full_pipeline_test()
    
    if success:
        print("\n✅ All tests passed! RAG pipeline is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return success


if __name__ == "__main__":
    run_pipeline_test()
