# PDF-QnA-Agent
Intelligent PDF Q&amp;A agent using Python, FAISS, and Local LLMs.
 
•	Built a complete pipeline to read PDFs, clean content, and split documents into meaningful chunks for easier searching.
• Added support for images using OCR (Tesseract) to read text in images and BLIP to generate image captions, including them in 
the search index. 
• Converted chunks into vector embeddings and stored them in FAISS for fast and smart semantic search across large 
documents. 
• Used MMR (Maximal Marginal Relevance) to select diverse and relevant chunks, avoiding repeated or similar content in 
answers. 
• Integrated a local LLM (TinyLlama/Mistral) to generate short, accurate answers and summaries based on the retrieved 
chunks. 
• Delivered a Streamlit web app for easy, interactive use with PDFs. 
• Focused on offline usage, modular code, and reusable functions (process_pdf, ask, summarize_pdf) for maintainability. 
• Tech Stack: Python, PyMuPDF, Tesseract OCR, BLIP, HuggingFace Transformers, FAISS, SentenceTransformers, PyTorch, 
Logging. 
