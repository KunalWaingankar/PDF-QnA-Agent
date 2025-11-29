# Your PDFQnAAgent class will go here

import logging
import os
import faiss
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch
import io
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import re
from logging_config import setup_logging
import random


class PDFQnAAgent:
    def __init__(self, logger=None, index_path="my_index.faiss"):
        # --- Setup Logger ---
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if re-initialized
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # --- Load Captioning Model for Images ---
        self.logger.info("Loading BLIP captioning model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # --- Sentence Embedding Model ---
        self.logger.info("Loading sentence transformer model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # --- Local LLM ---
        self.logger.info("Loading local Mistral model...")
        #self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # --- Paths & State ---
        self.index_path = index_path
        self.index = None
        self.embeddings = None
        self.chunks = None

        self.logger.info("✅ All models loaded successfully.")



#---------------------------------------------------------------------------------------------------------------------------------

    def extract_ordered_content(self, pdf_path):
        self.logger.info(f"📄 Starting PDF extraction: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            self.logger.error(f"❌ Failed to open PDF: {e}")
            return ""

        full_output = ""

        for page_num, page in enumerate(doc):
            self.logger.info(f"📃 Extracting Page {page_num + 1}")
            # print(f"\n--- Extracting Page {page_num + 1} ---")

            elements = []

            # -------- TEXT BLOCKS --------
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:  # It's a text block
                    bbox = block["bbox"]
                    block_lines = block["lines"]

                    block_spans = []
                    all_font_sizes = []
                    block_text = ""

                    for line in block_lines:
                        for span in line["spans"]:
                            block_spans.append(span)
                            block_text += span["text"] + " "
                            all_font_sizes.append(span["size"])

                    block_text = block_text.strip()

                    # If the text block is empty, skip
                    if not block_text:
                        continue

                    # -------- Determine Type (Heading / Paragraph / Caption) --------
                    avg_font_size = sum(all_font_sizes) / len(all_font_sizes)

                    if avg_font_size >= 15 and len(block_text.split()) < 10:
                        tag = "[🔷 Heading]"
                    elif avg_font_size <= 10 and len(block_text.split()) < 15:
                        tag = "[📷 Caption]"
                    else:
                        tag = "[📄 Paragraph]"

                    labeled_text = f"{tag}\n{block_text}"
                    elements.append(("text", bbox[1], labeled_text))  # bbox[1] is vertical y-pos

            # -------- IMAGE BLOCKS --------
            image_list = page.get_images(full=True)
            self.logger.info(f"🖼️ Found {len(image_list)} images on Page {page_num + 1}")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                bbox = page.get_image_bbox(img)

                # OCR - extract visible text from the image
                try:
                    ocr_text = pytesseract.image_to_string(img_pil).strip()
                except Exception as e:
                    self.logger.warning(f"⚠️ OCR failed on image {img_index} Page {page_num + 1}: {e}")
                    ocr_text = "[OCR Failed]"

                # Captioning - generate a description of the image
                try:
                    inputs = self.processor(images=img_pil, return_tensors="pt")
                    out = self.caption_model.generate(**inputs)
                    caption = self.processor.decode(out[0], skip_special_tokens=True)
                except Exception as e:
                    self.logger.warning(f"⚠️ Captioning failed on image {img_index} Page {page_num + 1}: {e}")
                    caption = "[Captioning Failed]"

                # Build final image block
                image_content = f"[📷 Image]\n📖 OCR: {ocr_text}\n🖼️ Caption: {caption}"
                elements.append(("image", bbox.y0, image_content))

            # -------- SORT ALL CONTENT BASED ON VISUAL POSITION (Top to Bottom) --------
            elements.sort(key=lambda x: x[1])

            # -------- ADD TO FINAL OUTPUT --------
            for el_type, y_pos, content in elements:
                full_output += content + "\n\n"

        self.logger.info("✅ Finished PDF extraction successfully.")
        return full_output.strip()



#-------------------------------------------------------------------------------------------------------------------------------------
    def chunk_and_clean_text(self, full_output, max_chunk_words=200, overlap_words=30):
        """
        Cleans and chunks extracted PDF text for embedding & retrieval, 
        with sentence fallback and optional overlap.

        Args:
            full_output (str): Raw output from extract_ordered_content().
            max_chunk_words (int): Maximum number of words per chunk.
            overlap_words (int): Number of words to overlap between chunks.

        Returns:
            List[str]: Cleaned, organized chunks ready for embedding.
        """
        # --- Step 0: Remove unwanted references/footers ---
        # Remove reference-like patterns: [1], (1), [a], (a)
        full_output = re.sub(r'\[\d+\]|\(\d+\)|\[[a-zA-Z]\]|\([a-zA-Z]\)', '', full_output)
        # Remove common footer patterns like "Page 1 of 10"
        full_output = re.sub(r'Page \d+ of \d+', '', full_output, flags=re.IGNORECASE)
        # Remove lines starting with References or Bibliography
        lines = full_output.split('\n')
        lines = [line for line in lines if not re.match(r'^(References|Bibliography)', line, re.IGNORECASE)]
        full_output = re.sub(r'(?i)\breferences\b|\bbibliography\b', '', full_output)
        full_output = re.sub(r'http\S+', '', full_output)  # Remove URLs
        full_output = re.sub(r'\[\📷 Caption\]|\[\📄 Paragraph\]', '', full_output)


        full_output = "\n".join(lines)


        # Step 1: Basic clean work
        full_output = re.sub(r'\s+', ' ', full_output)
        full_output = full_output.replace('\u200b', '')  # Removes hidden characters
        self.logger.info("Basic cleaning completed.")

        # Step 2: Split into blocks
        if "\n\n" in full_output:
            raw_blocks = full_output.split("\n\n")
            self.logger.info("Paragraph-based splitting used.")
        else:
            # Fallback: split into sentences if no paragraph breaks
            raw_blocks = re.split(r'(?<=[.?!])\s+', full_output)
            self.logger.info("Sentence-based splitting used as fallback.")

        self.logger.info(f"Found {len(raw_blocks)} blocks in the document.")

        chunks = []
        current_chunk_words = []

        # Step 3: Group into chunks with overlap
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            words = block.split()

            if len(current_chunk_words) + len(words) > max_chunk_words:
                # Save the current chunk
                chunk_text = " ".join(current_chunk_words).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    self.logger.info(f"Chunk {len(chunks)} finalized with {len(current_chunk_words)} words.")

                # Start new chunk with overlap
                overlap_words = min(overlap_words, max_chunk_words // 4)
                current_chunk_words = current_chunk_words[-overlap_words:] + words
            else:
                current_chunk_words.extend(words)

        # Add final chunk
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words).strip())
            self.logger.info(f"Final chunk {len(chunks)} added with {len(current_chunk_words)} words.")

        self.logger.info(f"Total {len(chunks)} chunks created successfully.")
        chunks = list(dict.fromkeys(chunks))
        return chunks

#------------------------------------------------------------------------------------------------------------------------------

    #def embed_chunks(self, chunks, index_path="my_index.faiss", load_if_exists=False):
    def embed_chunks(self, chunks, load_if_exists=False):
        """
        Takes a list of text chunks and returns a FAISS index and metadata.
        Optionally saves/loads the index to/from disk.

        Args:
            chunks (List[str]): The text chunks to embed.
            index_path (str): Path to save/load the FAISS index.
            load_if_exists (bool): If True, loads an existing index instead of creating new.

        Returns:
            index: FAISS index
            embeddings: Numpy array of embeddings
            chunks: Original chunks (used for later retrieval)
        """
        #logger = logging.getLogger(__name__)
        self.logger.info("Embedding process started...")

        if load_if_exists and os.path.exists(self.index_path):
            try:
                self.logger.info(f"Loading existing FAISS index from: {self.index_path}")
                print(f"📁 Loading existing FAISS index from: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                self.embeddings = None  # We don't have the original embedding array when loading
                return self.index, self.embeddings, chunks
            except Exception as e:
                self.logger.error("Failed to load FAISS index.", exc_info=True)
                return None, None, None



                # ---------- Case: Create new FAISS index ----------
        try:
            self.logger.info("Creating new FAISS index...")
            print("📌 Creating new FAISS index...")

            # 1. Create embeddings for each chunk
            self.embeddings = self.embed_model.encode(chunks, convert_to_numpy=True)

            # 2. Create and fill FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            self.logger.info("Embeddings added to FAISS index.")

            # 3. Save the index to disk
            faiss.write_index(self.index, self.index_path)
            self.logger.info(f"FAISS index saved at {self.index_path}")
            print(f"✅ FAISS index saved at: {self.index_path}")

            return self.index, self.embeddings, chunks

        except Exception as e:
            self.logger.error("Error during FAISS index creation or embedding.", exc_info=True)
            return None, None, None


#--------------------------------------------------------------------------------------------------------------------------
    """
    Given a user question, this finds the most relevant and diverse chunks from your PDF using FAISS,
    and returns them for context-aware LLM answering. Uses MMR for smarter selection.

    Args:
        question (str): User's question.
        top_k (int): Number of final chunks to select for LLM.
        candidate_top_k (int): Number of candidate chunks to retrieve from FAISS before MMR.
        lambda_mmr (float): Weight for relevance vs diversity in MMR (0-1).

    Returns:
        List[str]: Retrieved chunks used for context.
    """
    #def ask_question(self, question, index, embeddings, chunks, top_k=3):
    def ask_question(self, question, top_k=5, candidate_top_k=10, lambda_mmr=0.5, use_mmr=True):
        """
    Retrieves top relevant and diverse chunks from FAISS using MMR.
    Now prevents duplicate chunks and adds small randomness to break ties.
    """
        self.logger.info("🔍 New question received: '%s'", question)

        if self.index is None or self.chunks is None:
            self.logger.error("Index or chunks not available. Please process PDF first.")
            return []

        try:
            # Step 1: Embed the question
            question_vec = self.embed_model.encode([question])
            self.logger.info("✅ Question embedded successfully.")

            # Step 2: Retrieve candidate chunks from FAISS
            distances, indices = self.index.search(
                np.array(question_vec).astype("float32"),
                candidate_top_k
            )
            candidate_chunks = [self.chunks[i] for i in indices[0]]

            # Remove exact duplicate chunks while preserving order
            seen = set()
            candidate_chunks = [c for c in candidate_chunks if not (c in seen or seen.add(c))]

            self.logger.info(f"🔎 {len(candidate_chunks)} unique candidate chunks retrieved from FAISS.")

            if not use_mmr or len(candidate_chunks) <= top_k:
                self.logger.info(f"📄 Returning top {top_k} chunks without MMR.")
                return candidate_chunks[:top_k]

            # Step 3: Apply MMR to select final top_k diverse chunks
            import torch
            from sentence_transformers import util
            import random

            chunk_embeddings = self.embed_model.encode(candidate_chunks, convert_to_tensor=True)
            question_embedding = torch.tensor(question_vec)

            selected = []
            candidate_idx = list(range(len(candidate_chunks)))
            random.shuffle(candidate_idx)  # Breaks tie situations

            for _ in range(top_k):
                if not candidate_idx:
                    break

                scores = []
                for i in candidate_idx:
                    relevance = util.cos_sim(question_embedding, chunk_embeddings[i]).item()
                    diversity = max(
                        [util.cos_sim(chunk_embeddings[i], chunk_embeddings[j]).item() for j in selected],
                        default=0
                    )
                    mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * diversity
                    scores.append(mmr_score)

                best_idx = candidate_idx[np.argmax(scores)]
                selected.append(best_idx)
                candidate_idx.remove(best_idx)

            retrieved_chunks = [candidate_chunks[i] for i in selected]

            # Final duplicate check (just in case)
            retrieved_chunks = list(dict.fromkeys(retrieved_chunks))

            self.logger.info(f"📄 {len(retrieved_chunks)} final diverse chunks selected using MMR.")

            return retrieved_chunks

        except Exception as e:
            self.logger.error("❌ Error occurred while retrieving chunks.", exc_info=True)
            return []




#----------------------------------------------------------------------------------------------------------------------------

    def answer_question_local(self, question, retrieved_chunks):
        """
        Takes a user question and relevant document chunks,
        feeds them to a local LLM (e.g., Mistral) to generate a context-based answer.
        """

        #logger = logging.getLogger(__name__)
        self.logger.info("🧠 Starting local answer generation...")

        try:
            #self.logger.info(f"🧾 Retrieved Chunks: {retrieved_chunks}")

            filtered_chunks = []
            for c in retrieved_chunks:
                if all(util.cos_sim(self.embed_model.encode(c), self.embed_model.encode(fc)).item() < 0.9 for fc in filtered_chunks):
                    filtered_chunks.append(c)

            
            context = "\n\n".join(filtered_chunks[:5])

            # 1. Create the combined context from relevant PDF chunks
            #context = "\n\n".join(retrieved_chunks)

            # 2. Craft the prompt in instruction format
            prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
                        Answer in 3-4 lines. Do not repeat words, phrases and sentences. Focus only on the main content.

        Context: {context}

        Question: {question}

        Answer is 3-4 lines:
        """
            self.logger.debug("📝 Prompt constructed.")

            # 3. Tokenize the prompt and move to correct device (GPU/CPU)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.llm.device)
            self.logger.info("✅ Prompt tokenized and moved to device.")

            # 4. Generate the model output (answer)
            output = self.llm.generate(**inputs, max_new_tokens=100)
            self.logger.info("📤 Model response generated.")

            # 5. Decode the output tokens into readable text
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.logger.info("✅ Answer decoded successfully.")

            return answer.strip()

        except Exception as e:
            self.logger.error("❌ Error during local answer generation.", exc_info=True)
            return "Sorry, I couldn’t generate a response."


#--------------------------------------------------------------------------------------------------------------------------------

#----------------------------   MAIN  FUNCTIONS/METHODS   -----------------------------------------------------------------


    def process_pdf(self, pdf_path: str, index_path: str = "my_index.faiss"):
        """
        Extracts, cleans, chunks, and embeds the PDF end-to-end.
        Saves index and stores everything in memory for later use.
        """
        self.logger.info(f"🔄 Starting full processing pipeline for: {pdf_path}")
        self.pdf_path = pdf_path

        # Step 1: Extract structured content from PDF
        raw_output = self.extract_ordered_content(pdf_path)
        #self.logger.info(f"🧾 First 300 chars of chunk[0]: {self.chunks[0][:300]}")
        self.logger.info(f"✅ chunk_and_clean_text() returned: {self.chunks}")

        # Step 2: Clean and chunk the extracted content
        self.chunks = self.chunk_and_clean_text(raw_output)

        # Step 3: Embed chunks and build FAISS index
        self.index, self.embeddings, self.chunks = self.embed_chunks(self.chunks)

        self.logger.info("✅ PDF processing completed and stored in memory.")



    def ask(self, question: str, top_k: int = 5):
        """
        Takes a user question, searches the index for relevant chunks,
        and generates a local LLM answer using those chunks.
        """
        self.logger.info(f"🗣️ Received question: {question}")

        # Step 1: Retrieve top relevant chunks
        self.retrieved_chunks = self.ask_question(question, top_k=top_k, use_mmr=True)

        # Step 2: Generate answer using local model
        answer = self.answer_question_local(question, self.retrieved_chunks)

        return answer


    def summarize_pdf(self):
        """
        Summarizes the full PDF content using the loaded LLM.
        Assumes self.chunks is already populated via process_pdf().
        """
        self.logger.info("Starting PDF summarization...")

        if not self.chunks:
            self.logger.error("No chunks found. Did you run process_pdf()?")
            return "No PDF content found. Please run process_pdf() first."

        # Join all chunks into one big context
        full_text = "\n\n".join(self.chunks)

        # Build the summarization prompt
        prompt = f"""[INST] You are a helpful assistant. Summarize the following document clearly and concisely.

    Document:
    {full_text}

    Summary: [/INST]"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)

        # Generate summary
        output = self.llm.generate(**inputs, max_new_tokens=300)

        # Decode
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)

        self.logger.info("Summarization complete.")
        return summary.strip()



