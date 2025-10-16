import argparse
import torch
from transformers import AutoModel, AutoTokenizer
import psutil
import time
import logging
from typing import List
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class ModelBatchSizeOptimizer:
    def __init__(self, model_name: str, use_accelerator: bool = True):
        logger.info(f"Initializing ModelBatchSizeOptimizer with model: {model_name}, use_accelerator: {use_accelerator}")
        # Dynamic device selection: cuda > mps > cpu
        if use_accelerator and torch.cuda.is_available() and hasattr(torch, 'cuda'):
            try:
                self.device = torch.device("cuda")
                self.vram_limit = torch.cuda.get_device_properties(0).total_memory  # Dynamic VRAM detection
            except AssertionError:
                logger.warning("CUDA enabled but get_device_properties failed, using default VRAM limit")
                self.device = torch.device("cpu")
                self.vram_limit = 6 * 1024 * 1024 * 1024  # Fallback for GTX 1660
        elif use_accelerator and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.vram_limit = 8 * 1024 * 1024 * 1024  # Assume 8 GB for MPS (conservative estimate for M1)
        else:
            self.device = torch.device("cpu")
            self.vram_limit = 0  # No VRAM limit for CPU
        logger.debug(f"Selected device: {self.device}, VRAM limit: {self.vram_limit / (1024 * 1024):.2f} MB")
        
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # Apply FP16 only for stable models
        if model_name == "sentence-transformers/all-MiniLM-L12-v2":
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ram_limit = psutil.virtual_memory().available  # Dynamic RAM detection
        self.model_size = self._estimate_model_size()
        logger.info(f"Estimated model size: {self.model_size / (1024 * 1024):.2f} MB, Available RAM: {self.ram_limit / (1024 * 1024):.2f} MB")
    
    def _estimate_model_size(self) -> int:
        """Estimate model size in bytes (FP16 or FP32 based on model dtype)."""
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f"Model has {num_params} parameters")
        return num_params * (2 if self.model.dtype == torch.float16 else 4)
    
    def get_optimal_batch_size(self, sample_sentences: List[str], max_batch_size: int = 128) -> int:
        """
        Dynamically determine optimal batch size based on memory and throughput.
        Args:
            sample_sentences: List of sample sentences to test throughput.
            max_batch_size: Maximum batch size to test.
        Returns:
            Optimal batch size.
        """
        logger.info("Starting batch size optimization")
        if not sample_sentences:
            logger.warning("No sample sentences provided, returning batch size 1")
            return 1
        if len(sample_sentences) == 1:
            logger.info("Single sentence provided, returning batch size 1")
            return 1

        try:
            available_vram = self.vram_limit - self.model_size if self.vram_limit > 0 else float('inf')
            available_vram = max(0, available_vram - 1 * 1024 * 1024 * 1024)  # Reserve 1 GB
            available_ram = self.ram_limit - 4 * 1024 * 1024 * 1024  # Reserve 4 GB
            logger.debug(f"Available VRAM: {available_vram / (1024 * 1024):.2f} MB, Available RAM: {available_ram / (1024 * 1024):.2f} MB")
            
            encoded = self.tokenizer(sample_sentences[:1], padding=True, truncation=True, return_tensors="pt")
            input_size = sum(t.element_size() * t.numel() for t in encoded.values()) * (2 if self.model.dtype == torch.float16 else 4)
            logger.debug(f"Estimated input size per sentence: {input_size / 1024:.2f} KB")
            
            max_vram_batch = max(1, int(available_vram // input_size)) if self.vram_limit > 0 else float('inf')
            max_ram_batch = max(1, int(available_ram // input_size))
            max_batch = min(max_vram_batch, max_ram_batch, max_batch_size, len(sample_sentences))
            logger.info(f"Maximum batch size based on memory: {max_batch}")
            
            if max_batch < 1:
                logger.warning("Memory constraints too tight, returning batch size 1")
                return 1
            
            best_batch = 1
            best_throughput = 0.0
            test_sizes = [2**i for i in range(int(max_batch).bit_length()) if 2**i <= max_batch]
            logger.debug(f"Testing batch sizes: {test_sizes}")
            
            for batch_size in test_sizes:
                try:
                    logger.info(f"Testing batch size: {batch_size}")
                    start_time = time.time()
                    for i in range(0, len(sample_sentences), batch_size):
                        batch = sample_sentences[i:i + batch_size]
                        encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                        with torch.no_grad():
                            self.model(**encoded)
                    elapsed = time.time() - start_time
                    throughput = len(sample_sentences) / elapsed
                    logger.debug(f"Batch size {batch_size} throughput: {throughput:.2f} sentences/sec")
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch = batch_size
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"Out of memory with batch size {batch_size}, stopping")
                        break
                    logger.error(f"Unexpected error with batch size {batch_size}: {str(e)}")
                    raise e
            
            logger.info(f"Selected optimal batch size: {best_batch}, throughput: {best_throughput:.2f} sentences/sec")
            return best_batch
        except Exception as e:
            logger.error(f"Error during batch size optimization: {str(e)}")
            return 1
    
    def generate_embeddings(self, sentences: List[str], batch_size: int) -> torch.Tensor:
        """
        Generate embeddings for a list of sentences using the specified batch size.
        Args:
            sentences: List of sentences to encode.
            batch_size: Batch size for processing.
        Returns:
            Tensor of embeddings.
        """
        logger.info(f"Generating embeddings for {len(sentences)} sentences with batch size {batch_size}")
        # Filter out invalid inputs
        valid_sentences = [s for s in sentences if s and isinstance(s, str) and len(s.strip()) > 0]
        if len(valid_sentences) < len(sentences):
            logger.warning(f"Filtered out {len(sentences) - len(valid_sentences)} invalid sentences (empty or non-string)")
        if not valid_sentences:
            raise ValueError("No valid sentences provided for embedding")
        
        embeddings = []
        for i in tqdm(range(0, len(valid_sentences), batch_size), desc="Processing batches", total=(len(valid_sentences) + batch_size - 1) // batch_size):
            batch = valid_sentences[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            # Check for NaNs
            if torch.isnan(batch_embeddings).any():
                logger.warning(f"NaN detected in embeddings for batch {i // batch_size + 1}. Replacing with zeros.")
                batch_embeddings = torch.where(torch.isnan(batch_embeddings), torch.zeros_like(batch_embeddings), batch_embeddings)
            embeddings.append(batch_embeddings)
            logger.debug(f"Processed batch {i // batch_size + 1}, size: {len(batch)}")
        result = torch.cat(embeddings, dim=0)
        logger.info(f"Generated embeddings with shape: {result.shape}")
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch size and embedding optimizer.")
    parser.add_argument("-m", "--model", type=str, default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Embedding model name (default: sentence-transformers/all-MiniLM-L12-v2)")
    parser.add_argument("-f", "--file", type=str,
                        help="File with one sentence per line (default: builtin sentences)")
    args = parser.parse_args()
    
    # Load sentences
    if args.file and os.path.exists(args.file):
        with open(args.file, 'r', encoding='utf-8') as f:
            sample_sentences = [line.strip() for line in f if line.strip()]
    else:
        sample_sentences = [
            "This is a sample sentence for testing embeddings.",
            "Another sentence to demonstrate batch processing.",
            "Machine learning models require efficient resource usage.",
            "Dynamic batch sizing optimizes performance on limited hardware."
        ] * 25
    
    optimizer = ModelBatchSizeOptimizer(args.model, use_accelerator=True)
    optimal_batch_size = optimizer.get_optimal_batch_size(sample_sentences)
    print(f"Optimal batch size: {optimal_batch_size}")
    embeddings = optimizer.generate_embeddings(sample_sentences, optimal_batch_size)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Sample embedding: {embeddings[0][:5]}")
