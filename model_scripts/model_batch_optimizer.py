from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import psutil
import time
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelBatchSizeOptimizer:
    def __init__(self, model_name: str, use_cuda: bool = True):
        logger.info(f"Initializing ModelBatchSizeOptimizer with model: {model_name}, use_cuda: {use_cuda}")
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.debug(f"Selected device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vram_limit = 6 * 1024 * 1024 * 1024  # 6GB VRAM in bytes
        self.ram_limit = 16 * 1024 * 1024 * 1024  # 16GB RAM in bytes
        self.model_size = self._estimate_model_size()
        logger.info(f"Estimated model size: {self.model_size / (1024 * 1024):.2f} MB")

    def _estimate_model_size(self) -> int:
        """Estimate model size in bytes (FP16)."""
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f"Model has {num_params} parameters")
        return num_params * 2  # FP16: 2 bytes per parameter

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

        # Available memory: reserve 1GB VRAM, 4GB RAM for system
        available_vram = self.vram_limit - self.model_size - 1 * 1024 * 1024 * 1024
        available_ram = psutil.virtual_memory().available - 4 * 1024 * 1024 * 1024
        logger.debug(f"Available VRAM: {available_vram / (1024 * 1024):.2f} MB, Available RAM: {available_ram / (1024 * 1024):.2f} MB")

        # Encode sample sentences to estimate memory per input
        encoded = self.tokenizer(sample_sentences[:1], padding=True, truncation=True, return_tensors="pt")
        input_size = sum(t.element_size() * t.numel() for t in encoded.values()) * 2  # Approx. FP16
        logger.debug(f"Estimated input size per sentence: {input_size / 1024:.2f} KB")

        # Estimate max batch size based on memory
        max_vram_batch = int(available_vram // input_size)
        max_ram_batch = int(available_ram // input_size)
        max_batch = min(max_vram_batch, max_ram_batch, max_batch_size)
        logger.info(f"Maximum batch size based on memory: {max_batch}")

        if max_batch < 1:
            logger.warning("Memory constraints too tight, returning batch size 1")
            return 1

        # Test throughput for powers of 2 up to max_batch
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
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches", total=(len(sentences) + batch_size - 1) // batch_size):
            batch = sentences[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
            logger.debug(f"Processed batch {i // batch_size + 1}, size: {len(batch)}")
        result = torch.cat(embeddings, dim=0)
        logger.info(f"Generated embeddings with shape: {result.shape}")
        return result

if __name__ == "__main__":
    # Example usage
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    sample_sentences = [
        "This is a sample sentence for testing embeddings.",
        "Another sentence to demonstrate batch processing.",
        "Machine learning models require efficient resource usage.",
        "Dynamic batch sizing optimizes performance on limited hardware."
    ] * 25  # Create 100 sentences for realistic testing
    optimizer = ModelBatchSizeOptimizer(model_name, use_cuda=True)
    optimal_batch_size = optimizer.get_optimal_batch_size(sample_sentences)
    print(f"Optimal batch size: {optimal_batch_size}")
    embeddings = optimizer.generate_embeddings(sample_sentences, optimal_batch_size)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Sample embedding: {embeddings[0][:5]}")  # Show first 5 dimensions of first embedding
