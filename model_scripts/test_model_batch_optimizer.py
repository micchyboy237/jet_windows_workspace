import pytest
import torch
from model_batch_optimizer import ModelBatchSizeOptimizer

@pytest.fixture
def optimizer():
    handler = ModelBatchSizeOptimizer("sentence-transformers/all-MiniLM-L12-v2", use_cuda=False)
    yield handler
    # Cleanup: Clear model from memory
    del handler.model
    torch.cuda.empty_cache()

def test_get_optimal_batch_size_single_sentence(optimizer):
    # Given: A single sentence and default max batch size
    sample_sentences = ["This is a test sentence."]
    expected_batch_size = 1
    # When: Getting optimal batch size
    result = optimizer.get_optimal_batch_size(sample_sentences)
    # Then: Should return 1 for minimal input
    assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

def test_get_optimal_batch_size_multiple_sentences(optimizer):
    # Given: Multiple sentences to test throughput
    sample_sentences = ["Sentence " + str(i) for i in range(100)]
    expected_batch_size_lower_bound = 1
    expected_batch_size_upper_bound = 128
    # When: Getting optimal batch size
    result = optimizer.get_optimal_batch_size(sample_sentences)
    # Then: Should return a reasonable batch size within memory limits
    assert expected_batch_size_lower_bound <= result <= expected_batch_size_upper_bound, \
        f"Expected batch size between {expected_batch_size_lower_bound} and {expected_batch_size_upper_bound}, got {result}"

def test_get_optimal_batch_size_empty_input(optimizer):
    # Given: Empty sentence list
    sample_sentences = []
    expected_batch_size = 1
    # When: Getting optimal batch size
    result = optimizer.get_optimal_batch_size(sample_sentences)
    # Then: Should return 1 for empty input
    assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

def test_get_optimal_batch_size_memory_constraint(mocker, optimizer):
    # Given: Mock low memory availability
    mocker.patch('psutil.virtual_memory', return_value=mocker.Mock(available=5 * 1024 * 1024 * 1024))  # 5GB available RAM
    sample_sentences = ["Sentence " + str(i) for i in range(50)]
    expected_batch_size_upper_bound = 32
    # When: Getting optimal batch size
    result = optimizer.get_optimal_batch_size(sample_sentences)
    # Then: Should return a batch size within constrained memory
    assert result <= expected_batch_size_upper_bound, f"Expected batch size <= {expected_batch_size_upper_bound}, got {result}"

def test_generate_embeddings_main_block(optimizer):
    # Given: A set of sample sentences and a valid batch size
    sample_sentences = [
        "This is a test sentence.",
        "Another sentence for embedding."
    ]
    batch_size = 2
    expected_shape = torch.Size([2, 384])  # MiniLM-L12 has 384 hidden size
    # When: Generating embeddings using the main block logic
    result = optimizer.generate_embeddings(sample_sentences, batch_size)
    # Then: Should return embeddings with correct shape and device
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    assert result.device.type == optimizer.device.type, f"Expected device {optimizer.device.type}, got {result.device.type}"