
import pytest
import torch
from model_batch_optimizer import ModelBatchSizeOptimizer
from unittest.mock import Mock

@pytest.fixture
def optimizer():
    handler = ModelBatchSizeOptimizer("sentence-transformers/all-MiniLM-L12-v2", use_accelerator=False)
    yield handler
    del handler.model
    torch.cuda.empty_cache()

def test_get_optimal_batch_size_single_sentence(optimizer):
    """
    Given a single sentence,
    When calculating the optimal batch size,
    Then it should return 1.
    """
    sample_sentences = ["This is a test sentence."]
    expected_batch_size = 1
    result = optimizer.get_optimal_batch_size(sample_sentences)
    assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

def test_get_optimal_batch_size_multiple_sentences(optimizer):
    """
    Given multiple sentences,
    When calculating the optimal batch size,
    Then it should return a value between 1 and 128.
    """
    sample_sentences = ["Sentence " + str(i) for i in range(100)]
    expected_batch_size_lower_bound = 1
    expected_batch_size_upper_bound = 128
    result = optimizer.get_optimal_batch_size(sample_sentences)
    assert expected_batch_size_lower_bound <= result <= expected_batch_size_upper_bound, \
        f"Expected batch size between {expected_batch_size_lower_bound} and {expected_batch_size_upper_bound}, got {result}"

def test_get_optimal_batch_size_empty_input(optimizer):
    """
    Given an empty list of sentences,
    When calculating the optimal batch size,
    Then it should return 1.
    """
    sample_sentences = []
    expected_batch_size = 1
    result = optimizer.get_optimal_batch_size(sample_sentences)
    assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

def test_get_optimal_batch_size_memory_constraint(mocker, optimizer):
    """
    Given a constrained memory environment (5 GB available),
    When calculating the optimal batch size,
    Then it should return a batch size <= 32.
    """
    mocker.patch('psutil.virtual_memory', return_value=mocker.Mock(available=5 * 1024 * 1024 * 1024))
    sample_sentences = ["Sentence " + str(i) for i in range(50)]
    expected_batch_size_upper_bound = 32
    result = optimizer.get_optimal_batch_size(sample_sentences)
    assert result <= expected_batch_size_upper_bound, f"Expected batch size <= {expected_batch_size_upper_bound}, got {result}"

def test_generate_embeddings_main_block(optimizer):
    """
    Given a list of sentences,
    When generating embeddings,
    Then the output should have the correct shape, device, and no NaN values.
    """
    sample_sentences = [
        "This is a test sentence.",
        "Another sentence for embedding."
    ]
    batch_size = 2
    expected_shape = torch.Size([2, 384])
    result = optimizer.generate_embeddings(sample_sentences, batch_size)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    assert result.device.type == optimizer.device.type, f"Expected device {optimizer.device.type}, got {result.device.type}"
    assert not torch.isnan(result).any(), "Expected no NaN values in embeddings, but found NaNs"

def test_device_selection(mocker):
    """
    Given different device availability scenarios,
    When initializing ModelBatchSizeOptimizer,
    Then it should select the appropriate device (cuda, mps, or cpu).
    """
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    
    # Mock device properties for CUDA
    mock_device_properties = Mock()
    mock_device_properties.total_memory = 6 * 1024 * 1024 * 1024  # 6 GB for GTX 1660
    
    # Test CUDA
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.get_device_properties', return_value=mock_device_properties)
    mocker.patch('torch.backends.mps.is_available', return_value=False)
    optimizer = ModelBatchSizeOptimizer(model_name, use_accelerator=True)
    assert optimizer.device.type == "cuda", f"Expected device cuda, got {optimizer.device.type}"
    
    # Test MPS
    mocker.patch('torch.cuda.is_available', return_value=False)
    mocker.patch('torch.backends.mps.is_available', return_value=True)
    optimizer = ModelBatchSizeOptimizer(model_name, use_accelerator=True)
    assert optimizer.device.type == "mps", f"Expected device mps, got {optimizer.device.type}"
    
    # Test CPU (no accelerators available)
    mocker.patch('torch.cuda.is_available', return_value=False)
    mocker.patch('torch.backends.mps.is_available', return_value=False)
    optimizer = ModelBatchSizeOptimizer(model_name, use_accelerator=True)
    assert optimizer.device.type == "cpu", f"Expected device cpu, got {optimizer.device.type}"
    
    # Test CPU (accelerator disabled)
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.get_device_properties', return_value=mock_device_properties)
    mocker.patch('torch.backends.mps.is_available', return_value=True)
    optimizer = ModelBatchSizeOptimizer(model_name, use_accelerator=False)
    assert optimizer.device.type == "cpu", f"Expected device cpu, got {optimizer.device.type}"
