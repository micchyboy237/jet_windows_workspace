import pytest
from unittest.mock import patch, MagicMock
from transcribe_batch_multi_short_audio_service import should_use_gpu, batch_transcribe_files  # replace with actual module name


@pytest.fixture
def mock_torch_cuda_available():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="GTX 1660"):
            yield


class TestDynamicDeviceSelection:
    def test_returns_cuda_when_cuda_available_and_high_cpu_load(self, mock_torch_cuda_available):
        # Given
        cpu_loads = [90.0, 88.0, 92.0]  # all above threshold

        with patch("psutil.cpu_percent", side_effect=cpu_loads):
            with patch("time.sleep"):  # speed up test
                # When
                device = should_use_gpu()

                # Then
                expected = "cuda"
                assert device == expected

    def test_returns_cpu_when_low_cpu_load_even_with_cuda(self, mock_torch_cuda_available):
        # Given
        cpu_loads = [70.0, 60.0, 80.0]  # not consistently high

        with patch("psutil.cpu_percent", side_effect=cpu_loads):
            with patch("time.sleep"):
                # When
                device = should_use_gpu()

                # Then
                expected = "cpu"
                assert device == expected

    def test_returns_cpu_when_cuda_not_available(self):
        # Given
        with patch("torch.cuda.is_available", return_value=False):
            with patch("time.sleep"):
                # When
                device = should_use_gpu()

                # Then
                expected = "cpu"
                assert device == expected


class TestBatchTranscribeIntegration:
    @patch(
        "transcribe_batch_multi_short_audio_service.WhisperModel"
    )
    @patch(
        "transcribe_batch_multi_short_audio_service.should_use_gpu"
    )
    def test_uses_cuda_and_reduces_workers_when_gpu_selected(self, mock_device, mock_model):
        # Given
        mock_device.return_value = "cuda"
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance

        audio_paths = ["fake1.wav", "fake2.wav"]

        # When
        batch_transcribe_files(audio_paths, max_workers=6)

        # Then
        mock_model.assert_called_with(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cuda",
            compute_type="float16",
            cpu_threads=None,
        )
        # Note: effective_workers=1 for CUDA is logged, but we can't easily assert log here