"""
src/mivlm/models/loading.py의 device 선택 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_loading.py -v
"""
import torch

from mivlm.models.loading import pick_device


def _fake_hardware(monkeypatch, cuda, cuda_bf16, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: cuda_bf16)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_cuda_with_bf16(monkeypatch):
    _fake_hardware(monkeypatch, cuda=True, cuda_bf16=True, mps=False)

    assert pick_device() == ("cuda", torch.bfloat16)


def test_cuda_without_bf16_falls_back_to_fp16(monkeypatch):
    _fake_hardware(monkeypatch, cuda=True, cuda_bf16=False, mps=False)

    assert pick_device() == ("cuda", torch.float16)


def test_cuda_is_preferred_over_mps(monkeypatch):
    _fake_hardware(monkeypatch, cuda=True, cuda_bf16=True, mps=True)

    assert pick_device() == ("cuda", torch.bfloat16)


def test_mps_when_no_cuda(monkeypatch):
    _fake_hardware(monkeypatch, cuda=False, cuda_bf16=False, mps=True)

    assert pick_device() == ("mps", torch.bfloat16)


def test_cpu_when_nothing_else(monkeypatch):
    _fake_hardware(monkeypatch, cuda=False, cuda_bf16=False, mps=False)

    assert pick_device() == ("cpu", torch.float32)
