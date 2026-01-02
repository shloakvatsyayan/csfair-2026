import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from legacy_objtracking.DeviceDetector import DeviceDetector


class DeviceTester:

    def __init__(self):
        self._detector = DeviceDetector()

    def test(self):
        device_info = self._detector.get_device_info()
        device = self._detector.get_device()

        print("=" * 50)
        print("DEVICE DETECTION RESULTS")
        print("=" * 50)
        print(f"Selected device: {device.upper()}")
        print(f"Running on CPU: {device_info['is_cpu']}")
        print(f"Running on GPU (CUDA): {device_info['is_cuda']}")
        print(f"Running on GPU (MPS - macOS): {device_info['is_mps']}")
        print()

        if device_info['is_cuda']:
            print("CUDA Details:")
            print(f"  CUDA version: {device_info['cuda_version']}")
            print(f"  Device count: {device_info['device_count']}")
            print("  Device names:")
            for name in device_info['device_names']:
                print(f"    - {name}")
            print()

        if device_info['is_mps']:
            print("MPS (Metal Performance Shaders) Details:")
            print("  MPS is available and will be used for GPU acceleration")
            print("  This is macOS's GPU backend (Metal)")
            print()

        if device_info['is_cpu']:
            print("CPU Details:")
            print("  Running on CPU - no GPU acceleration available")
            print()

        # Test actual device usage
        print("Testing device with a sample tensor:")
        test_tensor = torch.randn(3, 3)
        test_tensor = test_tensor.to(device)
        print(f"  Tensor device: {test_tensor.device}")
        print(f"  Tensor created successfully on {device.upper()}")


def main():
    tester = DeviceTester()
    tester.test()


if __name__ == "__main__":
    main()

