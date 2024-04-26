use candle_core::utils::{cuda_is_available, metal_is_available};
use glowrs::{Device, Result};

pub fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build polars-candle with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build polars-candle with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_device() {
        let device = get_device(true).unwrap();
        assert!(matches!(device, Device::Cpu));

        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            let device = get_device(false).unwrap();
            assert!(matches!(device, Device::Cpu));
        }

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            let device = get_device(false).unwrap();
            assert!(matches!(device, Device::Cuda(_)));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let device = get_device(false).unwrap();
            assert!(matches!(device, Device::Metal(_)));
        }
    }
}
