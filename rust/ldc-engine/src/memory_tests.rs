#[cfg(test)]
mod memory_tests {
    use crate::{
        TrainingSample, FeatureSeries, Direction, OptimizedTrainingSample,
        MemoryPool, MemoryThresholdMonitor, MemoryStatus, LDCEngine
    };

    #[test]
    fn test_optimized_training_sample_conversion() {
        let original_sample = TrainingSample {
            features: FeatureSeries {
                f1: 1.0,
                f2: 2.0,
                f3: 3.0,
                f4: 4.0,
                f5: 5.0,
            },
            label: Direction::Long,
            timestamp: 1234567890,
            bar_index: 42,
        };

        let optimized_sample = OptimizedTrainingSample::from_training_sample(&original_sample);
        let converted_back = optimized_sample.to_training_sample();

        assert_eq!(original_sample.features.f1, converted_back.features.f1);
        assert_eq!(original_sample.features.f2, converted_back.features.f2);
        assert_eq!(original_sample.features.f3, converted_back.features.f3);
        assert_eq!(original_sample.features.f4, converted_back.features.f4);
        assert_eq!(original_sample.features.f5, converted_back.features.f5);
        assert_eq!(original_sample.label, converted_back.label);
        assert_eq!(original_sample.timestamp, converted_back.timestamp);
        assert_eq!(original_sample.bar_index, converted_back.bar_index);
    }

    #[test]
    fn test_memory_pool_basic_operations() {
        let mut pool = MemoryPool::new(1).unwrap(); // 1MB pool

        // Test allocation
        let ptr1 = pool.allocate(64, 8);
        assert!(ptr1.is_ok());
        assert_eq!(pool.allocation_count(), 1);
        assert!(pool.allocated_bytes() >= 64);

        // Test deallocation
        if let Ok(ptr) = ptr1 {
            assert!(pool.deallocate(ptr));
            assert_eq!(pool.deallocation_count(), 1);
        }

        // Test utilization
        let utilization = pool.utilization_percent();
        assert!(utilization >= 0.0 && utilization <= 100.0);
    }

    #[test]
    fn test_memory_threshold_monitor() {
        let mut monitor = MemoryThresholdMonitor::new(100, 80.0, 95.0); // 100MB threshold

        // Test normal usage
        let status = monitor.check_memory_usage(50); // 50MB usage
        assert!(matches!(status, MemoryStatus::Normal));

        // Test warning threshold
        let status = monitor.check_memory_usage(85); // 85MB usage (85% of 100MB)
        match status {
            MemoryStatus::Warning { usage_percent, usage_mb } => {
                assert_eq!(usage_mb, 85);
                assert!(usage_percent >= 80.0);
            }
            MemoryStatus::Normal => {
                // This might happen due to timing constraints in check_memory_usage
                // Let's force a check by creating a new monitor
                let mut new_monitor = MemoryThresholdMonitor::new(100, 80.0, 95.0);
                new_monitor.last_check_timestamp = std::time::Instant::now() - std::time::Duration::from_secs(2);
                let status = new_monitor.check_memory_usage(85);
                if let MemoryStatus::Warning { usage_percent, usage_mb } = status {
                    assert_eq!(usage_mb, 85);
                    assert!(usage_percent >= 80.0);
                } else {
                    panic!("Expected warning status after forced check");
                }
            }
            _ => panic!("Expected warning or normal status"),
        }

        // Test critical threshold
        let mut critical_monitor = MemoryThresholdMonitor::new(100, 80.0, 95.0);
        critical_monitor.last_check_timestamp = std::time::Instant::now() - std::time::Duration::from_secs(2);
        let status = critical_monitor.check_memory_usage(98); // 98MB usage (98% of 100MB)
        if let MemoryStatus::Critical { usage_percent, usage_mb } = status {
            assert_eq!(usage_mb, 98);
            assert!(usage_percent >= 95.0);
        } else {
            panic!("Expected critical status");
        }
    }

    #[test]
    fn test_ldc_engine_optimized_storage() {
        let mut engine = LDCEngine::new();

        // Add some training samples
        let sample1 = TrainingSample {
            features: FeatureSeries { f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0 },
            label: Direction::Long,
            timestamp: 1000,
            bar_index: 1,
        };

        let sample2 = TrainingSample {
            features: FeatureSeries { f1: 2.0, f2: 3.0, f3: 4.0, f4: 5.0, f5: 6.0 },
            label: Direction::Short,
            timestamp: 2000,
            bar_index: 2,
        };

        engine.add_training_sample(sample1.clone());
        engine.add_training_sample(sample2.clone());

        assert_eq!(engine.training_samples_count(), 2);

        // Enable optimized storage
        engine.enable_optimized_storage().unwrap();
        assert!(engine.use_optimized_storage);
        assert_eq!(engine.optimized_samples.len(), 2);

        // Test getting samples for search
        let samples = engine.get_training_samples_for_search_optimized(None);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].features.f1, sample1.features.f1);
        assert_eq!(samples[1].features.f1, sample2.features.f1);

        // Disable optimized storage
        engine.disable_optimized_storage().unwrap();
        assert!(!engine.use_optimized_storage);
        assert_eq!(engine.training_samples.len(), 2);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let mut engine = LDCEngine::new();

        // Add samples and check memory tracking
        for i in 0..10 {
            let sample = TrainingSample {
                features: FeatureSeries { 
                    f1: i as f32, 
                    f2: (i + 1) as f32, 
                    f3: (i + 2) as f32, 
                    f4: (i + 3) as f32, 
                    f5: (i + 4) as f32 
                },
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: (1000 + i) as i64,
                bar_index: i,
            };
            engine.add_training_sample_optimized(sample).unwrap();
        }

        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_training_samples, 10);

        // Check memory pool stats if available
        if let Some((allocated, peak, alloc_count, dealloc_count, utilization)) = engine.get_memory_pool_stats() {
            assert!(utilization >= 0.0 && utilization <= 100.0);
            assert!(alloc_count >= 0);
            assert!(dealloc_count >= 0);
            assert!(peak >= allocated);
        }
    }
}