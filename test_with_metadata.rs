// Simple test to verify the with_metadata method signature changes
use std::collections::HashMap;

struct TestEvent {
    metadata: HashMap<String, String>,
}

impl TestEvent {
    fn new() -> Self {
        Self {
            metadata: HashMap::new(),
        }
    }
    
    // This is the new signature we implemented
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        let value = value.into();
        self.metadata.insert(key, value);
        self
    }
}

fn main() {
    // Test with String literals
    let event1 = TestEvent::new()
        .with_metadata("key1", "value1")
        .with_metadata("key2", "value2");
    
    // Test with owned Strings
    let key = String::from("key3");
    let value = String::from("value3");
    let event2 = TestEvent::new()
        .with_metadata(key, value);
    
    // Test with &str
    let event3 = TestEvent::new()
        .with_metadata("key4", "value4");
    
    // Test with mixed types
    let event4 = TestEvent::new()
        .with_metadata(String::from("key5"), "value5")
        .with_metadata("key6", String::from("value6"));
    
    println!("All tests passed! The new signature works with various string types.");
}