use std::fs::File;
use std::io::Write;
use transcribe_rs::punct::add_punctuation;

fn main() {
    let mut file = File::create("/Users/zac/clawd/transcribe-rs/punct_result.txt").unwrap();

    // Test with proper spacing in input
    let texts = vec![
        // Input with proper English spacing
        "hello world how are you today",
        "this is a test for english punctuation",
        // Mixed with Chinese
        "你好吗 how are you today 我很好",
        "今天天气不错 we should go outside",
        "这是测试 test for mixed language",
        // Original paraformer output (no spaces in English)
        "让我测试一条完整的语音这条语音其实包含了englishandchinesedoyouknowthechinesemeans",
    ];

    writeln!(file, "=== Punctuation Test Results ===").unwrap();
    for text in texts {
        let result = add_punctuation(text);
        writeln!(file, "Input:  {}", text).unwrap();
        writeln!(file, "Output: {}", result).unwrap();
        writeln!(file, "").unwrap();
        println!("Input:  {}", text);
        println!("Output: {}", result);
        println!("");
    }
}
