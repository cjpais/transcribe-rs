use std::path::Path;
use transcribe_rs::punct::PunctModel;

const DEFAULT_PUNCT_MODEL: &str =
    "models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = PunctModel::new(Path::new(DEFAULT_PUNCT_MODEL))?;

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

    println!("=== Punctuation Test Results ===");
    for text in texts {
        let result = model.add_punctuation(text);
        println!("Input:  {}", text);
        println!("Output: {}", result);
        println!();
    }

    Ok(())
}
