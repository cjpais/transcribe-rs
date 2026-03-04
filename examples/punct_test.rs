use std::path::Path;
use transcribe_rs::punct::PunctModel;

const DEFAULT_PUNCT_MODEL: &str =
    "models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = PunctModel::new(Path::new(DEFAULT_PUNCT_MODEL))?;

    // 测试 sherpa-paraformer 的输出（无标点）
    let text = "让我测试一条完整的语音这条语音包含了englishandchinesedoyouknowthechinesemeans";
    let result = model.add_punctuation(text);
    println!("原文本: {}", text);
    println!("加标点: {}", result);
    println!();

    // 测试问句
    let question = "你叫什么名字";
    let result = model.add_punctuation(question);
    println!("问句: {} -> {}", question, result);

    Ok(())
}
