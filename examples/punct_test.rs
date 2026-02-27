use transcribe_rs::punct::add_punctuation;

fn main() {
    // 测试 sherpa-paraformer 的输出（无标点）
    let text = "让我测试一条完整的语音这条语音包含了englishandchinesedoyouknowthechinesemeans";

    let result = add_punctuation(text);

    println!("原文本: {}", text);
    println!("加标点: {}", result);
    println!();

    // 测试问句
    let question = "你叫什么名字";
    println!("问句: {} -> {}", question, add_punctuation(question));
}
