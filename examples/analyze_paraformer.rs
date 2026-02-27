use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn main() {
    let model_path = "models/sherpa-paraformer/model.int8.onnx";

    let session = Session::builder()
        .expect("Failed to create session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization level")
        .with_execution_providers(vec![CPUExecutionProvider::default().build()])
        .expect("Failed to set execution providers")
        .commit_from_file(model_path)
        .expect("Failed to load model");

    println!("=== Inputs ===");
    for input in &session.inputs {
        println!("{}: {:?}", input.name, input.input_type);
    }
    println!("=== Outputs ===");
    for output in &session.outputs {
        println!("{}: {:?}", output.name, output.output_type);
    }
}
