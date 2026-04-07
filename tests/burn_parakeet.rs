use burn::prelude::*;
use std::path::Path;

type B = burn::backend::Wgpu;

/// Load a .npy file as a flat Vec<f32>.
/// Minimal npy parser — handles only f32 little-endian C-contiguous arrays.
fn load_npy_flat(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        panic!("Failed to read {path}: {e}. Run `uv run --with 'parakeet-mlx' --with numpy scripts/dump_reference_tensors.py` first.");
    });
    // npy format: 6-byte magic + 2-byte version + 2-byte header_len + header + data
    assert_eq!(&data[..6], b"\x93NUMPY", "Not a .npy file: {path}");
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let data_start = 10 + header_len;
    let float_data = &data[data_start..];
    float_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare two flat f32 slices, print stats, return (max_abs_err, mean_abs_err).
fn compare(name: &str, ours: &[f32], reference: &[f32], tol: f32) -> (f32, f32) {
    assert_eq!(
        ours.len(),
        reference.len(),
        "{name}: length mismatch: ours={} ref={}",
        ours.len(),
        reference.len()
    );

    let mut max_err: f32 = 0.0;
    let mut sum_err: f64 = 0.0;
    let mut max_idx = 0;

    for (i, (&a, &b)) in ours.iter().zip(reference.iter()).enumerate() {
        let err = (a - b).abs();
        sum_err += err as f64;
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }

    let mean_err = sum_err / ours.len() as f64;
    let status = if max_err <= tol { "PASS" } else { "FAIL" };

    println!(
        "[{status}] {name}: max_err={max_err:.6} (at idx {max_idx}, ours={:.6} ref={:.6}), mean_err={mean_err:.6}, tol={tol}",
        ours[max_idx], reference[max_idx]
    );

    (max_err, mean_err as f32)
}

#[test]
fn compare_with_mlx_reference() {
    let _ = env_logger::try_init();

    let ref_dir = Path::new("scripts/reference_tensors");
    if !ref_dir.exists() {
        panic!(
            "Reference tensors not found. Run:\n  \
             uv run --with 'parakeet-mlx' --with numpy scripts/dump_reference_tensors.py"
        );
    }

    let model_dir = Path::new("models/parakeet-tdt-0.6b-v3");
    let wav_path = Path::new("samples/dots.wav");

    // --- Load model ---
    println!("\n=== Loading model ===");
    let model = transcribe_rs::burn_parakeet::BurnParakeetModel::load(model_dir)
        .expect("Failed to load model");

    // --- Load audio ---
    let samples = transcribe_rs::audio::read_wav_samples(wav_path).unwrap();
    println!("Audio: {} samples ({:.2}s)", samples.len(), samples.len() as f64 / 16000.0);

    // =================================================================
    // Stage 1: Mel features
    // =================================================================
    println!("\n=== Stage 1: Mel features ===");
    let mel = transcribe_rs::burn_parakeet::preprocess::nemo_preprocess(&samples);
    let num_mels = mel.nrows();
    let num_frames = mel.ncols();
    println!("Our mel: [{num_mels}, {num_frames}]");

    let ref_mel = load_npy_flat("scripts/reference_tensors/mel.npy");
    // Our mel is [num_mels, num_frames] (freq-major), ref is [1, T, 128] (time-major)
    // Convert ours to time-major for comparison
    let our_mel_time_major: Vec<f32> = mel.t().iter().copied().collect();

    let min_len = our_mel_time_major.len().min(ref_mel.len());
    println!("Comparing {min_len} values (our={}, ref={})", our_mel_time_major.len(), ref_mel.len());
    let (mel_max_err, _) = compare("mel", &our_mel_time_major[..min_len], &ref_mel[..min_len], 0.5);

    if mel_max_err > 0.5 {
        println!("\nSample mel values (first frame, first 10 bins):");
        for i in 0..10.min(num_mels) {
            println!("  bin {i}: ours={:.4} ref={:.4}", our_mel_time_major[i], ref_mel[i]);
        }
        println!("\nStopping at mel — fix preprocessing first.");
        return;
    }

    // =================================================================
    // Stage 2: Pre-encode
    // =================================================================
    println!("\n=== Stage 2: Pre-encode ===");
    let mel_data: Vec<f32> = mel.t().iter().copied().collect();
    let mel_flat: Tensor<B, 1> = Tensor::from_floats(&mel_data[..], &model.device);
    let mel_tensor: Tensor<B, 3> = mel_flat.reshape([1, num_frames, num_mels]);

    let pre_enc_out = model.encoder.pre_encode.forward(mel_tensor.clone());
    let pre_enc_data: Vec<f32> = pre_enc_out.to_data().to_vec().unwrap();
    let pre_enc_dims = pre_enc_out.dims();
    println!("Our pre_encode: {:?}", pre_enc_dims);

    let ref_pre_enc = load_npy_flat("scripts/reference_tensors/pre_encode.npy");
    let min_len = pre_enc_data.len().min(ref_pre_enc.len());
    let (pre_enc_err, _) = compare("pre_encode", &pre_enc_data[..min_len], &ref_pre_enc[..min_len], 10.0);

    if pre_enc_err > 10.0 {
        println!("\nSample pre_encode values (t=0, first 10 dims):");
        for i in 0..10 {
            println!("  dim {i}: ours={:.4} ref={:.4}", pre_enc_data[i], ref_pre_enc[i]);
        }
        println!("\nStopping at pre_encode — fix this stage first.");
        return;
    }

    // =================================================================
    // Stage 3: Positional encoding
    // =================================================================
    println!("\n=== Stage 3: Positional encoding ===");
    let pre_enc_t = pre_enc_dims[1];
    let pos_emb = model.encoder.make_pos_encoding(pre_enc_t, &model.device);
    let pos_emb_data: Vec<f32> = pos_emb.to_data().to_vec().unwrap();
    println!("Our pos_emb: {:?}", pos_emb.dims());

    let ref_pos_emb = load_npy_flat("scripts/reference_tensors/pos_emb.npy");
    let min_len = pos_emb_data.len().min(ref_pos_emb.len());
    compare("pos_emb", &pos_emb_data[..min_len], &ref_pos_emb[..min_len], 1e-4);

    // =================================================================
    // Stage 4: Conformer layer 0 (sub-stages)
    // =================================================================
    println!("\n=== Stage 4: Conformer layer 0 ===");
    let layer = &model.encoder.layers[0];

    // 4a: FF1
    let x_l = pre_enc_out.clone();
    let x_l = x_l.clone() + layer.ff1.forward(layer.norm_ff1.forward(x_l)) * 0.5;
    let ff1_data: Vec<f32> = x_l.to_data().to_vec().unwrap();
    let ref_ff1 = load_npy_flat("scripts/reference_tensors/layer0_after_ff1.npy");
    let (ff1_err, _) = compare("layer0_after_ff1", &ff1_data[..ff1_data.len().min(ref_ff1.len())], &ref_ff1[..ff1_data.len().min(ref_ff1.len())], 1.0);
    if ff1_err > 1.0 {
        println!("Stopping at FF1.");
        return;
    }

    // 4b: Self-attention
    let x_norm = layer.norm_attn.forward(x_l.clone());
    let attn_out = layer.attn.forward(x_norm, pos_emb.clone(), None);
    let attn_data: Vec<f32> = attn_out.to_data().to_vec().unwrap();
    let ref_attn = load_npy_flat("scripts/reference_tensors/layer0_attn_out.npy");
    let (attn_err, _) = compare("layer0_attn_out", &attn_data[..attn_data.len().min(ref_attn.len())], &ref_attn[..attn_data.len().min(ref_attn.len())], 5.0);
    let x_l = x_l + attn_out;
    if attn_err > 5.0 {
        println!("Stopping at attention.");
        return;
    }

    // 4c: Conv module — compare conv.forward() vs manual
    let conv_full_out = layer.conv.forward(layer.norm_conv.forward(x_l.clone()));
    let conv_full_data: Vec<f32> = conv_full_out.to_data().to_vec().unwrap();

    let conv = &layer.conv;
    let cx = layer.norm_conv.forward(x_l.clone());
    // Transpose to [B, C, T] for Conv1d
    let cx = cx.swap_dims(1, 2);

    // pw1
    let cx = conv.pw1.forward(cx);
    // Compare pw1 output (transpose back to [B, T, C] for comparison with MLX)
    let cx_for_cmp = cx.clone().swap_dims(1, 2);
    let cx_data: Vec<f32> = cx_for_cmp.to_data().to_vec().unwrap();
    let ref_pw1 = load_npy_flat("scripts/reference_tensors/layer0_conv_after_pw1.npy");
    let (pw1_err, _) = compare("conv_pw1", &cx_data[..cx_data.len().min(ref_pw1.len())], &ref_pw1[..cx_data.len().min(ref_pw1.len())], 1.0);
    if pw1_err > 1.0 { println!("Stopping at conv pw1."); return; }

    // GLU
    let [b, c2, t] = cx.dims();
    let c = c2 / 2;
    let a = cx.clone().slice([0..b, 0..c, 0..t]);
    let gate = cx.slice([0..b, c..c2, 0..t]);
    let cx = a * burn::tensor::activation::sigmoid(gate);
    let cx_for_cmp = cx.clone().swap_dims(1, 2);
    let cx_data: Vec<f32> = cx_for_cmp.to_data().to_vec().unwrap();
    let ref_glu = load_npy_flat("scripts/reference_tensors/layer0_conv_after_glu.npy");
    let (glu_err, _) = compare("conv_glu", &cx_data[..cx_data.len().min(ref_glu.len())], &ref_glu[..cx_data.len().min(ref_glu.len())], 1.0);
    if glu_err > 1.0 { println!("Stopping at conv glu."); return; }

    // Depthwise conv (with fused BN — compare against post-BN reference)
    let cx = conv.dw.forward(cx);

    // Identity BN (fused into DW conv, so BN is ~no-op)
    let d_model = conv.bn_weight.val().dims()[0];
    let eps = 1e-5;
    let mean = conv.bn_running_mean.val().reshape([1, d_model, 1]);
    let var = conv.bn_running_var.val().reshape([1, d_model, 1]);
    let w = conv.bn_weight.val().reshape([1, d_model, 1]);
    let bi = conv.bn_bias.val().reshape([1, d_model, 1]);
    let cx = (cx - mean) / (var + eps).sqrt() * w + bi;

    let cx_for_cmp = cx.clone().swap_dims(1, 2);
    let cx_data: Vec<f32> = cx_for_cmp.to_data().to_vec().unwrap();
    let ref_bn = load_npy_flat("scripts/reference_tensors/layer0_conv_after_bn.npy");
    let (bn_err, _) = compare("conv_dw_fused_bn", &cx_data[..cx_data.len().min(ref_bn.len())], &ref_bn[..cx_data.len().min(ref_bn.len())], 1.0);
    if bn_err > 1.0 { println!("Stopping at conv dw+bn."); return; }

    // SiLU + pw2
    let cx = cx.clone() * burn::tensor::activation::sigmoid(cx);
    let cx = conv.pw2.forward(cx);
    let conv_out = cx.swap_dims(1, 2);  // back to [B, T, C]
    let conv_data: Vec<f32> = conv_out.to_data().to_vec().unwrap();
    let ref_conv = load_npy_flat("scripts/reference_tensors/layer0_conv_out.npy");
    let (conv_err, _) = compare("conv_out_manual", &conv_data[..conv_data.len().min(ref_conv.len())], &ref_conv[..conv_data.len().min(ref_conv.len())], 50.0);
    compare("conv_full_vs_manual", &conv_full_data[..conv_full_data.len().min(conv_data.len())], &conv_data[..conv_full_data.len().min(conv_data.len())], 0.01);
    let x_l = x_l + conv_out;
    if conv_err > 50.0 {
        println!("Stopping at conv module.");
        return;
    }

    // 4d: FF2
    let x_l = x_l.clone() + layer.ff2.forward(layer.norm_ff2.forward(x_l)) * 0.5;
    let ff2_data: Vec<f32> = x_l.to_data().to_vec().unwrap();
    let ref_ff2 = load_npy_flat("scripts/reference_tensors/layer0_after_ff2.npy");
    let (_ff2_err, _) = compare("layer0_after_ff2", &ff2_data[..ff2_data.len().min(ref_ff2.len())], &ref_ff2[..ff2_data.len().min(ref_ff2.len())], 200.0);

    // 4e: Final norm
    let layer0_out = layer.norm_out.forward(x_l);
    let layer0_data: Vec<f32> = layer0_out.to_data().to_vec().unwrap();
    println!("Our layer0: {:?}", layer0_out.dims());

    let ref_layer0 = load_npy_flat("scripts/reference_tensors/layer0.npy");
    let min_len = layer0_data.len().min(ref_layer0.len());
    let (_l0_err, _) = compare("layer0", &layer0_data[..min_len], &ref_layer0[..min_len], 200.0);

    // Debug: check Conv1d bias state
    println!("pw1 bias: {}", model.encoder.layers[0].conv.pw1.bias.is_some());
    println!("dw bias: {}", model.encoder.layers[0].conv.dw.bias.is_some());
    println!("pw2 bias: {}", model.encoder.layers[0].conv.pw2.bias.is_some());

    // Step-by-step direct forward to find where it diverges
    println!("\n--- Direct forward step-by-step ---");
    {
        let layer = &model.encoder.layers[0];
        let x = pre_enc_out.clone();
        // FF1
        let r = x.clone();
        let x = r + layer.ff1.forward(layer.norm_ff1.forward(x)) * 0.5;
        let xd: Vec<f32> = x.to_data().to_vec().unwrap();
        compare("direct_ff1", &xd, &ff1_data, 0.001);

        // Attention
        let r = x.clone();
        let attn = layer.attn.forward(layer.norm_attn.forward(x), pos_emb.clone(), None);
        let x = r + attn;
        let xd_after_attn: Vec<f32> = x.to_data().to_vec().unwrap();
        let ref_after_attn = load_npy_flat("scripts/reference_tensors/layer0_after_attn.npy");
        compare("direct_after_attn", &xd_after_attn[..xd_after_attn.len().min(ref_after_attn.len())], &ref_after_attn[..xd_after_attn.len().min(ref_after_attn.len())], 1.0);

        // Conv — split into norm + conv to compare inputs
        let r = x.clone();
        let norm_out = layer.norm_conv.forward(x);
        let norm_data: Vec<f32> = norm_out.to_data().to_vec().unwrap();
        // Compare with manual norm_conv output (saved from manual path as cx before transpose)
        // Actually, compare r (the residual) against what we had manually
        let r_data: Vec<f32> = r.to_data().to_vec().unwrap();
        compare("direct_r_vs_after_attn_ref", &r_data[..r_data.len().min(ref_after_attn.len())], &ref_after_attn[..r_data.len().min(ref_after_attn.len())], 1.0);

        let conv_result = layer.conv.forward(norm_out);
        let conv_result_data: Vec<f32> = conv_result.to_data().to_vec().unwrap();
        compare("direct_conv_result_vs_ref", &conv_result_data[..conv_result_data.len().min(ref_conv.len())], &ref_conv[..conv_result_data.len().min(ref_conv.len())], 1.0);

        let x = r + conv_result;
        let xd_after_conv: Vec<f32> = x.to_data().to_vec().unwrap();
        let ref_after_conv = load_npy_flat("scripts/reference_tensors/layer0_after_conv.npy");
        compare("direct_after_conv", &xd_after_conv[..xd_after_conv.len().min(ref_after_conv.len())], &ref_after_conv[..xd_after_conv.len().min(ref_after_conv.len())], 1.0);

        // FF2
        let r = x.clone();
        let x = r + layer.ff2.forward(layer.norm_ff2.forward(x)) * 0.5;
        let xd_after_ff2: Vec<f32> = x.to_data().to_vec().unwrap();
        let ref_after_ff2 = load_npy_flat("scripts/reference_tensors/layer0_after_ff2.npy");
        compare("direct_after_ff2", &xd_after_ff2[..xd_after_ff2.len().min(ref_after_ff2.len())], &ref_after_ff2[..xd_after_ff2.len().min(ref_after_ff2.len())], 1.0);

        // norm_out
        let x = layer.norm_out.forward(x);
        let xd_final: Vec<f32> = x.to_data().to_vec().unwrap();
        compare("direct_layer0_final", &xd_final, &layer0_data, 0.001);
    }

    // =================================================================
    // Stage 4.5: Layers 1-5
    // =================================================================
    println!("\n=== Stage 4.5: Layers 1-5 ===");
    let mut x_running = layer0_out.clone();
    for li in 1..=5 {
        x_running = model.encoder.layers[li].forward(x_running, pos_emb.clone(), None);
        let data: Vec<f32> = x_running.to_data().to_vec().unwrap();
        let ref_path = format!("scripts/reference_tensors/layer{li}.npy");
        if let Ok(_) = std::fs::metadata(&ref_path) {
            let ref_data = load_npy_flat(&ref_path);
            let min_len = data.len().min(ref_data.len());
            compare(&format!("layer{li}"), &data[..min_len], &ref_data[..min_len], 200.0);
        }
    }

    // =================================================================
    // Stage 5: Full encoder
    // =================================================================
    println!("\n=== Stage 5: Full encoder ===");
    let encoder_out = model.encoder.forward(mel_tensor);
    let enc_data: Vec<f32> = encoder_out.to_data().to_vec().unwrap();
    println!("Our encoder: {:?}", encoder_out.dims());

    let ref_enc = load_npy_flat("scripts/reference_tensors/encoder_out.npy");
    let min_len = enc_data.len().min(ref_enc.len());
    let (enc_err, _) = compare("encoder_out", &enc_data[..min_len], &ref_enc[..min_len], 0.05);

    if enc_err > 0.05 {
        println!("\nSample encoder_out values (t=0, first 10 dims):");
        for i in 0..10 {
            println!("  dim {i}: ours={:.4} ref={:.4}", enc_data[i], ref_enc[i]);
        }
    }

    // =================================================================
    // Stage 6: Joint logits at t=0
    // =================================================================
    println!("\n=== Stage 6: Joint logits at t=0 ===");
    let enc_dim = encoder_out.dims()[2];
    let enc_t0 = encoder_out.clone().slice([0..1, 0..1, 0..enc_dim]).reshape([1, enc_dim]);
    let pred_zeros: Tensor<B, 2> = Tensor::zeros([1, 640], &model.device);
    let joint_logits = model.joint.forward(enc_t0, pred_zeros);
    let joint_data: Vec<f32> = joint_logits.to_data().to_vec().unwrap();

    let ref_joint = load_npy_flat("scripts/reference_tensors/joint_logits_t0.npy");
    let min_len = joint_data.len().min(ref_joint.len());
    compare("joint_logits_t0", &joint_data[..min_len], &ref_joint[..min_len], 1.0);

    // Print our top-5
    let mut indexed: Vec<(usize, f32)> = joint_data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nOur top-5 logits:");
    for &(idx, val) in &indexed[..5] {
        println!("  [{idx}] {val:.4}");
    }

    println!("\nDone!");
}
