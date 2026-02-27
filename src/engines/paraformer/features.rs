use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};

#[derive(Debug, Clone)]
pub struct FbankConfig {
    pub num_bins: usize,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: i32,
    pub low_freq: f32,
    pub high_freq: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            num_bins: 80,
            fft_size: 512,
            window_size: 400,
            hop_size: 160,
            sample_rate: 16000,
            low_freq: 0.0,
            high_freq: 8000.0,
        }
    }
}

fn compute_mel_filterbank(config: &FbankConfig) -> Vec<Vec<f32>> {
    let num_bins = config.num_bins;
    let fft_size = config.fft_size;
    let sample_rate = config.sample_rate as f32;
    let low_freq = config.low_freq;
    let high_freq = config.high_freq;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let low_mel = hz_to_mel(low_freq);
    let high_mel = hz_to_mel(high_freq);

    let num_points = num_bins + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz * fft_size as f32) / sample_rate).floor() as usize)
        .collect();

    let mut filterbank = vec![vec![0.0f32; fft_size / 2 + 1]; num_bins];
    for (i, filter) in filterbank.iter_mut().enumerate() {
        let left = fft_bins[i];
        let center = fft_bins[i + 1];
        let right = fft_bins[i + 2];

        for j in left..center {
            if j < filter.len() {
                filter[j] = (j - left) as f32 / (center - left) as f32;
            }
        }
        for j in center..right {
            if j < filter.len() {
                filter[j] = (right - j) as f32 / (right - center) as f32;
            }
        }
    }

    filterbank
}

pub fn compute_fbank(samples: &[f32], config: &FbankConfig) -> Array2<f32> {
    if samples.len() < config.window_size {
        return Array2::zeros((0, config.num_bins));
    }

    let num_frames = (samples.len() - config.window_size) / config.hop_size + 1;
    let filterbank = compute_mel_filterbank(config);
    let fft_size = config.fft_size;
    let half_fft = fft_size / 2 + 1;

    let window: Vec<f32> = (0..config.window_size)
        .map(|i| {
            0.54 - 0.46
                * (2.0 * std::f32::consts::PI * i as f32 / (config.window_size - 1) as f32).cos()
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut features = Vec::with_capacity(num_frames * config.num_bins);
    for frame_idx in 0..num_frames {
        let start = frame_idx * config.hop_size;

        let mut buffer: Vec<Complex<f32>> = samples[start..start + config.window_size]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        buffer.resize(fft_size, Complex::new(0.0, 0.0));

        fft.process(&mut buffer);

        let mut power = vec![0.0f32; half_fft];
        for (i, c) in buffer.iter().take(half_fft).enumerate() {
            power[i] = c.norm_sqr();
        }

        for filter in &filterbank {
            let mut sum = 0.0f32;
            for (i, &w) in filter.iter().take(half_fft).enumerate() {
                sum += w * power[i];
            }
            features.push(if sum > 1e-10 {
                10.0 * sum.log10()
            } else {
                -80.0
            });
        }
    }

    Array2::from_shape_vec((num_frames, config.num_bins), features).unwrap()
}

pub fn apply_lfr(features: &Array2<f32>, window_size: usize, window_shift: usize) -> Array2<f32> {
    let num_frames = features.nrows();
    let feat_dim = features.ncols();

    if num_frames < window_size || window_size == 0 || window_shift == 0 {
        return Array2::zeros((0, feat_dim.saturating_mul(window_size)));
    }

    let num_output_frames = (num_frames - window_size) / window_shift + 1;
    let mut output = Array2::<f32>::zeros((num_output_frames, feat_dim * window_size));

    for i in 0..num_output_frames {
        let start = i * window_shift;
        let frame_data = features.slice(ndarray::s![start..start + window_size, ..]);
        for j in 0..window_size {
            for k in 0..feat_dim {
                output[[i, j * feat_dim + k]] = frame_data[[j, k]];
            }
        }
    }

    output
}

pub fn apply_mean_cmvn(features: &mut Array2<f32>, mean: &Array1<f32>) {
    if features.ncols() == 0 || mean.is_empty() {
        return;
    }

    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            if j < mean.len() {
                features[[i, j]] -= mean[j];
            }
        }
    }
}
