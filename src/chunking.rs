use crate::vad::SileroVad;
use anyhow::Result;
use log::{debug, info, warn};
use std::path::Path;

pub struct SmartChunker;

impl SmartChunker {
    pub fn chunk_audio<F, P>(
        audio: &[f32],
        vad_model_path: &Path,
        mut callback: F,
        mut progress_callback: P,
    ) -> Result<String>
    where
        F: FnMut(Vec<f32>) -> Result<String>,
        P: FnMut(f64),
    {
        // Smart Chunking Configuration
        const TARGET_CHUNK_DURATION_SECONDS: usize = 30;
        const SAMPLE_RATE: usize = 16000;
        const TARGET_CHUNK_SIZE: usize = TARGET_CHUNK_DURATION_SECONDS * SAMPLE_RATE;

        // Initialize VAD
        let mut vad = SileroVad::new(vad_model_path.to_path_buf())
            .map_err(|e| anyhow::anyhow!("Failed to initialize VAD: {}", e))?;

        let total_samples = audio.len();
        let mut full_transcription = String::new();
        let mut start_idx = 0;

        while start_idx < total_samples {
            // Determine the end index for this chunk
            let end_idx = if start_idx + TARGET_CHUNK_SIZE >= total_samples {
                total_samples
            } else {
                // Look for a silence point around the target chunk size
                // We'll search in a window of +/- 5 seconds around the 30s mark
                let search_window_samples = 5 * SAMPLE_RATE;
                let target_end = start_idx + TARGET_CHUNK_SIZE;
                let search_start = target_end
                    .saturating_sub(search_window_samples)
                    .max(start_idx);
                let search_end = (target_end + search_window_samples).min(total_samples);

                let mut best_cut_idx = target_end.min(total_samples);
                let mut found_silence = false;

                // Iterate through frames in the search window to find silence
                // Silero VAD expects 30ms frames (480 samples at 16kHz)
                const VAD_FRAME_SIZE: usize = 480; // 30ms * 16000Hz / 1000

                // Align search start to frame boundary relative to start_idx
                let aligned_search_start =
                    start_idx + ((search_start - start_idx) / VAD_FRAME_SIZE) * VAD_FRAME_SIZE;

                for current_pos in (aligned_search_start..search_end).step_by(VAD_FRAME_SIZE) {
                    if current_pos + VAD_FRAME_SIZE > total_samples {
                        break;
                    }

                    let frame = &audio[current_pos..current_pos + VAD_FRAME_SIZE];
                    match vad.push_frame(frame) {
                        Ok(frame_type) => {
                            if !frame_type.is_speech() {
                                // Found silence! Use the end of this frame as cut point
                                best_cut_idx = current_pos + VAD_FRAME_SIZE;
                                found_silence = true;

                                // Optimization: If we are close enough to target, break?
                                // For now, let's just take the first valid silence we find in the window
                                // that is closest to target_end if we were to search more exhaustively.
                                // But the original logic had a break condition that was a bit ambiguous.
                                // Let's stick to: find first silence in window?
                                // The original code had:
                                // if (best_cut_idx - target_end).abs() < (target_end - best_cut_idx).abs() { break; }
                                // which is always false or true depending on signs?
                                // Let's just break on first silence found to be safe and fast.
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("VAD error at sample {}: {}", current_pos, e);
                        }
                    }
                }

                // If no silence found, just cut at target size
                if !found_silence {
                    debug!("No silence found in search window, hard cutting at 30s");
                    target_end.min(total_samples)
                } else {
                    debug!("Found silence at sample {}, cutting there", best_cut_idx);
                    best_cut_idx
                }
            };

            let chunk_len = end_idx - start_idx;
            let chunk_vec = audio[start_idx..end_idx].to_vec();

            debug!(
                "Processing chunk: start={}, len={} samples",
                start_idx, chunk_len
            );

            // Perform transcription for the current chunk via callback
            let chunk_result = callback(chunk_vec)?;

            // Append chunk result to full transcription
            if !full_transcription.is_empty() {
                full_transcription.push(' ');
            }
            full_transcription.push_str(&chunk_result);

            // Move to next chunk
            start_idx = end_idx;

            // Report progress
            let progress = (start_idx as f64 / total_samples as f64) * 100.0;
            progress_callback(progress);
        }

        Ok(full_transcription)
    }
}
