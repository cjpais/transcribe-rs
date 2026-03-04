//! Prompt template construction for Qwen3-ASR.
//!
//! Builds the token ID sequence:
//!   `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`

use super::config::SpecialTokens;

/// Well-known text token IDs from the Qwen3-ASR tokenizer vocabulary.
/// These are fixed across all Qwen3/Qwen2.5 tokenizer versions — they map to
/// the English words "system", "user", "assistant" and the newline character in
/// the base GPT-2 BPE vocabulary that Qwen inherits.
const SYSTEM_TOKEN_ID: i64 = 9125; // "system"
const USER_TOKEN_ID: i64 = 882; // "user"
const ASSISTANT_TOKEN_ID: i64 = 77091; // "assistant"
const NEWLINE_TOKEN_ID: i64 = 198; // "\n"

/// Compute the number of encoder output tokens from a mel frame count.
///
/// Matches the native Qwen3-ASR formula: windowed conv with 100-frame windows,
/// three stride-2 conv layers on the remainder, 13 tokens per full window.
pub fn get_feat_extract_output_lengths(input_lengths: usize) -> usize {
    let leave = input_lengths % 100;
    let mut t = leave.div_ceil(2);
    t = t.div_ceil(2);
    t = t.div_ceil(2);
    t + (input_lengths / 100) * 13
}

/// Build the complete prompt token ID sequence for ASR transcription.
///
/// `audio_token_count` is the number of encoder output tokens (use
/// `get_feat_extract_output_lengths(mel_frames)` to compute from mel frame count).
pub fn build_prompt_ids(special_tokens: &SpecialTokens, audio_token_count: usize) -> Vec<i64> {
    let mut ids = Vec::with_capacity(audio_token_count + 15);

    // System turn (empty content)
    ids.push(special_tokens.im_start_token_id);
    ids.push(SYSTEM_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);
    ids.push(special_tokens.im_end_token_id);
    ids.push(NEWLINE_TOKEN_ID);

    // User turn with audio
    ids.push(special_tokens.im_start_token_id);
    ids.push(USER_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);
    ids.push(special_tokens.audio_start_token_id);

    // Audio pad tokens — replaced by encoder output embeddings at runtime
    ids.extend(std::iter::repeat_n(
        special_tokens.audio_pad_token_id,
        audio_token_count,
    ));

    ids.push(special_tokens.audio_end_token_id);
    ids.push(special_tokens.im_end_token_id);
    ids.push(NEWLINE_TOKEN_ID);

    // Assistant turn prefix
    ids.push(special_tokens.im_start_token_id);
    ids.push(ASSISTANT_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);

    ids
}

/// Find the `[start, end)` range of `audio_pad` token positions in the prompt.
pub fn get_audio_pad_range(
    prompt_ids: &[i64],
    audio_pad_token_id: i64,
) -> Result<(usize, usize), String> {
    let start = prompt_ids
        .iter()
        .position(|&id| id == audio_pad_token_id)
        .ok_or_else(|| "No audio_pad tokens in prompt".to_string())?;
    let end = prompt_ids
        .iter()
        .rposition(|&id| id == audio_pad_token_id)
        .ok_or_else(|| "No audio_pad tokens in prompt".to_string())?
        + 1;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_special_tokens() -> SpecialTokens {
        SpecialTokens {
            eos_token_ids: vec![151643, 151645],
            pad_token_id: 151643,
            im_start_token_id: 151644,
            im_end_token_id: 151645,
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
            audio_pad_token_id: 151676,
        }
    }

    #[test]
    fn test_encoder_output_lengths() {
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(1), 1);
        assert_eq!(get_feat_extract_output_lengths(100), 13);
        assert_eq!(get_feat_extract_output_lengths(200), 26);
        assert_eq!(get_feat_extract_output_lengths(997), 130);
    }

    #[test]
    fn test_build_prompt_ids_structure() {
        let st = test_special_tokens();
        let ids = build_prompt_ids(&st, 10);

        // Should start with im_start, system, newline, im_end, newline
        assert_eq!(ids[0], st.im_start_token_id);
        assert_eq!(ids[1], SYSTEM_TOKEN_ID);
        assert_eq!(ids[2], NEWLINE_TOKEN_ID);
        assert_eq!(ids[3], st.im_end_token_id);
        assert_eq!(ids[4], NEWLINE_TOKEN_ID);

        // User turn
        assert_eq!(ids[5], st.im_start_token_id);
        assert_eq!(ids[6], USER_TOKEN_ID);
        assert_eq!(ids[7], NEWLINE_TOKEN_ID);
        assert_eq!(ids[8], st.audio_start_token_id);

        // 10 audio_pad tokens at positions 9..19
        for i in 9..19 {
            assert_eq!(ids[i], st.audio_pad_token_id);
        }

        // audio_end, im_end, newline
        assert_eq!(ids[19], st.audio_end_token_id);
        assert_eq!(ids[20], st.im_end_token_id);
        assert_eq!(ids[21], NEWLINE_TOKEN_ID);

        // Assistant turn
        assert_eq!(ids[22], st.im_start_token_id);
        assert_eq!(ids[23], ASSISTANT_TOKEN_ID);
        assert_eq!(ids[24], NEWLINE_TOKEN_ID);

        assert_eq!(ids.len(), 25); // 15 fixed + 10 audio_pad
    }

    #[test]
    fn test_audio_pad_range() {
        let st = test_special_tokens();
        let ids = build_prompt_ids(&st, 10);
        let (start, end) = get_audio_pad_range(&ids, st.audio_pad_token_id).unwrap();
        assert_eq!(start, 9);
        assert_eq!(end, 19);
        assert_eq!(end - start, 10);
    }
}
