//! Prompt template construction for Qwen3-ASR.
//!
//! Builds the token ID sequence for the chat-style prompt that Qwen3-ASR expects.
//! Standard prompt (no language hint):
//!   `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
//!
//! With language hint (official Qwen3-ASR template):
//!   `...<|im_start|>assistant\nlanguage {Name}<asr_text>`
//! This forces the model to skip language detection and go straight to transcription.

use super::config::SpecialTokens;

/// Well-known text token IDs from the Qwen3-ASR tokenizer vocabulary.
/// These are fixed across all Qwen3/Qwen2.5 tokenizer versions — they map to
/// the English words "system", "user", "assistant" and the newline character in
/// the base GPT-2 BPE vocabulary that Qwen inherits.
const SYSTEM_TOKEN_ID: i64 = 9125; // "system"
const USER_TOKEN_ID: i64 = 882; // "user"
const ASSISTANT_TOKEN_ID: i64 = 77091; // "assistant"
const NEWLINE_TOKEN_ID: i64 = 198; // "\n"
/// Token for "language" — the fixed prefix of the assistant's language tag.
const LANGUAGE_TOKEN_ID: i64 = 11528; // "language"

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
/// This builds the standard prompt with an empty system turn and no language
/// hint. Equivalent to `build_prompt_ids_with_language(st, n, None)`.
#[cfg(test)]
pub(crate) fn build_prompt_ids(
    special_tokens: &SpecialTokens,
    audio_token_count: usize,
) -> Vec<i64> {
    build_prompt_ids_with_language(special_tokens, audio_token_count, None)
}

/// Build a prompt with optional language hint using the official Qwen3-ASR template.
///
/// When `language_token_ids` is provided, the assistant prefix is extended with
/// `language {Name}<asr_text>`, forcing the model to skip language detection and
/// begin transcription immediately. This matches the official Qwen3-ASR inference
/// template (see `qwen_asr/core/vllm_backend/qwen3_asr.py`).
///
/// The forced `<asr_text>` token also prevents degenerate output (e.g. "ology")
/// from int4 quantization noise on non-speech audio, since the model never gets
/// the chance to produce a wrong first token.
///
/// `language_token_ids` should be the tokenized form of ` {Name}` (with leading
/// space) — e.g. `encode(" English")` → `[6364]`. The "language" prefix token
/// is added automatically.
///
/// When `language_token_ids` is `None`, builds the standard prompt with no
/// language hint (model auto-detects language).
pub fn build_prompt_ids_with_language(
    special_tokens: &SpecialTokens,
    audio_token_count: usize,
    language_token_ids: Option<&[i64]>,
) -> Vec<i64> {
    let mut ids = Vec::with_capacity(audio_token_count + 20);

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

    // Language hint: force "language {Name}<asr_text>" as assistant prefix.
    // This pre-fills the assistant's response so the model skips language
    // detection and begins transcription after <asr_text>.
    if let Some(lang_ids) = language_token_ids {
        ids.push(LANGUAGE_TOKEN_ID); // "language"
        ids.extend_from_slice(lang_ids); // " English" (with leading space)
        ids.push(special_tokens.asr_text_token_id); // <asr_text>
    }

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
            asr_text_token_id: 151704,
        }
    }

    #[test]
    fn test_encoder_output_lengths() {
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(1), 1);
        assert_eq!(get_feat_extract_output_lengths(99), 13);
        assert_eq!(get_feat_extract_output_lengths(100), 13);
        assert_eq!(get_feat_extract_output_lengths(200), 26);
        assert_eq!(get_feat_extract_output_lengths(997), 130);
    }

    #[test]
    fn test_build_prompt_ids_structure() {
        let st = test_special_tokens();
        let ids = build_prompt_ids(&st, 10);

        assert_eq!(ids[0], st.im_start_token_id);
        assert_eq!(ids[1], SYSTEM_TOKEN_ID);
        assert_eq!(ids[2], NEWLINE_TOKEN_ID);
        assert_eq!(ids[3], st.im_end_token_id);
        assert_eq!(ids[4], NEWLINE_TOKEN_ID);

        assert_eq!(ids[5], st.im_start_token_id);
        assert_eq!(ids[6], USER_TOKEN_ID);
        assert_eq!(ids[7], NEWLINE_TOKEN_ID);
        assert_eq!(ids[8], st.audio_start_token_id);

        for i in 9..19 {
            assert_eq!(ids[i], st.audio_pad_token_id);
        }

        assert_eq!(ids[19], st.audio_end_token_id);
        assert_eq!(ids[20], st.im_end_token_id);
        assert_eq!(ids[21], NEWLINE_TOKEN_ID);

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

    #[test]
    fn test_build_prompt_ids_with_language() {
        let st = test_special_tokens();
        // " English" (with leading space) as the reference tokenizer produces
        let lang_ids: &[i64] = &[6364];
        let ids = build_prompt_ids_with_language(&st, 5, Some(lang_ids));

        // Standard prompt structure up to assistant turn
        assert_eq!(ids[0], st.im_start_token_id); // system turn
        assert_eq!(ids[3], st.im_end_token_id);
        assert_eq!(ids[5], st.im_start_token_id); // user turn
        assert_eq!(ids[8], st.audio_start_token_id);
        for i in 9..14 {
            assert_eq!(ids[i], st.audio_pad_token_id); // 5 audio tokens
        }
        assert_eq!(ids[14], st.audio_end_token_id);
        assert_eq!(ids[15], st.im_end_token_id);

        // Assistant turn with forced language prefix
        assert_eq!(ids[17], st.im_start_token_id);
        assert_eq!(ids[18], ASSISTANT_TOKEN_ID);
        assert_eq!(ids[19], NEWLINE_TOKEN_ID);
        assert_eq!(ids[20], LANGUAGE_TOKEN_ID); // "language"
        assert_eq!(ids[21], 6364); // " English"
        assert_eq!(ids[22], st.asr_text_token_id); // <asr_text>

        assert_eq!(ids.len(), 23); // 20 standard + 3 language hint
    }

    #[test]
    fn test_language_none_matches_standard_prompt() {
        let st = test_special_tokens();
        let standard = build_prompt_ids(&st, 10);
        let with_none = build_prompt_ids_with_language(&st, 10, None);
        assert_eq!(standard, with_none);
    }
}
