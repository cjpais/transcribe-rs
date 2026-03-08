use crate::TranscriptionResult;

/// Apply inverse text normalization (ITN) to a transcription result.
///
/// Converts spoken-form text to written-form, e.g. "twenty three dollars" → "$23".
/// Modifies `result.text` and each segment's text in place.
pub fn apply_itn(result: &mut TranscriptionResult) {
    result.text = text_processing_rs::normalize_sentence(&result.text);
    if let Some(segments) = &mut result.segments {
        for segment in segments.iter_mut() {
            segment.text = text_processing_rs::normalize_sentence(&segment.text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TranscriptionSegment;

    #[test]
    fn test_itn_basic() {
        let mut result = TranscriptionResult {
            text: "twenty three dollars".to_string(),
            segments: None,
        };
        apply_itn(&mut result);
        assert_eq!(result.text, "$23");
    }

    #[test]
    fn test_itn_with_segments() {
        let mut result = TranscriptionResult {
            text: "twenty three dollars and fifty cents".to_string(),
            segments: Some(vec![
                TranscriptionSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "twenty three dollars".to_string(),
                },
                TranscriptionSegment {
                    start: 1.0,
                    end: 2.0,
                    text: "fifty cents".to_string(),
                },
            ]),
        };
        apply_itn(&mut result);
        assert_eq!(result.text, "$23.50");
        let segments = result.segments.as_ref().unwrap();
        assert_eq!(segments[0].text, "$23");
        assert_eq!(segments[1].text, "$0.50");
    }

    #[test]
    fn test_itn_no_change() {
        let mut result = TranscriptionResult {
            text: "hello world".to_string(),
            segments: None,
        };
        apply_itn(&mut result);
        assert_eq!(result.text, "hello world");
    }
}
