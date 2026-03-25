use crate::{TranscriptionResult, TranscriptionSegment};

/// Default separator for merging chunk texts.
pub const DEFAULT_MERGE_SEPARATOR: &str = " ";

/// Merge chunk results into a single result.
///
/// Text is joined with `separator` (default `" "`). Use `""` for
/// languages that don't use space separators (Chinese, Japanese, etc.).
/// Segments are concatenated (timestamps should already be adjusted to
/// session-relative time by the caller).
pub fn merge_sequential(results: &[TranscriptionResult]) -> TranscriptionResult {
    merge_sequential_with_separator(results, DEFAULT_MERGE_SEPARATOR)
}

/// Merge chunk results with a custom separator between chunk texts.
pub fn merge_sequential_with_separator(
    results: &[TranscriptionResult],
    separator: &str,
) -> TranscriptionResult {
    let text = results
        .iter()
        .map(|r| r.text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(separator);

    let segments = {
        let all: Vec<TranscriptionSegment> = results
            .iter()
            .filter_map(|r| r.segments.as_ref())
            .flatten()
            .cloned()
            .collect();
        if all.is_empty() {
            None
        } else {
            Some(all)
        }
    };

    TranscriptionResult { text, segments }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_empty() {
        let result = merge_sequential(&[]);
        assert_eq!(result.text, "");
        assert!(result.segments.is_none());
    }

    #[test]
    fn merge_single() {
        let results = vec![TranscriptionResult {
            text: "hello world".to_string(),
            segments: Some(vec![TranscriptionSegment {
                start: 0.0,
                end: 1.0,
                text: "hello world".to_string(),
            }]),
        }];
        let merged = merge_sequential(&results);
        assert_eq!(merged.text, "hello world");
        assert_eq!(merged.segments.unwrap().len(), 1);
    }

    #[test]
    fn merge_multiple_texts() {
        let results = vec![
            TranscriptionResult {
                text: "hello".to_string(),
                segments: None,
            },
            TranscriptionResult {
                text: "world".to_string(),
                segments: None,
            },
        ];
        let merged = merge_sequential(&results);
        assert_eq!(merged.text, "hello world");
        assert!(merged.segments.is_none());
    }

    #[test]
    fn merge_skips_empty_text() {
        let results = vec![
            TranscriptionResult {
                text: "hello".to_string(),
                segments: None,
            },
            TranscriptionResult {
                text: "  ".to_string(),
                segments: None,
            },
            TranscriptionResult {
                text: "world".to_string(),
                segments: None,
            },
        ];
        let merged = merge_sequential(&results);
        assert_eq!(merged.text, "hello world");
    }

    #[test]
    fn merge_concatenates_segments() {
        let results = vec![
            TranscriptionResult {
                text: "hello".to_string(),
                segments: Some(vec![TranscriptionSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello".to_string(),
                }]),
            },
            TranscriptionResult {
                text: "world".to_string(),
                segments: Some(vec![TranscriptionSegment {
                    start: 5.0,
                    end: 6.0,
                    text: "world".to_string(),
                }]),
            },
        ];
        let merged = merge_sequential(&results);
        let segs = merged.segments.unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].start, 0.0);
        assert_eq!(segs[1].start, 5.0);
    }

    #[test]
    fn merge_trims_whitespace() {
        let results = vec![
            TranscriptionResult {
                text: "  hello  ".to_string(),
                segments: None,
            },
            TranscriptionResult {
                text: "  world  ".to_string(),
                segments: None,
            },
        ];
        let merged = merge_sequential(&results);
        assert_eq!(merged.text, "hello world");
    }
}
