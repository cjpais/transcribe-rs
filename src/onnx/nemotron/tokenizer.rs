//! Minimal SentencePiece vocabulary loader for Nemotron.
//!
//! Ported from parakeet-rs (MIT). Nemotron ships a `tokenizer.model`
//! (SentencePiece protobuf), whereas transcribe-rs's existing helpers
//! (`decode::tokens::load_vocab`, `decode::sentencepiece`) only handle the
//! `token id` text format and piece joining. Rather than require a re-exported
//! `vocab.txt`, this parses the protobuf directly so the published ONNX bundle
//! can be consumed unchanged. The parser walks just enough of the wire format
//! to pull each piece string (field 1 of each `SentencePiece` sub-message);
//! everything else is skipped.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::TranscribeError;

/// SentencePiece "▁" word-boundary marker (U+2581).
const SPACE_MARKER: char = '\u{2581}';

/// Detect SentencePiece pieces that encode a language tag like `<en-US>` or
/// `<en>`. The multilingual model emits these inline with text; they are
/// stripped from the user-visible transcript.
pub(super) fn is_lang_tag(piece: &str) -> bool {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return false;
    }
    let inner = &bytes[1..bytes.len() - 1];
    match inner.len() {
        2 => inner[0].is_ascii_lowercase() && inner[1].is_ascii_lowercase(),
        5 => {
            inner[0].is_ascii_lowercase()
                && inner[1].is_ascii_lowercase()
                && inner[2] == b'-'
                && inner[3].is_ascii_uppercase()
                && inner[4].is_ascii_uppercase()
        }
        _ => false,
    }
}

/// Vocabulary parsed from a SentencePiece `tokenizer.model` protobuf.
pub(super) struct SentencePieceVocab {
    pieces: Vec<String>,
}

impl SentencePieceVocab {
    pub(super) fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TranscribeError> {
        let mut file = File::open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let pieces = Self::parse_model(&data)?;
        Ok(Self { pieces })
    }

    /// Walk the top-level `ModelProto`, collecting field 1 (`pieces`,
    /// length-delimited `SentencePiece` messages).
    fn parse_model(data: &[u8]) -> Result<Vec<String>, TranscribeError> {
        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (header, read) = read_varint(&data[pos..])?;
            pos += read;
            let field_num = header >> 3;
            let wire_type = header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, read) = read_varint(&data[pos..])?;
                    pos += read;
                    let len = len as usize;
                    if pos + len > data.len() {
                        break;
                    }
                    let piece_msg = &data[pos..pos + len];
                    pos += len;
                    if let Ok(piece) = parse_piece(piece_msg) {
                        pieces.push(piece);
                    }
                }
                (_, 0) => {
                    let (_, read) = read_varint(&data[pos..])?;
                    pos += read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, read) = read_varint(&data[pos..])?;
                    pos += read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        if pieces.is_empty() {
            return Err(TranscribeError::Config(
                "no tokens found in tokenizer.model".into(),
            ));
        }
        Ok(pieces)
    }

    /// Decode token ids to text, converting the "▁" marker back to spaces.
    pub(super) fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(piece) = self.pieces.get(id) {
                out.push_str(&piece.replace(SPACE_MARKER, " "));
            }
        }
        out.trim_start().to_string()
    }

    /// Decode a single token id (no trimming). Reserved for future per-token
    /// / timestamped output; the offline path uses [`Self::decode`].
    #[allow(dead_code)]
    pub(super) fn decode_single(&self, id: usize) -> String {
        self.pieces
            .get(id)
            .map(|p| p.replace(SPACE_MARKER, " "))
            .unwrap_or_default()
    }

    pub(super) fn size(&self) -> usize {
        self.pieces.len()
    }

    /// Token ids whose pieces look like language tags (`<en-US>`, `<fr>`, ...).
    /// Empty for the English-only vocabulary.
    pub(super) fn lang_tag_ids(&self) -> Vec<usize> {
        self.pieces
            .iter()
            .enumerate()
            .filter_map(|(i, p)| is_lang_tag(p).then_some(i))
            .collect()
    }
}

/// Parse a `SentencePiece` sub-message, returning its `piece` string (field 1).
fn parse_piece(data: &[u8]) -> Result<String, TranscribeError> {
    let mut pos = 0;
    let mut piece = String::new();

    while pos < data.len() {
        let (header, read) = read_varint(&data[pos..])?;
        pos += read;
        let field_num = header >> 3;
        let wire_type = header & 0x7;

        match (field_num, wire_type) {
            (1, 2) => {
                let (len, read) = read_varint(&data[pos..])?;
                pos += read;
                let len = len as usize;
                if pos + len <= data.len() {
                    piece = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
                }
                pos += len;
            }
            (_, 0) => {
                let (_, read) = read_varint(&data[pos..])?;
                pos += read;
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let (len, read) = read_varint(&data[pos..])?;
                pos += read + len as usize;
            }
            (_, 5) => pos += 4,
            _ => break,
        }
    }

    Ok(piece)
}

/// Read a base-128 varint, returning `(value, bytes_consumed)`.
fn read_varint(data: &[u8]) -> Result<(u64, usize), TranscribeError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    let mut pos = 0;

    while pos < data.len() && pos < 10 {
        let byte = data[pos];
        result |= ((byte & 0x7F) as u64) << shift;
        pos += 1;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
    }
    Err(TranscribeError::Config("invalid varint in tokenizer.model".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_tag_detection() {
        assert!(is_lang_tag("<en-US>"));
        assert!(is_lang_tag("<fr>"));
        assert!(is_lang_tag("<ja-JP>"));
        assert!(!is_lang_tag("hello"));
        assert!(!is_lang_tag("<EN>")); // first two must be lowercase
        assert!(!is_lang_tag("<abc>")); // 3 inner chars is not a tag
        assert!(!is_lang_tag("<en_US>")); // separator must be '-'
        assert!(!is_lang_tag("<>"));
    }

    /// Encode a length-delimited (wire type 2) field.
    fn ld_field(field_num: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![((field_num << 3) | 2) as u8];
        // payloads in these tests are < 128 bytes, so varint == single byte
        assert!(payload.len() < 128);
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    }

    /// Build a minimal SentencePiece ModelProto from piece strings.
    fn synth_model(pieces: &[&str]) -> Vec<u8> {
        let mut model = Vec::new();
        for p in pieces {
            let piece_msg = ld_field(1, p.as_bytes()); // SentencePiece.piece = field 1
            model.extend(ld_field(1, &piece_msg)); // ModelProto.pieces = field 1
        }
        model
    }

    #[test]
    fn parses_pieces_and_decodes_spaces() {
        // "▁he", "llo", "▁world"
        let model = synth_model(&["\u{2581}he", "llo", "\u{2581}world"]);
        let vocab = SentencePieceVocab {
            pieces: SentencePieceVocab::parse_model(&model).unwrap(),
        };
        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.decode(&[0, 1, 2]), "hello world");
        assert_eq!(vocab.decode_single(2), " world");
        // out-of-range ids are skipped, not panics
        assert_eq!(vocab.decode(&[2, 99]), "world");
    }

    #[test]
    fn collects_lang_tag_ids() {
        let model = synth_model(&["<en-US>", "\u{2581}hi", "<fr>", "there"]);
        let vocab = SentencePieceVocab {
            pieces: SentencePieceVocab::parse_model(&model).unwrap(),
        };
        assert_eq!(vocab.lang_tag_ids(), vec![0, 2]);
    }

    #[test]
    fn empty_model_errors() {
        assert!(SentencePieceVocab::parse_model(&[]).is_err());
    }
}
