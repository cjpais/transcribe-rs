//! BBPE (Byte-level BPE) symbol table for Icefall/sherpa-onnx Zipformer models.
//!
//! Handles byte-to-unicode mapping and auto-detects encoding mode based on
//! whether a `bbpe.model` file is present alongside the tokens file.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Whether tokens use BBPE byte encoding or standard BPE (literal UTF-8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenEncoding {
    Bbpe,
    Bpe,
}

/// Symbol table for Icefall/sherpa-onnx Zipformer models with BBPE support.
pub struct BbpeSymbolTable {
    id_to_sym: HashMap<i32, String>,
    encoding: TokenEncoding,
}

impl BbpeSymbolTable {
    /// Load a symbol table, auto-detecting encoding mode.
    ///
    /// If a `bbpe.model` file exists in the same directory as `path`,
    /// BBPE encoding is used; otherwise standard BPE is assumed.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        Self::load_autodetect(path)
    }

    /// Load with explicit auto-detection of encoding based on sibling files.
    pub fn load_autodetect(path: &Path) -> Result<Self, std::io::Error> {
        let encoding = if let Some(dir) = path.parent() {
            if dir.join("bbpe.model").exists() {
                log::debug!("Detected BBPE encoding (bbpe.model found)");
                TokenEncoding::Bbpe
            } else {
                log::debug!("Detected standard BPE encoding (no bbpe.model)");
                TokenEncoding::Bpe
            }
        } else {
            TokenEncoding::Bbpe
        };
        Self::load_with_encoding(path, encoding)
    }

    /// Load with an explicitly specified encoding.
    pub fn load_with_encoding(
        path: &Path,
        encoding: TokenEncoding,
    ) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();
        for line in contents.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() != 2 {
                continue;
            }
            if let Ok(id) = parts[0].parse::<i32>() {
                id_to_sym.insert(id, parts[1].to_string());
            }
        }
        Ok(Self {
            id_to_sym,
            encoding,
        })
    }

    /// Look up a symbol by token ID.
    pub fn get(&self, id: i32) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    /// Decode a sequence of token IDs to a UTF-8 string.
    pub fn decode(&self, token_ids: &[i32]) -> String {
        match self.encoding {
            TokenEncoding::Bbpe => self.decode_bbpe(token_ids),
            TokenEncoding::Bpe => self.decode_bpe(token_ids),
        }
    }

    fn decode_bbpe(&self, token_ids: &[i32]) -> String {
        let mut raw_bytes = Vec::new();
        for &id in token_ids {
            let Some(sym) = self.get(id) else { continue };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            for c in sym.chars() {
                if c == '\u{2581}' {
                    raw_bytes.push(b' ');
                } else if let Some(byte_val) = bbpe_char_to_byte(c) {
                    raw_bytes.push(byte_val);
                }
            }
        }
        let text = String::from_utf8_lossy(&raw_bytes);
        normalize_text(text.trim())
    }

    fn decode_bpe(&self, token_ids: &[i32]) -> String {
        let mut text = String::new();
        for &id in token_ids {
            let Some(sym) = self.get(id) else { continue };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            text.push_str(&sym.replace('\u{2581}', " "));
        }
        normalize_text(text.trim())
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{F900}'..='\u{FAFF}' |
        '\u{2E80}'..='\u{2EFF}' |
        '\u{3000}'..='\u{303F}' |
        '\u{FF00}'..='\u{FFEF}'
    )
}

fn normalize_text(text: &str) -> String {
    let text = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    for i in 0..chars.len() {
        let c = chars[i];
        if c == ' ' {
            let prev_cjk = i > 0 && is_cjk(chars[i - 1]);
            let next_cjk = i + 1 < chars.len() && is_cjk(chars[i + 1]);
            if prev_cjk && next_cjk {
                continue;
            }
        }
        result.push(c);
    }
    result
}

/// BBPE codepoint table: maps byte value (index) to Unicode codepoint.
const BBPE_CODEPOINTS: [u32; 256] = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
    275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 308,
    309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326, 327, 328, 330,
    331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
    350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
    369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
];

/// Convert a BBPE-encoded Unicode character back to its original byte value.
pub fn bbpe_char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if (32..=126).contains(&cp) {
        return Some(cp as u8);
    }
    for (byte_val, &mapped_cp) in BBPE_CODEPOINTS.iter().enumerate() {
        if mapped_cp == cp {
            return Some(byte_val as u8);
        }
    }
    None
}
