use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct SymbolTable {
    id_to_sym: HashMap<i32, String>,
}

impl SymbolTable {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in contents.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }

            // Format: "token id" and keeps support for whitespace-like tokens.
            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() != 2 {
                continue;
            }

            if let Ok(id) = parts[0].parse::<i32>() {
                id_to_sym.insert(id, parts[1].to_string());
            }
        }

        Ok(Self { id_to_sym })
    }

    pub fn get(&self, id: i32) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    pub fn decode(&self, token_ids: &[i32]) -> String {
        let mut text = String::new();
        let mut prev_join_to_next = false;

        for &id in token_ids {
            let Some(sym) = self.get(id) else {
                continue;
            };

            if is_special_symbol(sym) {
                continue;
            }

            let joins_next = sym.ends_with("@@");
            let clean = sym.trim_end_matches("@@");

            if clean.is_empty() {
                prev_join_to_next = joins_next;
                continue;
            }

            if clean.starts_with('▁') {
                let piece = clean.trim_start_matches('▁');
                if !piece.is_empty() {
                    if !text.is_empty() && !text.ends_with(' ') {
                        text.push(' ');
                    }
                    text.push_str(piece);
                }
                prev_join_to_next = joins_next;
                continue;
            }

            if !text.is_empty() && !prev_join_to_next {
                let prev_char = text.chars().last();
                let curr_is_ascii_word = is_ascii_word_piece(clean);
                let prev_is_ascii_word = prev_char.map(is_ascii_word_char).unwrap_or(false);
                let prev_is_cjk = prev_char.map(is_cjk).unwrap_or(false);

                // For mixed Chinese-English output, restore missing word boundaries.
                if curr_is_ascii_word && (prev_is_ascii_word || prev_is_cjk) && !text.ends_with(' ')
                {
                    text.push(' ');
                }
            }

            text.push_str(clean);
            prev_join_to_next = joins_next;
        }

        text.trim().to_string()
    }
}

fn is_special_symbol(sym: &str) -> bool {
    sym == "<blank>"
        || sym == "<s>"
        || sym == "</s>"
        || sym == "<unk>"
        || sym == "<OOV>"
        || (sym.starts_with('<') && sym.ends_with('>'))
}

fn is_ascii_word_piece(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_ascii_word_char)
}

fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn is_cjk(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
}
