use std::fs;
use std::path::Path;

/// Load a vocabulary file where each line is `token id`.
///
/// Returns a Vec indexed by token ID, and the blank token index.
/// Replaces `▁` (U+2581) with space in token strings.
pub fn load_vocab(path: &Path) -> Result<(Vec<String>, Option<i32>), std::io::Error> {
    let content = fs::read_to_string(path)?;

    let mut max_id = 0;
    let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
    let mut blank_idx: Option<i32> = None;

    for line in content.lines() {
        let parts: Vec<&str> = line.trim_end().split(' ').collect();
        if parts.len() >= 2 {
            let token = parts[0].to_string();
            if let Ok(id) = parts[1].parse::<usize>() {
                if token == "<blk>" {
                    blank_idx = Some(id as i32);
                }
                tokens_with_ids.push((token, id));
                max_id = max_id.max(id);
            }
        }
    }

    let mut vocab = vec![String::new(); max_id + 1];
    for (token, id) in tokens_with_ids {
        vocab[id] = token.replace('\u{2581}', " ");
    }

    log::info!("Loaded {} vocab tokens from {:?}", vocab.len(), path);
    Ok((vocab, blank_idx))
}
