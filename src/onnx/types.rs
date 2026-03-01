use std::fmt;

/// Quantization type for ONNX model loading.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Quantization {
    #[default]
    FP32,
    Int8,
}

/// Language selection for models that support it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Auto,
    Chinese,
    English,
    Japanese,
    Korean,
    Cantonese,
}

impl Language {
    pub fn as_str(&self) -> &str {
        match self {
            Language::Auto => "auto",
            Language::Chinese => "zh",
            Language::English => "en",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Cantonese => "yue",
        }
    }
}

impl Default for Language {
    fn default() -> Self {
        Language::Auto
    }
}

impl std::str::FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Language::Auto),
            "zh" | "chinese" => Ok(Language::Chinese),
            "en" | "english" => Ok(Language::English),
            "ja" | "japanese" => Ok(Language::Japanese),
            "ko" | "korean" => Ok(Language::Korean),
            "yue" | "cantonese" => Ok(Language::Cantonese),
            _ => Err(format!("Unknown language: {}", s)),
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Moonshine model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonshineVariant {
    Tiny,
    TinyAr,
    TinyZh,
    TinyJa,
    TinyKo,
    TinyUk,
    TinyVi,
    Base,
    BaseEs,
}

impl MoonshineVariant {
    pub fn num_layers(&self) -> usize {
        match self {
            MoonshineVariant::Tiny
            | MoonshineVariant::TinyAr
            | MoonshineVariant::TinyZh
            | MoonshineVariant::TinyJa
            | MoonshineVariant::TinyKo
            | MoonshineVariant::TinyUk
            | MoonshineVariant::TinyVi => 6,
            MoonshineVariant::Base | MoonshineVariant::BaseEs => 8,
        }
    }

    pub fn num_key_value_heads(&self) -> usize {
        8
    }

    pub fn head_dim(&self) -> usize {
        match self {
            MoonshineVariant::Tiny
            | MoonshineVariant::TinyAr
            | MoonshineVariant::TinyZh
            | MoonshineVariant::TinyJa
            | MoonshineVariant::TinyKo
            | MoonshineVariant::TinyUk
            | MoonshineVariant::TinyVi => 36,
            MoonshineVariant::Base | MoonshineVariant::BaseEs => 52,
        }
    }

    pub fn token_rate(&self) -> usize {
        match self {
            MoonshineVariant::Tiny | MoonshineVariant::Base | MoonshineVariant::BaseEs => 6,
            MoonshineVariant::TinyUk => 8,
            MoonshineVariant::TinyAr
            | MoonshineVariant::TinyZh
            | MoonshineVariant::TinyJa
            | MoonshineVariant::TinyKo
            | MoonshineVariant::TinyVi => 13,
        }
    }
}

impl Default for MoonshineVariant {
    fn default() -> Self {
        MoonshineVariant::Tiny
    }
}

/// Timestamp granularity for Parakeet output.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum TimestampGranularity {
    #[default]
    Token,
    Word,
    Segment,
}
