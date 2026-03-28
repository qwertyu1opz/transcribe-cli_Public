use std::path::Path;
const VIDEO_EXTENSIONS: &[&str] = &[
    "3gp", "avi", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts", "ts", "webm",
    "wmv",
];

pub fn is_video_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .is_some_and(|extension| VIDEO_EXTENSIONS.iter().any(|known| known == &extension))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::is_video_file;

    #[test]
    fn detects_known_video_extensions_case_insensitively() {
        assert!(is_video_file(Path::new("clip.MP4")));
        assert!(is_video_file(Path::new("clip.webm")));
        assert!(!is_video_file(Path::new("clip.mp3")));
    }
}
