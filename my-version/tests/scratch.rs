//! #731: `Scratch` gives each caller its own fresh directory and removes it on drop.

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;

#[test]
fn scratch_directories_are_unique_fresh_and_removed_on_drop() {
    let a = Scratch::new("x");
    let b = Scratch::new("x");
    assert_ne!(a.path(), b.path());
    assert!(a.path().is_dir() && b.path().is_dir());

    let name = b.path().file_name().unwrap().to_str().unwrap().to_string();
    let stem = name.strip_suffix("-x").unwrap();
    let (head, n) = stem.rsplit_once('-').unwrap();
    let next = b
        .path()
        .with_file_name(format!("{head}-{}-x", n.parse::<usize>().unwrap() + 1));
    std::fs::create_dir(&next).unwrap();
    std::fs::write(next.join("stale"), "old").unwrap();
    let c = Scratch::new("x");
    assert_eq!(c.path(), next);
    assert_eq!(std::fs::read_dir(c.path()).unwrap().count(), 0);

    let (pa, pb, pc) = (a.path().to_path_buf(), b.path().to_path_buf(), c.path().to_path_buf());
    std::fs::write(a.join("f"), "x").unwrap();
    drop((a, b, c));
    assert!(!pa.exists() && !pb.exists() && !pc.exists());
}
