//! #731: `Scratch` gives each caller its own fresh directory and removes it on drop.
//! #785: a sibling owner lock lets the next run sweep the directories of killed runs.

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn owner_of(dir: &Path) -> PathBuf {
    dir.with_file_name(format!("{}.owner", dir.file_name().unwrap().to_str().unwrap()))
}

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
    assert!(owner_of(&pa).exists() && owner_of(&pb).exists() && owner_of(&pc).exists());
    std::fs::write(a.join("f"), "x").unwrap();
    drop((a, b, c));
    assert!(!pa.exists() && !pb.exists() && !pc.exists());
    assert!(!owner_of(&pa).exists() && !owner_of(&pb).exists() && !owner_of(&pc).exists());
}

#[test]
fn a_dead_owners_directory_is_swept_and_a_live_one_is_kept() {
    let base = Scratch::new("sweep-base");
    let make = |name: &str| {
        let d = base.join(name);
        std::fs::create_dir(&d).unwrap();
        std::fs::write(d.join("f"), "x").unwrap();
        d
    };
    let dead = make("rcsr-test-x-1-0-dead");
    std::fs::write(owner_of(&dead), "").unwrap();
    let live = make("rcsr-test-x-1-1-live");
    let held = std::fs::File::create(owner_of(&live)).unwrap();
    held.lock().unwrap();
    let plain = make("rcsr-test-x-1-2-plain");
    assert_eq!(scratch::sweep_dead_owners_in(base.path()), 1);
    assert!(!dead.exists() && !owner_of(&dead).exists(), "an unlocked owner file marks a dead directory");
    assert!(live.is_dir() && owner_of(&live).exists(), "a locked owner file keeps its directory");
    assert!(plain.is_dir(), "a directory without an owner file is never swept");
}

#[test]
#[ignore = "child process of a_killed_owners_directory_is_swept_by_the_next_sweep"]
fn hold_a_scratch_until_killed() {
    let s = Scratch::new("held");
    println!("SCRATCH {}", s.path().display());
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}

#[test]
fn a_killed_owners_directory_is_swept_by_the_next_sweep() {
    let base = Scratch::new("kill-base");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["hold_a_scratch_until_killed", "--exact", "--ignored", "--nocapture"])
        .env("TMPDIR", base.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let path = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| l.unwrap())
        .find_map(|l| l.split_once("SCRATCH ").map(|(_, p)| PathBuf::from(p)))
        .expect("the child prints its scratch path");
    assert!(path.starts_with(base.path()));
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(path.is_dir() && owner_of(&path).exists(), "a killed owner leaves its directory");
    assert_eq!(scratch::sweep_dead_owners_in(base.path()), 1);
    assert!(!path.exists() && !owner_of(&path).exists());
}
