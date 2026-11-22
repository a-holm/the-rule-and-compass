//! #981, #1319: the build script reruns only when this checkout's commit can change,
//! never for another worktree's ref changes, and always records the commit of
//! the checkout being built. Each case builds a probe crate that carries the
//! real `build.rs`, inside a temporary repository with a linked worktree.

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Command;

fn clean_git_env(cmd: &mut Command) -> &mut Command {
    for (k, _) in std::env::vars() {
        if k.starts_with("GIT_") {
            cmd.env_remove(k);
        }
    }
    cmd.env("GIT_CONFIG_GLOBAL", "/dev/null").env("GIT_CONFIG_NOSYSTEM", "1")
}

fn git(dir: &Path, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    cmd.args(["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main"]).args(args).current_dir(dir);
    let out = clean_git_env(&mut cmd).output()
        .expect("run git");
    assert!(out.status.success(), "git {:?}: {}", args, String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

struct Build {
    fresh: bool,
    compiled: bool,
    commit: String,
    dirty: bool,
    source_sha: String,
    stdout: String,
    log: String,
}

struct Fixture {
    _tmp: Scratch,
    main: PathBuf,
    wt: PathBuf,
    target: PathBuf,
}

impl Fixture {
    fn new(tag: &str, reflog: bool) -> Fixture {
        let tmp = Scratch::new(tag);
        let main = tmp.join("main");
        let probe = main.join("probe");
        std::fs::create_dir_all(probe.join("src/cuda")).unwrap();
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        std::fs::copy(here.join("build.rs"), probe.join("build.rs")).unwrap();
        std::fs::copy(here.join("src/cuda/arch.rs"), probe.join("src/cuda/arch.rs")).unwrap();
        std::fs::copy(here.join("src/sha256_core.rs"), probe.join("src/sha256_core.rs")).unwrap();
        std::fs::write(probe.join("src/cuda/eval.cu"), "// kernel\n").unwrap();
        std::fs::write(
            probe.join("Cargo.toml"),
            "[package]\nname = \"probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\nbuild = \"build.rs\"\n",
        )
        .unwrap();
        std::fs::write(
            probe.join("src/main.rs"),
            "fn main() { println!(\"{} {} {}\", env!(\"RCSR_BUILD_COMMIT\"), env!(\"RCSR_BUILD_DIRTY\"), option_env!(\"RCSR_BUILD_SOURCE_SHA256\").unwrap_or(\"-\")); }\n",
        )
        .unwrap();
        std::fs::write(probe.join(".gitignore"), "/Cargo.lock\n").unwrap();
        git(&main, &["init", "-q"]);
        if !reflog {
            git(&main, &["config", "core.logAllRefUpdates", "false"]);
        }
        git(&main, &["add", "."]);
        git(&main, &["commit", "-q", "-m", "first"]);
        git(&main, &["pack-refs", "--all"]);
        let wt = tmp.join("wt");
        git(&main, &["worktree", "add", "-q", "-b", "w", wt.to_str().unwrap()]);
        let target = tmp.join("target");
        Fixture { _tmp: tmp, main, wt, target }
    }

    fn probe(&self) -> PathBuf {
        self.wt.join("probe")
    }

    fn build(&self) -> Build {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        let mut cmd = Command::new(cargo);
        let out = clean_git_env(&mut cmd)
            .args(["build", "-vv", "--offline"])
            .current_dir(self.probe())
            .env("CARGO_TARGET_DIR", &self.target)
            .env_remove("RCSR_BUILD_COMMIT")
            .env_remove("RCSR_BUILD_DIRTY")
            .env_remove("RCSR_BUILD_SOURCE_SHA256")
            .env_remove("RUSTC_WRAPPER")
            .output()
            .expect("run cargo");
        let log = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{}", log);
        let ran = Command::new(self.target.join("debug/probe")).output().expect("run probe");
        let stdout = String::from_utf8_lossy(&ran.stdout).into_owned();
        let mut fields = stdout.split_whitespace();
        let mut next = || fields.next().unwrap_or_default().to_string();
        let (commit, dirty, source_sha) = (next(), next(), next());
        Build {
            fresh: log.contains("Fresh probe"),
            compiled: log.contains("Compiling probe"),
            commit,
            dirty: dirty == "true",
            source_sha,
            stdout,
            log,
        }
    }

    fn common_dir(&self) -> PathBuf {
        self.main.join(".git")
    }

    fn assert_fresh(&self, what: &str) {
        let b = self.build();
        assert!(b.fresh && !b.compiled, "{what}: the probe was rebuilt\n{}", b.log);
    }

    fn assert_rebuilt_at_head(&self, what: &str) {
        let b = self.build();
        assert!(b.compiled, "{what}: the probe was not rebuilt\n{}", b.log);
        assert_eq!(b.commit, git(&self.wt, &["rev-parse", "HEAD"]), "{what}: stale commit");
    }

    fn commit_in_wt(&self, text: &str) {
        std::fs::write(self.probe().join("note.txt"), text).unwrap();
        git(&self.wt, &["add", "."]);
        git(&self.wt, &["commit", "-q", "-m", text]);
    }
}

fn scenario(reflog: bool) {
    let f = Fixture::new(if reflog { "reflog" } else { "noreflog" }, reflog);
    let first = f.build();
    assert!(first.compiled);
    assert_eq!(first.commit, git(&f.wt, &["rev-parse", "HEAD"]));
    f.assert_fresh("second build");

    let packed = f.common_dir().join("packed-refs");
    if reflog {
        git(&f.main, &["pack-refs", "--all"]);
        f.assert_fresh("pack-refs --all from the other worktree");
        git(&f.main, &["branch", "other"]);
        git(&f.main, &["pack-refs", "--all"]);
        git(&f.main, &["branch", "-D", "other"]);
        f.assert_fresh("branch -D in the other worktree");
    } else {
        std::fs::write(&packed, "# pack-refs with: peeled fully-peeled sorted\n").unwrap();
    }
    std::fs::OpenOptions::new().append(true).open(&packed).unwrap().set_modified(std::time::SystemTime::now()).unwrap();
    f.assert_fresh("touch packed-refs");

    f.commit_in_wt("second");
    f.assert_rebuilt_at_head("commit in the worktree");
    f.assert_fresh("build after the commit");

    if reflog {
        git(&f.main, &["pack-refs", "--all"]);
        f.assert_fresh("pack-refs after the commit");
        f.commit_in_wt("third");
        f.assert_rebuilt_at_head("commit on a packed branch");
    } else {
        f.commit_in_wt("third");
        f.assert_rebuilt_at_head("second commit without a reflog");
    }

    let first_commit = git(&f.wt, &["rev-parse", "HEAD~2"]);
    git(&f.wt, &["checkout", "-q", "--detach", &first_commit]);
    f.assert_rebuilt_at_head("detached checkout");
    f.assert_fresh("build on the detached head");
    f.commit_in_wt("detached");
    f.assert_rebuilt_at_head("commit on a detached head");

    std::fs::write(f.probe().join("src/cuda/eval.cu"), "// changed\n").unwrap();
    assert!(f.build().compiled, "touching a .cu source must rerun the script");
}

#[test]
fn other_worktree_ref_changes_do_not_rebuild_and_own_commits_do() {
    scenario(true);
}

#[test]
fn without_a_reflog_own_commits_still_rebuild_and_other_refs_do_not() {
    scenario(false);
}

fn edit_main(f: &Fixture, text: &str) {
    let path = f.probe().join("src/main.rs");
    let mut source = std::fs::read_to_string(&path).unwrap();
    source.push_str(text);
    std::fs::write(path, source).unwrap();
}

#[test]
fn an_uncommitted_source_edit_marks_the_build_dirty_with_a_digest_of_that_source() {
    let f = Fixture::new("dirty-source", true);
    let clean = f.build();
    assert!(!clean.dirty && clean.source_sha == "-", "a clean build records no digest: dirty={} sha={} stdout={:?}\n{}", clean.dirty, clean.source_sha, clean.stdout, clean.log);
    f.assert_fresh("second clean build");

    edit_main(&f, "// edit a\n");
    let a = f.build();
    assert!(a.compiled && a.dirty, "an edit under src must rerun the script\n{}", a.log);
    assert_eq!(a.commit, clean.commit, "the commit stays the clean commit");
    assert_eq!(a.source_sha.len(), 64, "{}", a.source_sha);
    f.assert_fresh("build after the edit");

    edit_main(&f, "// edit b\n");
    let b = f.build();
    assert!(b.dirty && b.source_sha.len() == 64 && b.source_sha != a.source_sha, "a different edit gives a different digest");

    git(&f.wt, &["checkout", "--", "probe/src/main.rs"]);
    let reverted = f.build();
    assert!(reverted.compiled && !reverted.dirty && reverted.source_sha == "-", "a reverted edit gives the clean build again\n{}", reverted.log);
}

#[test]
fn an_untracked_new_source_file_makes_the_build_dirty() {
    let f = Fixture::new("dirty-untracked", true);
    assert!(!f.build().dirty);
    std::fs::write(f.probe().join("src/extra.rs"), "// new\n").unwrap();
    let b = f.build();
    assert!(b.compiled && b.dirty && b.source_sha.len() == 64, "{}", b.log);
}
