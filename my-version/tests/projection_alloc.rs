//! #294: an allocator hook proves the projected path never builds a full table.
//!
//! This is its own test binary so the counter is not polluted by unrelated
//! tests running in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rcsr::data;
use rcsr::store::{self, FileStore, Projection};

struct Counting;

static TOTAL: AtomicUsize = AtomicUsize::new(0);
static MAX_SINGLE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        MAX_SINGLE.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            TOTAL.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            MAX_SINGLE.fetch_max(new_size, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-projection-alloc-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_projected_path_never_allocates_a_full_table() {
    let rows = 20_000usize;
    let cols = 364usize;
    let mut header = String::from("ts,symbol");
    for c in 0..(cols - 2) {
        header.push_str(&format!(",v{}", c));
    }
    header.push('\n');
    let mut text = header;
    for r in 0..rows {
        text.push_str(&format!("{},{}", 1_700_000_000i64 + r as i64 * 60, if r % 2 == 0 { "A" } else { "B" }));
        for c in 0..(cols - 2) {
            text.push_str(&format!(",{}", (c + 1) as f64 * 0.5 + (r % 101) as f64 * 0.01));
        }
        text.push('\n');
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    drop(text);
    let dir = tmp_dir("wide");
    let path = dir.join("wide.rcsb").to_string_lossy().into_owned();
    store::write_store_with_time(&path, &table, 4096, Some(0)).unwrap();
    let full_table_bytes = rows * (cols - 1) * 8 + rows * 4;
    drop(table);

    let fs = FileStore::open(&path).unwrap();
    let mut names = vec!["ts".to_string()];
    names.extend((0..20).map(|c| format!("v{}", c)));
    let proj = Projection::resolve_owned(&fs, &names).unwrap();

    TOTAL.store(0, Ordering::Relaxed);
    MAX_SINGLE.store(0, Ordering::Relaxed);
    let projected = fs.to_table_projected(&proj).unwrap();
    let total = TOTAL.load(Ordering::Relaxed);
    let largest = MAX_SINGLE.load(Ordering::Relaxed);

    assert_eq!(projected.n_cols(), 21);
    assert_eq!(projected.names, proj.names());
    // A full-table image would allocate ~58 MB here. The projected path
    // allocates 21 columns plus one group buffer; require a wide margin.
    assert!(
        total < full_table_bytes / 4,
        "projected path allocated {} bytes; a full table is {}",
        total,
        full_table_bytes
    );
    assert!(
        largest < full_table_bytes / 4,
        "largest single allocation {} bytes",
        largest
    );
    let _ = std::fs::remove_dir_all(&dir);
}
