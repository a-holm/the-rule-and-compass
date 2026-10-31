#![allow(dead_code)]

use rcsr_acquire::transport::{Response, Transport};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub fn fixture(name: &str) -> Vec<u8> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    fs::read(&p).unwrap_or_else(|e| panic!("fixture {}: {e}", p.display()))
}

pub fn fixture_text(name: &str) -> String {
    String::from_utf8(fixture(name)).expect("utf8 fixture")
}

/// Fresh empty directory under the target tmp dir.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch");
    dir
}

/// Serves canned responses by URL and records every request.
#[derive(Default)]
pub struct Fake {
    pub routes: HashMap<String, (u16, Vec<(String, String)>, Vec<u8>)>,
    pub requests: Mutex<Vec<String>>,
}

impl Fake {
    pub fn route(&mut self, url: &str, status: u16, headers: &[(&str, &str)], body: &[u8]) {
        let h = headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        self.routes.insert(url.to_string(), (status, h, body.to_vec()));
    }

    pub fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl Transport for Fake {
    fn get(&self, url: &str, dest: &Path) -> Result<Response, String> {
        self.requests.lock().unwrap().push(url.to_string());
        match self.routes.get(url) {
            Some((status, headers, body)) => {
                fs::write(dest, body).map_err(|e| e.to_string())?;
                Ok(Response { status: *status, headers: headers.clone() })
            }
            None => Err(format!("no route for {url}")),
        }
    }
}
