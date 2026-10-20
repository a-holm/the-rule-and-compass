//! Minimal ZIP reader for market-data archives (issue #123).
//!
//! Binance Vision monthly klines are ZIP files with one deflated CSV per
//! archive, and Bitstamp publishes the same shape. The core crate has no
//! third-party dependencies, so this module implements the small subset of ZIP
//! that matters:
//!
//! * the central directory (method 0 *stored* and 8 *deflate*);
//! * a DEFLATE decoder (stored/fixed/dynamic Huffman blocks);
//! * ZIP64 and encrypted entries are refused with a clear message.
//!
//! Entries are decompressed whole. A monthly 1m kline CSV is a few MB; the
//! caller streams one archive at a time.

use std::io::Read;

/// One file inside a ZIP archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub compressed_size: usize,
    pub uncompressed_size: usize,
    pub data_offset: usize,
}

/// A parsed archive: central-directory metadata only, bytes are read on demand.
#[derive(Clone, Debug)]
pub struct ZipArchive {
    bytes: Vec<u8>,
    pub entries: Vec<ZipEntry>,
}

impl ZipArchive {
    pub fn open(path: &str) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("{}: {}", path, e))?;
        Self::from_bytes(bytes).map_err(|e| format!("{}: {}", path, e))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let eocd = find_eocd(&bytes).ok_or("not a ZIP archive (no end-of-central-directory)")?;
        let entries_count = u16::from_le_bytes([bytes[eocd + 10], bytes[eocd + 11]]) as usize;
        let cd_size = u32::from_le_bytes([
            bytes[eocd + 12],
            bytes[eocd + 13],
            bytes[eocd + 14],
            bytes[eocd + 15],
        ]) as usize;
        let cd_offset = u32::from_le_bytes([
            bytes[eocd + 16],
            bytes[eocd + 17],
            bytes[eocd + 18],
            bytes[eocd + 19],
        ]) as usize;
        if entries_count == 0xffff || cd_size == 0xffff_ffff || cd_offset == 0xffff_ffff {
            return Err("ZIP64 archives are not supported; unzip the archive first".to_string());
        }
        if cd_offset + cd_size > bytes.len() {
            return Err("truncated ZIP central directory".to_string());
        }
        let mut entries = Vec::with_capacity(entries_count);
        let mut p = cd_offset;
        for _ in 0..entries_count {
            if p + 46 > bytes.len() || u32_at(&bytes, p) != 0x0201_4b50 {
                return Err(format!("bad central-directory entry at offset {}", p));
            }
            let flags = u16_at(&bytes, p + 8);
            if flags & 1 != 0 {
                return Err("encrypted ZIP entries are not supported".to_string());
            }
            let method = u16_at(&bytes, p + 10);
            let compressed_size = u32_at(&bytes, p + 20) as usize;
            let uncompressed_size = u32_at(&bytes, p + 24) as usize;
            let name_len = u16_at(&bytes, p + 28) as usize;
            let extra_len = u16_at(&bytes, p + 30) as usize;
            let comment_len = u16_at(&bytes, p + 32) as usize;
            let local_offset = u32_at(&bytes, p + 42) as usize;
            if p + 46 + name_len > bytes.len() {
                return Err("truncated ZIP entry name".to_string());
            }
            let name = String::from_utf8_lossy(&bytes[p + 46..p + 46 + name_len]).to_string();
            // Read the local header to find the actual data offset.
            if local_offset + 30 > bytes.len() || u32_at(&bytes, local_offset) != 0x0403_4b50 {
                return Err(format!("bad local header for '{}'", name));
            }
            let l_name = u16_at(&bytes, local_offset + 26) as usize;
            let l_extra = u16_at(&bytes, local_offset + 28) as usize;
            let data_offset = local_offset + 30 + l_name + l_extra;
            if data_offset + compressed_size > bytes.len() {
                return Err(format!("truncated data block for '{}'", name));
            }
            entries.push(ZipEntry {
                name,
                method,
                compressed_size,
                uncompressed_size,
                data_offset,
            });
            p += 46 + name_len + extra_len + comment_len;
        }
        Ok(ZipArchive { bytes, entries })
    }

    /// Decompress one entry and return its bytes. Directories are empty.
    pub fn read(&self, entry: &ZipEntry) -> Result<Vec<u8>, String> {
        let raw = &self.bytes[entry.data_offset..entry.data_offset + entry.compressed_size];
        let out = match entry.method {
            0 => raw.to_vec(),
            8 => inflate(raw)?,
            other => {
                return Err(format!(
                    "'{}': compression method {} is not supported (stored/deflate only)",
                    entry.name, other
                ))
            }
        };
        if entry.uncompressed_size != 0 && out.len() != entry.uncompressed_size {
            return Err(format!(
                "'{}': inflated {} bytes, header says {}",
                entry.name,
                out.len(),
                entry.uncompressed_size
            ));
        }
        Ok(out)
    }

    /// Concatenate every file whose name ends with `ext` (case-insensitive).
    pub fn read_extension(&self, ext: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
        let mut out = Vec::new();
        for e in &self.entries {
            if e.name.to_ascii_lowercase().ends_with(&ext.to_ascii_lowercase()) {
                out.push((e.name.clone(), self.read(e)?));
            }
        }
        Ok(out)
    }
}

fn u16_at(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}

fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}

fn find_eocd(b: &[u8]) -> Option<usize> {
    if b.len() < 22 {
        return None;
    }
    let start = b.len().saturating_sub(65_557);
    (start..=b.len() - 22)
        .rev()
        .find(|&p| u32_at(b, p) == 0x0605_4b50)
}

// ---------------------------------------------------------------------------
// DEFLATE (RFC 1951)
// ---------------------------------------------------------------------------

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, bit: 0 }
    }

    fn need(&self, n: u32) -> Result<(), String> {
        if self.pos as u64 * 8 + self.bit as u64 + n as u64 > self.data.len() as u64 * 8 {
            Err("truncated DEFLATE stream".to_string())
        } else {
            Ok(())
        }
    }

    fn read(&mut self, n: u32) -> Result<u32, String> {
        self.need(n)?;
        let mut v = 0u32;
        for i in 0..n {
            let byte = self.data[self.pos];
            let bit = (byte >> self.bit) & 1;
            v |= (bit as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

/// Canonical Huffman decoding table. DEFLATE transmits Huffman codes
/// most-significant-bit first while other fields are least-significant-bit
/// first; `read` returns bits LSB-first, so the decoder accumulates a
/// left-shifted code.
struct Huffman {
    /// (length, canonical code, symbol), sorted by length for a linear search.
    codes: Vec<(u8, u16, u16)>,
}


impl Huffman {
    fn from_lengths(lengths: &[u8]) -> Result<Self, String> {
        let max = *lengths.iter().max().unwrap_or(&0) as usize;
        if max == 0 {
            return Ok(Huffman { codes: Vec::new() });
        }
        if max > 15 {
            return Err("DEFLATE code length exceeds 15".to_string());
        }
        let mut count = vec![0u16; max + 1];
        for &l in lengths {
            if l > 0 {
                count[l as usize] += 1;
            }
        }
        let mut next = vec![0u16; max + 1];
        let mut code = 0u16;
        for bits in 1..=max {
            code = (code + count[bits - 1]) << 1;
            next[bits] = code;
        }
        let mut codes = Vec::new();
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            codes.push((l, c, sym as u16));
        }
        codes.sort_by_key(|&(l, c, _)| (l, c));
        Ok(Huffman { codes })
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, String> {
        let mut code = 0u16;
        let mut len = 0u8;
        loop {
            code = (code << 1) | bits.read(1)? as u16;
            len += 1;
            if let Some(&(_, _, sym)) = self.codes.iter().find(|&&(l, c, _)| l == len && c == code) {
                return Ok(sym);
            }
            if len > 15 {
                return Err("invalid DEFLATE Huffman code".to_string());
            }
        }
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13,
];

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = vec![0u8; 288];
    for (i, l) in lit.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let dist = vec![5u8; 30];
    (
        Huffman::from_lengths(&lit).expect("fixed literal table"),
        Huffman::from_lengths(&dist).expect("fixed distance table"),
    )
}

/// Inflate a raw DEFLATE stream (no zlib header, as inside a ZIP entry).
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut bits = Bits::new(data);
    let mut out = Vec::new();
    loop {
        let final_block = bits.read(1)? == 1;
        let kind = bits.read(2)?;
        match kind {
            0 => {
                bits.align();
                if bits.pos + 4 > data.len() {
                    return Err("truncated stored DEFLATE block".to_string());
                }
                let len = u16::from_le_bytes([data[bits.pos], data[bits.pos + 1]]) as usize;
                let nlen = u16::from_le_bytes([data[bits.pos + 2], data[bits.pos + 3]]) as usize;
                if len != (!nlen & 0xffff) {
                    return Err("stored DEFLATE block length check failed".to_string());
                }
                bits.pos += 4;
                if bits.pos + len > data.len() {
                    return Err("truncated stored DEFLATE block data".to_string());
                }
                out.extend_from_slice(&data[bits.pos..bits.pos + len]);
                bits.pos += len;
            }
            1 | 2 => {
                let (lit, dist) = if kind == 1 {
                    fixed_tables()
                } else {
                    dynamic_tables(&mut bits)?
                };
                loop {
                    let sym = lit.decode(&mut bits)?;
                    match sym {
                        0..=255 => out.push(sym as u8),
                        256 => break,
                        257..=285 => {
                            let i = (sym - 257) as usize;
                            let len = LEN_BASE[i] as usize + bits.read(LEN_EXTRA[i] as u32)? as usize;
                            let dsym = dist.decode(&mut bits)? as usize;
                            if dsym >= 30 {
                                return Err("invalid DEFLATE distance symbol".to_string());
                            }
                            let d = DIST_BASE[dsym] as usize
                                + bits.read(DIST_EXTRA[dsym] as u32)? as usize;
                            if d == 0 || d > out.len() {
                                return Err("DEFLATE back-reference before output start".to_string());
                            }
                            let start = out.len() - d;
                            for k in 0..len {
                                let b = out[start + k];
                                out.push(b);
                            }
                        }
                        _ => return Err(format!("invalid DEFLATE literal symbol {}", sym)),
                    }
                }
            }
            _ => return Err("invalid DEFLATE block type".to_string()),
        }
        if final_block {
            break;
        }
    }
    Ok(out)
}

fn dynamic_tables(bits: &mut Bits) -> Result<(Huffman, Huffman), String> {
    let hlit = bits.read(5)? as usize + 257;
    let hdist = bits.read(5)? as usize + 1;
    let hclen = bits.read(4)? as usize + 4;
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut cl = vec![0u8; 19];
    for &i in ORDER.iter().take(hclen) {
        cl[i] = bits.read(3)? as u8;
    }
    let clh = Huffman::from_lengths(&cl)?;
    let mut lengths = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        let sym = clh.decode(bits)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths.last().ok_or("repeat code with no previous length")?;
                let n = 3 + bits.read(2)? as usize;
                lengths.extend(std::iter::repeat(prev).take(n));
            }
            17 => {
                let n = 3 + bits.read(3)? as usize;
                lengths.extend(std::iter::repeat(0).take(n));
            }
            18 => {
                let n = 11 + bits.read(7)? as usize;
                lengths.extend(std::iter::repeat(0).take(n));
            }
            _ => return Err("invalid code-length symbol".to_string()),
        }
    }
    if lengths.len() != hlit + hdist {
        return Err("DEFLATE code lengths overrun".to_string());
    }
    let lit = Huffman::from_lengths(&lengths[..hlit])?;
    let dist = Huffman::from_lengths(&lengths[hlit..])?;
    Ok((lit, dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflate_fixed_huffman() {
        let bytes = hex("cb48cdc9c957c8c04e720100");
        assert_eq!(inflate(&bytes).unwrap(), b"hello hello hello hello hello\n");
    }

    #[test]
    fn inflate_dynamic_huffman_from_a_real_kline_row() {
        let bytes = hex("edccbb0d80400c03d09e59ac280e4920fb2fc6f1d981c62e5c58d6638fbf01ddad56d30233763ec380b15b82dfafee21abed4006da8de858bf806f94254b962c59b264c992254bd6afd605");
        let expected = "1690000000,100.5,101.2,99.8,100.9,123.4,1690000059,12456.7,42,60.1,6200.2,0\n"
            .repeat(40);
        assert_eq!(inflate(&bytes).unwrap(), expected.as_bytes());
    }

    #[test]
    fn stored_zip_roundtrips() {
        let zip = stored_zip("BTCUSDT-1m-2024-01.csv", b"time,open\n1,2\n");
        let a = ZipArchive::from_bytes(zip).unwrap();
        assert_eq!(a.entries.len(), 1);
        let (_, data) = a.read_extension(".csv").unwrap().remove(0);
        assert_eq!(data, b"time,open\n1,2\n");
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Build a one-entry ZIP using the stored method (local + central + EOCD).
    fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let crc = crc32(data);
        // local
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        let cd = out.len();
        // central
        out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        let cd_size = out.len() - cd;
        // EOCD
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(cd_size as u32).to_le_bytes());
        out.extend_from_slice(&(cd as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let m = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & m);
            }
        }
        !crc
    }
}
