//! Files that are not trying to be read.
//!
//! The parser takes numbers out of a file and allocates according to them, so
//! the interesting inputs are the ones where those numbers are wrong on
//! purpose: a length longer than the file, a count that would reserve a
//! gigabyte, a shape whose product wraps, the same name twice.
//!
//! Every one of these must be an `Err`. None may panic, and none may allocate
//! according to a number the file has not earned -- which on a 32-bit target
//! also means none may quietly truncate a `u64` length into a `usize`.

use gguf::{GgufError, GgufHeader, ParseLimits};

const MAGIC: u32 = 0x4655_4747;

/// A header prefix: magic, version, tensor count, metadata count.
fn head(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&tensors.to_le_bytes());
    out.extend_from_slice(&metadata.to_le_bytes());
    out
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A metadata entry whose value is a u32.
fn u32_entry(out: &mut Vec<u8>, key: &str, value: u32) {
    string(out, key);
    out.extend_from_slice(&4u32.to_le_bytes()); // ValueType::U32
    out.extend_from_slice(&value.to_le_bytes());
}

#[test]
fn a_string_longer_than_the_file_is_refused_not_reserved() {
    let mut bytes = head(0, 1);
    // A key that claims to be sixteen exabytes.
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            error,
            GgufError::TooLarge { .. } | GgufError::TooLargeForMachine(_)
        ),
        "got {error:?}"
    );
}

#[test]
fn an_array_longer_than_the_file_is_refused_not_reserved() {
    let mut bytes = head(0, 1);
    string(&mut bytes, "big");
    bytes.extend_from_slice(&9u32.to_le_bytes()); // ValueType::Array
    bytes.extend_from_slice(&4u32.to_le_bytes()); // of U32
    bytes.extend_from_slice(&(1u64 << 40).to_le_bytes()); // a trillion of them
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    // Four bytes each, and there are none left: the count cannot be honoured.
    assert!(matches!(error, GgufError::TooLarge { .. }), "got {error:?}");
}

#[test]
fn a_tensor_count_larger_than_the_file_is_refused() {
    let bytes = head(u64::MAX, 0);
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(matches!(error, GgufError::TooLarge { .. }), "got {error:?}");
}

#[test]
fn a_metadata_count_larger_than_the_file_is_refused() {
    let bytes = head(0, u64::MAX);
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(matches!(error, GgufError::TooLarge { .. }), "got {error:?}");
}

#[test]
fn a_shape_whose_product_overflows_is_refused() {
    let mut bytes = head(1, 0);
    string(&mut bytes, "huge");
    bytes.extend_from_slice(&2u32.to_le_bytes()); // two dimensions
    bytes.extend_from_slice(&(u64::MAX / 2).to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes()); // product wraps
    bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
    bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(error, GgufError::Overflow(_) | GgufError::TooLarge { .. }),
        "got {error:?}"
    );
}

#[test]
fn the_same_metadata_key_twice_is_refused() {
    let mut bytes = head(0, 2);
    u32_entry(&mut bytes, "general.alignment", 32);
    u32_entry(&mut bytes, "general.alignment", 64);
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            error,
            GgufError::Duplicate {
                what: "metadata key",
                ..
            }
        ),
        "got {error:?}"
    );
}

#[test]
fn the_same_tensor_name_twice_is_refused() {
    let mut bytes = head(2, 0);
    for _ in 0..2 {
        string(&mut bytes, "same");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            error,
            GgufError::Duplicate {
                what: "tensor name",
                ..
            }
        ),
        "got {error:?}"
    );
}

#[test]
fn an_alignment_that_is_not_a_power_of_two_is_refused() {
    for bad in [0u32, 3, 100] {
        let mut bytes = head(0, 1);
        u32_entry(&mut bytes, "general.alignment", bad);
        let error = GgufHeader::from_bytes(&bytes).unwrap_err();
        assert!(
            matches!(error, GgufError::BadAlignment(_)),
            "alignment {bad} gave {error:?}"
        );
    }
}

#[test]
fn a_truncated_file_is_an_error_at_every_length() {
    // Every prefix of a plausible header. None may panic.
    let mut full = head(1, 1);
    u32_entry(&mut full, "general.alignment", 32);
    string(&mut full, "weight");
    full.extend_from_slice(&1u32.to_le_bytes());
    full.extend_from_slice(&4u64.to_le_bytes());
    full.extend_from_slice(&0u32.to_le_bytes());
    full.extend_from_slice(&0u64.to_le_bytes());

    assert!(
        GgufHeader::from_bytes(&full).is_ok(),
        "the full header should parse"
    );
    for cut in 0..full.len() {
        let _ = GgufHeader::from_bytes(&full[..cut]); // must not panic
    }
}

#[test]
fn nothing_at_all_is_an_error() {
    for bytes in [vec![], vec![0u8], vec![0u8; 8], vec![0xff; 64]] {
        assert!(
            GgufHeader::from_bytes(&bytes).is_err(),
            "{} bytes parsed",
            bytes.len()
        );
    }
}

#[test]
fn limits_are_the_callers_to_set() {
    let mut bytes = head(0, 1);
    u32_entry(&mut bytes, "general.alignment", 32);
    assert!(GgufHeader::from_bytes(&bytes).is_ok());

    // One entry, and a limit of zero.
    let strict = ParseLimits {
        max_metadata_entries: 0,
        ..ParseLimits::default()
    };
    let error = GgufHeader::from_bytes_with_limits(&bytes, strict).unwrap_err();
    assert!(
        matches!(
            error,
            GgufError::TooLarge {
                what: "metadata entry count",
                ..
            }
        ),
        "got {error:?}"
    );
}

#[test]
fn a_tensor_offset_that_would_overflow_is_refused() {
    let mut bytes = head(1, 0);
    string(&mut bytes, "far");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes()); // four F32 values = 16 bytes
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // offset at the very end
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(matches!(error, GgufError::Overflow(_)), "got {error:?}");
}

#[test]
fn too_many_dimensions_is_refused() {
    let mut bytes = head(1, 0);
    string(&mut bytes, "deep");
    bytes.extend_from_slice(&9u32.to_le_bytes()); // nine dimensions
                                                  // Enough trailing bytes that the tensor count is plausible for the file;
                                                  // otherwise this is refused earlier, for a different and also correct
                                                  // reason, and would not exercise the dimension check at all.
    bytes.extend_from_slice(&[0u8; 96]);
    let error = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(error, GgufError::TooManyDims { .. }),
        "got {error:?}"
    );
}

/// Whatever bytes come out of this, nothing panics and nothing hangs.
#[test]
fn arbitrary_bytes_never_panic() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..2000 {
        let len = (next() % 256) as usize + 16;
        let mut bytes: Vec<u8> = (0..len).map(|_| (next() & 0xff) as u8).collect();
        // Half of them look like a GGUF to begin with, so the fuzz reaches
        // past the magic check and into the parts that allocate.
        if round % 2 == 0 {
            bytes[..4].copy_from_slice(&MAGIC.to_le_bytes());
            bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        }
        let _ = GgufHeader::from_bytes(&bytes);
    }
}
