use super::*;

fn make_simple_gguf() -> Vec<u8> {
    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(), Value::String("llama".into()));
    md.insert("general.alignment".into(), Value::U32(32));
    md.insert("llama.embedding_length".into(), Value::U32(4));
    md.insert("llama.block_count".into(), Value::U32(2));
    md.insert(
        "test.array.u32".into(),
        Value::Array(Array::U32(vec![1, 2, 3, 4])),
    );
    md.insert(
        "test.array.string".into(),
        Value::Array(Array::String(vec!["alpha".into(), "beta".into()])),
    );

    let t1_data: Vec<u8> = (0u32..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let t1 = TensorInfo {
        name: "weight.a".into(),
        shape: vec![4, 2],
        dtype: GgmlType::F32,
        offset: 0,
    };

    let t2_data: Vec<u8> = (0u32..4)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    let t2 = TensorInfo {
        name: "weight.b".into(),
        shape: vec![4],
        dtype: GgmlType::F32,
        offset: 0,
    };

    write_to_vec(&md, &[(t1, t1_data), (t2, t2_data)], 32).unwrap()
}

#[test]
fn roundtrip_simple() {
    let bytes = make_simple_gguf();
    let f = GgufHeader::from_bytes(&bytes).unwrap();

    assert_eq!(f.architecture().unwrap(), "llama");
    assert_eq!(f.get_u64("llama.embedding_length").unwrap(), 4);
    assert_eq!(f.get_u64("llama.block_count").unwrap(), 2);
    assert_eq!(f.alignment(), 32);

    let arr = f.get("test.array.u32").unwrap().as_array().unwrap();
    match arr {
        Array::U32(v) => assert_eq!(v, &vec![1, 2, 3, 4]),
        _ => panic!("expected U32 array"),
    }

    let arr = f.get("test.array.string").unwrap().as_array().unwrap();
    match arr {
        Array::String(v) => assert_eq!(v, &vec!["alpha".to_string(), "beta".into()]),
        _ => panic!("expected String array"),
    }

    assert_eq!(f.tensors().len(), 2);
    let a = f.tensor_by_name("weight.a").unwrap();
    assert_eq!(a.shape, vec![4, 2]);
    assert_eq!(a.dtype, GgmlType::F32);
    assert_eq!(a.numel(), 8);
    assert_eq!(a.nbytes(), 32);
    let range = f.tensor_range(a);
    assert_eq!(range.bytes, 32);

    let b = f.tensor_by_name("weight.b").unwrap();
    assert_eq!(b.numel(), 4);

    // Verify tensor data round-trips. A header says where the body is;
    // slicing it is the caller's job, because only the caller has the file.
    let bytes_a = &bytes[range.offset as usize..range.end() as usize];
    let recovered: Vec<f32> = bytes_a
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(recovered, (0..8).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn rejects_bad_magic() {
    let bytes = vec![0u8; 64];
    let err = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::BadMagic(_)));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&99u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let err = GgufHeader::from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion(99, _)));
}

#[test]
fn alignment_padding_is_correct() {
    let bytes = make_simple_gguf();
    let f = GgufHeader::from_bytes(&bytes).unwrap();
    assert_eq!(f.tensor_data_start() % f.alignment(), 0);
}
