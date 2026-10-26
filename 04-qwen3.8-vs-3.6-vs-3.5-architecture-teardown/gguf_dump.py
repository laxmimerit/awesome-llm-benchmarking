"""Dump GGUF metadata + tensor summary from Ollama blobs (read-only)."""
import struct, sys, json

GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_val(f, t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return bool(struct.unpack('<B', f.read(1))[0])
    if t == 8: return read_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        vals = [read_val(f, et) for _ in range(n)]
        if n > 12:
            return {'_array_len': n, '_elem_type': GGUF_TYPES.get(et), '_head': vals[:6], '_tail': vals[-3:]}
        return vals
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f'unknown type {t}')

GGML_DTYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16'}

def dump(path):
    out = {'path': path, 'meta': {}, 'tensors': []}
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f'not gguf: {magic}'
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        out['version'] = version
        out['n_tensors'] = n_tensors
        for _ in range(n_kv):
            key = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            out['meta'][key] = read_val(f, t)
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{nd}Q', f.read(8*nd)))
            dt = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            out['tensors'].append({'name': name, 'dims': dims, 'dtype': GGML_DTYPES.get(dt, dt)})
    return out

if __name__ == '__main__':
    results = {}
    for label, path in json.loads(sys.argv[1]).items():
        results[label] = dump(path)
    with open(sys.argv[2], 'w') as f:
        json.dump(results, f, indent=1)
    # console summary
    for label, r in results.items():
        m = r['meta']
        arch = m.get('general.architecture')
        print(f"=== {label} === v{r['version']} tensors={r['n_tensors']} arch={arch}")
        for k in sorted(m):
            if k.startswith(('general.', f'{arch}.', 'tokenizer.ggml.model', 'tokenizer.ggml.pre')) and not isinstance(m[k], dict):
                print(f"  {k} = {m[k]}")
            elif isinstance(m[k], dict) and '_array_len' in m[k]:
                print(f"  {k} = array[{m[k]['_array_len']}] of {m[k]['_elem_type']}")
