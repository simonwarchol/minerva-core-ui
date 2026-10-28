/**
 * Float16 conversion utilities — converts uint16 intensity values (0-65535)
 * to IEEE 754 half-precision for WebGPU's r16float texture format.
 */

/** Convert a float32 value to float16 binary representation (as uint16). */
// Module-level scratch buffers — shared across all calls to avoid per-call allocation
const _f32Scratch = new Float32Array(1);
const _u32Scratch = new Uint32Array(_f32Scratch.buffer);

export function float32ToFloat16Bits(f32: number): number {
  // Write float32 into the shared scratch buffer and read back as uint32 bits.
  // Avoids allocating a new ArrayBuffer + DataView on every call (287k calls per brick).
  _f32Scratch[0] = f32;
  const bits = _u32Scratch[0]!;

  // Extract IEEE 754 float32 components
  const sign = (bits >> 31) & 0x1;
  let exp = (bits >> 23) & 0xff;
  let frac = bits & 0x7fffff;

  // Handle special cases
  if (exp === 0xff) {
    // Infinity or NaN
    return (sign << 15) | 0x7c00 | (frac ? 1 : 0);
  }

  if (exp === 0) {
    // Zero or denormal
    return sign << 15;
  }

  // Rebias exponent: float32 bias=127, float16 bias=15
  exp = exp - 127 + 15;

  if (exp >= 0x1f) {
    // Overflow to infinity
    return (sign << 15) | 0x7c00;
  }

  if (exp <= 0) {
    // Underflow to zero
    return sign << 15;
  }

  // Normal case: pack into float16 format
  // Sign (1 bit) | Exponent (5 bits) | Mantissa (10 bits)
  frac = frac >> 13; // Keep top 10 bits of mantissa
  return (sign << 15) | (exp << 10) | frac;
}

/** Decode a uint16 float16 bit pattern to a JavaScript number. */
export function float16BitsToFloat32(bits: number): number {
  const sign = (bits >> 15) & 0x1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;

  if (exp === 0) {
    // Subnormal: (-1)^s * 2^-14 * (mant / 1024)
    return (sign ? -1 : 1) * 2 ** -14 * (mant / 1024);
  }
  if (exp === 31) {
    // Infinity or NaN
    return mant ? NaN : sign ? -Infinity : Infinity;
  }
  // Normal: (-1)^s * 2^(exp-15) * (1 + mant/1024)
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + mant / 1024);
}

// Encode LUT: uint16 intensity (0-65535, pre-normalised by /65535) → float16 bits. 128 KB.
let _u16ToF16Lut: Uint16Array | null = null;

/** Lazily-built LUT mapping a raw uint16 intensity directly to its float16 bit pattern. */
export function getUint16ToFloat16Lut(): Uint16Array {
  if (!_u16ToF16Lut) {
    _u16ToF16Lut = new Uint16Array(65536);
    for (let i = 0; i < 65536; i++)
      _u16ToF16Lut[i] = float32ToFloat16Bits(i / 65535);
  }
  return _u16ToF16Lut;
}

// Decode LUT: float16 bit pattern → float32. 256 KB.
let _f16ToF32Lut: Float32Array | null = null;

/** Lazily-built LUT mapping every float16 bit pattern to its decoded float32 value. */
export function getFloat16ToFloat32Lut(): Float32Array {
  if (!_f16ToF32Lut) {
    _f16ToF32Lut = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) _f16ToF32Lut[i] = float16BitsToFloat32(i);
  }
  return _f16ToF32Lut;
}

/** Convert a Uint16Array (0-65535) to float16 binary format for r16float textures. */
export function uint16ToFloat16(uint16Data: Uint16Array): Uint16Array {
  const lut = getUint16ToFloat16Lut();
  const float16Data = new Uint16Array(uint16Data.length);

  for (let i = 0; i < uint16Data.length; i++) {
    float16Data[i] = lut[uint16Data[i]!]!;
  }

  return float16Data;
}

const NativeFloat16Array = (
  globalThis as unknown as {
    Float16Array?: new (a: ArrayLike<number>) => { buffer: ArrayBuffer };
  }
).Float16Array;

/** Float32 values → float16 bits (native Float16Array conversion when available). */
export function float32ToFloat16Array(f: Float32Array): Uint16Array {
  if (NativeFloat16Array) {
    return new Uint16Array(new NativeFloat16Array(f).buffer);
  }
  const out = new Uint16Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = float32ToFloat16Bits(f[i]!);
  return out;
}
