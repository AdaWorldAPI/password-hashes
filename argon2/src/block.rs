//! Argon2 memory block functions

use core::{
    convert::{AsMut, AsRef},
    num::Wrapping,
    ops::{BitXor, BitXorAssign},
};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

// The scalar reference: under `ndarray-simd` only the parity tests reach it.
#[cfg_attr(feature = "ndarray-simd", allow(dead_code))]
const TRUNC: u64 = u32::MAX as u64;

#[rustfmt::skip]
macro_rules! permute_step {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        $a = (Wrapping($a) + Wrapping($b) + (Wrapping(2) * Wrapping(($a & TRUNC) * ($b & TRUNC)))).0;
        $d = ($d ^ $a).rotate_right(32);
        $c = (Wrapping($c) + Wrapping($d) + (Wrapping(2) * Wrapping(($c & TRUNC) * ($d & TRUNC)))).0;
        $b = ($b ^ $c).rotate_right(24);

        $a = (Wrapping($a) + Wrapping($b) + (Wrapping(2) * Wrapping(($a & TRUNC) * ($b & TRUNC)))).0;
        $d = ($d ^ $a).rotate_right(16);
        $c = (Wrapping($c) + Wrapping($d) + (Wrapping(2) * Wrapping(($c & TRUNC) * ($d & TRUNC)))).0;
        $b = ($b ^ $c).rotate_right(63);
    };
}

macro_rules! permute {
    (
        $v0:expr, $v1:expr, $v2:expr, $v3:expr,
        $v4:expr, $v5:expr, $v6:expr, $v7:expr,
        $v8:expr, $v9:expr, $v10:expr, $v11:expr,
        $v12:expr, $v13:expr, $v14:expr, $v15:expr,
    ) => {
        permute_step!($v0, $v4, $v8, $v12);
        permute_step!($v1, $v5, $v9, $v13);
        permute_step!($v2, $v6, $v10, $v14);
        permute_step!($v3, $v7, $v11, $v15);
        permute_step!($v0, $v5, $v10, $v15);
        permute_step!($v1, $v6, $v11, $v12);
        permute_step!($v2, $v7, $v8, $v13);
        permute_step!($v3, $v4, $v9, $v14);
    };
}

/// Structure for the (1 KiB) memory block implemented as 128 64-bit words.
#[derive(Copy, Clone, Debug)]
#[repr(align(64))]
pub struct Block([u64; Self::SIZE / 8]);

impl Block {
    /// Memory block size in bytes
    pub const SIZE: usize = 1024;

    /// Returns a Block initialized with zeros.
    #[must_use]
    pub const fn new() -> Self {
        Self([0u64; Self::SIZE / 8])
    }

    /// Storage index of canonical word `w` (`0..128`).
    ///
    /// Without `ndarray-simd` blocks are stored in canonical order and this is
    /// the identity. With it, blocks are stored in the ROW layout
    /// [`Block::compress_simd`] computes in: canonical word `16i + k` (row
    /// `i`, word `k`) lives in vector `k`, lane `i`, i.e. at `8k + i`. The map
    /// is a public compile-time permutation — it never depends on data, so it
    /// adds no secret-dependent addressing — and it is what lets compress skip
    /// the six transposes that only existed to restore canonical order. Every
    /// read or write of a block's words BY INDEX must go through it; whole-
    /// block operations (XOR, copy, compress) are layout-independent.
    #[inline(always)]
    pub(crate) const fn stored(w: usize) -> usize {
        #[cfg(feature = "ndarray-simd")]
        {
            8 * (w % 16) + w / 16
        }
        #[cfg(not(feature = "ndarray-simd"))]
        {
            w
        }
    }

    /// Rewrite this block's words from storage order ([`Block::stored`]) into
    /// canonical RFC 9106 word order, for a block about to leave the crate
    /// through a public API. A no-op without `ndarray-simd`, where the two
    /// orders coincide.
    #[inline]
    pub(crate) fn canonicalize(&mut self) {
        #[cfg(feature = "ndarray-simd")]
        {
            let stored = self.0;
            for w in 0..Self::SIZE / 8 {
                self.0[w] = stored[Self::stored(w)];
            }
        }
    }

    /// Load a block from a block-sized byte slice (canonical word order in
    /// `input`, stored order in `self`; see [`Block::stored`]).
    #[inline(always)]
    pub(crate) fn load(&mut self, input: &[u8; Block::SIZE]) {
        for (i, chunk) in input.chunks(8).enumerate() {
            self.0[Self::stored(i)] =
                u64::from_le_bytes(chunk.try_into().expect("should be 8 bytes"));
        }
    }

    /// NOTE: do not call this directly. It should only be called via
    /// `Argon2::compress`. Under `ndarray-simd` it is the scalar reference
    /// the parity tests hold [`Block::compress_simd`] to.
    #[cfg_attr(feature = "ndarray-simd", allow(dead_code))]
    #[inline(always)]
    pub(crate) fn compress(rhs: &Self, lhs: &Self) -> Self {
        let r = *rhs ^ lhs;

        // Apply permutations rowwise
        let mut q = r;
        for chunk in q.0.chunks_exact_mut(16) {
            #[rustfmt::skip]
            permute!(
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
                chunk[8], chunk[9], chunk[10], chunk[11],
                chunk[12], chunk[13], chunk[14], chunk[15],
            );
        }

        // Apply permutations columnwise
        for i in 0..8 {
            let b = i * 2;

            #[rustfmt::skip]
            permute!(
                q.0[b], q.0[b + 1],
                q.0[b + 16], q.0[b + 17],
                q.0[b + 32], q.0[b + 33],
                q.0[b + 48], q.0[b + 49],
                q.0[b + 64], q.0[b + 65],
                q.0[b + 80], q.0[b + 81],
                q.0[b + 96], q.0[b + 97],
                q.0[b + 112], q.0[b + 113],
            );
        }

        q ^= &r;
        q
    }

    /// Vertical-lane variant of [`Block::compress`] over `ndarray::simd::U64x8`,
    /// on blocks kept in the stored ROW layout ([`Block::stored`]).
    ///
    /// Each of the two passes runs eight *independent* permutations (one per
    /// row, then one per column pair), so lane `i` of every vector carries
    /// permutation `i` and the eight `G` rounds of a pass execute as one. The
    /// backend (AVX-512, AVX2, NEON, wasm-simd128 or scalar) is chosen at
    /// compile time by `ndarray::simd`, so this crate carries no intrinsics
    /// and no `unsafe`.
    ///
    /// Because blocks are STORED in the row layout, vector `k` of `rhs ^ lhs`
    /// already is `v[k]` (lane `i` = canonical word `16i + k`): nothing moves
    /// on load or store. The only lane changes are the row → column
    /// rendezvous and its return, 2 [`U64x8::transpose8`] each — the ones the
    /// algorithm itself requires (a column `G` reads words from four rows).
    /// The diagonal `G` step of each pass is only a choice of registers.
    ///
    /// Equivalent to [`Block::compress`] conjugated by the storage map: for
    /// canonical blocks `a`, `b`,
    /// `compress_simd(S(a), S(b)) == S(compress(a, b))`.
    ///
    /// NOTE: do not call this directly. It should only be called via
    /// `Argon2::compress`.
    #[cfg(feature = "ndarray-simd")]
    #[inline(always)]
    pub(crate) fn compress_simd(rhs: &Self, lhs: &Self) -> Self {
        use core::array::from_fn;
        use ndarray::simd::U64x8;

        /// `a + b + 2 * lo32(a) * lo32(b)` in every lane (RFC 9106 `fBlaMka`).
        #[inline(always)]
        fn blamka(a: U64x8, b: U64x8) -> U64x8 {
            let m = a.mul_lo32(b);
            a + b + m + m
        }

        #[inline(always)]
        fn g(v: &mut [U64x8; 16], a: usize, b: usize, c: usize, d: usize) {
            v[a] = blamka(v[a], v[b]);
            v[d] = (v[d] ^ v[a]).rotate_right(32);
            v[c] = blamka(v[c], v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(24);
            v[a] = blamka(v[a], v[b]);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = blamka(v[c], v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(63);
        }

        #[inline(always)]
        fn permute(v: &mut [U64x8; 16]) {
            g(v, 0, 4, 8, 12);
            g(v, 1, 5, 9, 13);
            g(v, 2, 6, 10, 14);
            g(v, 3, 7, 11, 15);
            g(v, 0, 5, 10, 15);
            g(v, 1, 6, 11, 12);
            g(v, 2, 7, 8, 13);
            g(v, 3, 4, 9, 14);
        }

        /// Transposes the even-indexed and the odd-indexed halves of `x`
        /// separately and interleaves the results again: `y[2j + b]` lane `i`
        /// is `x[2i + b]` lane `j`.
        #[inline(always)]
        fn transpose_pairs(x: &[U64x8; 16]) -> [U64x8; 16] {
            let even = U64x8::transpose8(from_fn(|i| x[2 * i]));
            let odd = U64x8::transpose8(from_fn(|i| x[2 * i + 1]));
            from_fn(|k| {
                if k & 1 == 0 {
                    even[k >> 1]
                } else {
                    odd[k >> 1]
                }
            })
        }

        // Row layout, straight from storage: `r[k]` lane `i` = word `16i + k`.
        let r: [U64x8; 16] = from_fn(|k| {
            U64x8::from_slice(&rhs.0[8 * k..8 * k + 8])
                ^ U64x8::from_slice(&lhs.0[8 * k..8 * k + 8])
        });
        let mut v = r;
        permute(&mut v);

        // Column layout: `c[2r + b]` lane `i` = row `r`, word `2i + b`.
        let mut c = transpose_pairs(&v);
        permute(&mut c);

        // Back to the row layout (`transpose_pairs` is its own inverse) — which
        // is also the storage layout — and add `R` back in.
        let v = transpose_pairs(&c);
        let mut q = Self::new();
        for k in 0..16 {
            (v[k] ^ r[k]).copy_to_slice(&mut q.0[8 * k..8 * k + 8]);
        }
        q
    }
}

impl Default for Block {
    fn default() -> Self {
        Self([0u64; Self::SIZE / 8])
    }
}

impl AsRef<[u64]> for Block {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

impl AsMut<[u64]> for Block {
    fn as_mut(&mut self) -> &mut [u64] {
        &mut self.0
    }
}

impl BitXor<&Block> for Block {
    type Output = Block;

    fn bitxor(mut self, rhs: &Block) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl BitXorAssign<&Block> for Block {
    fn bitxor_assign(&mut self, rhs: &Block) {
        for (dst, src) in self.0.iter_mut().zip(rhs.0.iter()) {
            *dst ^= src;
        }
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for Block {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

/// Custom implementation of `Box<[Block]>` until `Box::try_new_zeroed_slice` is stabilized.
#[cfg(feature = "alloc")]
pub(crate) struct Blocks {
    p: core::ptr::NonNull<Block>,
    len: usize,
}

#[cfg(feature = "alloc")]
impl Blocks {
    pub fn new(len: usize) -> Option<Self> {
        use alloc::alloc::{Layout, alloc_zeroed};
        use core::ptr::NonNull;

        if len == 0 {
            return None;
        }

        let layout = Layout::array::<Block>(len).ok()?;
        // SAFETY: `alloc_zeroed` is used correctly with non-zero layout
        let p = unsafe { alloc_zeroed(layout) };

        let p = NonNull::new(p.cast())?;
        Some(Self { p, len })
    }

    pub fn as_slice(&mut self) -> &mut [Block] {
        // SAFETY: `self.p` is a valid non-zero pointer that points to memory of the necessary size
        unsafe { core::slice::from_raw_parts_mut(self.p.as_ptr(), self.len) }
    }
}

#[cfg(feature = "alloc")]
impl Drop for Blocks {
    fn drop(&mut self) {
        use alloc::alloc::{Layout, dealloc};
        // SAFETY: layout was checked during construction
        let layout = unsafe { Layout::array::<Block>(self.len).unwrap_unchecked() };
        // SAFETY: we use `dealloc` correctly with the previously allocated pointer
        unsafe {
            dealloc(self.p.as_ptr().cast(), layout);
        }
    }
}

#[cfg(all(test, feature = "ndarray-simd"))]
mod tests {
    use super::Block;

    /// `SplitMix64`, so the test needs no RNG dependency.
    fn fill(seed: u64) -> Block {
        let mut s = seed;
        let mut b = Block::new();
        for w in b.0.iter_mut() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *w = z ^ (z >> 31);
        }
        b
    }

    /// Canonical order -> stored order.
    fn to_stored(b: &Block) -> Block {
        let mut s = Block::new();
        for w in 0..128 {
            s.0[Block::stored(w)] = b.0[w];
        }
        s
    }

    #[test]
    fn stored_is_a_permutation_that_fixes_word_zero() {
        let mut seen = [false; 128];
        for w in 0..128 {
            let s = Block::stored(w);
            assert!(!seen[s], "storage index {s} reused");
            seen[s] = true;
        }
        assert_eq!(Block::stored(0), 0);
        // Not the identity: a map that is the identity would make the
        // conjugation test below vacuous.
        assert_ne!(Block::stored(1), 1);
    }

    #[test]
    fn compress_simd_is_scalar_compress_in_stored_order() {
        for seed in 0..64 {
            let (rhs, lhs) = (fill(2 * seed), fill(2 * seed + 1));
            assert_eq!(
                Block::compress_simd(&to_stored(&rhs), &to_stored(&lhs)).0,
                to_stored(&Block::compress(&rhs, &lhs)).0,
                "seed {seed}"
            );
        }
    }
}
