//! Argon2 memory block functions

use core::{
    convert::{AsMut, AsRef},
    num::Wrapping,
    ops::{BitXor, BitXorAssign},
    slice,
};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

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

    /// Load a block from a block-sized byte slice
    #[inline(always)]
    pub(crate) fn load(&mut self, input: &[u8; Block::SIZE]) {
        for (i, chunk) in input.chunks(8).enumerate() {
            self.0[i] = u64::from_le_bytes(chunk.try_into().expect("should be 8 bytes"));
        }
    }

    /// Iterate over the `u64` values contained in this block
    #[inline(always)]
    pub(crate) fn iter(&self) -> slice::Iter<'_, u64> {
        self.0.iter()
    }

    /// NOTE: do not call this directly. It should only be called via
    /// `Argon2::compress`.
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

    /// Vertical-lane variant of [`Block::compress`] over `ndarray::simd::U64x8`.
    ///
    /// Each of the two passes runs eight *independent* permutations (one per
    /// row, then one per column pair), so lane `i` of every vector carries
    /// permutation `i` and the eight `G` rounds of a pass execute as one. The
    /// result is bit-identical to [`Block::compress`]; the backend (AVX-512,
    /// AVX2, NEON, wasm-simd128 or scalar) is chosen at compile time by
    /// `ndarray::simd`, so this crate carries no intrinsics and no `unsafe`.
    ///
    /// NOTE: do not call this directly. It should only be called via
    /// `Argon2::compress`.
    #[cfg(feature = "ndarray-simd")]
    #[inline(always)]
    pub(crate) fn compress_simd(rhs: &Self, lhs: &Self) -> Self {
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

        /// Runs one pass: word `k` of permutation `i` lives at `q[at(i, k)]`.
        #[inline(always)]
        fn pass(q: &mut [u64; 128], at: fn(usize, usize) -> usize) {
            let mut v = [U64x8::splat(0); 16];
            for (k, vk) in v.iter_mut().enumerate() {
                let mut lanes = [0u64; 8];
                for (i, lane) in lanes.iter_mut().enumerate() {
                    *lane = q[at(i, k)];
                }
                *vk = U64x8::from_array(lanes);
            }
            permute(&mut v);
            for (k, vk) in v.iter().enumerate() {
                for (i, lane) in vk.to_array().into_iter().enumerate() {
                    q[at(i, k)] = lane;
                }
            }
        }

        let r = *rhs ^ lhs;
        let mut q = r;
        // Row pass: permutation `i` is the 16 consecutive words of row `i`.
        pass(&mut q.0, |i, k| 16 * i + k);
        // Column pass: permutation `i` is column pair `2i, 2i + 1` of all rows.
        pass(&mut q.0, |i, k| 2 * i + 16 * (k >> 1) + (k & 1));
        q ^= &r;
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
        unsafe { slice::from_raw_parts_mut(self.p.as_ptr(), self.len) }
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
