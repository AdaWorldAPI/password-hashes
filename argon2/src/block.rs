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

    /// Folded-layout variant of [`Block::compress`] over `ndarray::simd::U64x8`.
    ///
    /// Inputs and output are in the *folded* word order (see [`fold_index`]):
    /// storage vector `m` (words `8m .. 8m + 8`) holds the column-pass lanes
    /// `c[m]`, i.e. lane `i` of `c[2r + b]` is canonical word `16r + 2i + b`.
    /// Every block `fill_blocks` touches stays in that order from one call to
    /// the next, so the canonical order is never materialized between calls:
    ///
    /// - load: the 16 storage vectors *are* the column layout (0 transposes);
    /// - column → row layout for the row pass (2 transposes);
    /// - row → column layout, the rendezvous the algorithm requires
    ///   (2 transposes);
    /// - store: the column-pass result is already in storage order
    ///   (0 transposes).
    ///
    /// 4 transposes per call, against 8 when blocks are kept canonical. Each
    /// pass runs eight independent permutations, lane `i` carrying
    /// permutation `i`; the diagonal `G` step is only a choice of registers.
    /// The result is bit-identical to [`Block::compress`] under the fold; the
    /// backend is chosen at compile time by `ndarray::simd`, so this crate
    /// carries no intrinsics and no `unsafe`.
    ///
    /// NOTE: do not call this directly. It should only be called via
    /// `Argon2::compress`.
    #[cfg(feature = "ndarray-simd")]
    #[inline(always)]
    pub(crate) fn compress_folded(rhs: &Self, lhs: &Self) -> Self {
        use lanes::{U64x8, permute, transpose_pairs};

        // `R = rhs ^ lhs`, one storage chunk per vector: already `c[m]`.
        let mut r = [U64x8::splat(0); 16];
        for ((rv, a), b) in r
            .iter_mut()
            .zip(rhs.0.chunks_exact(8))
            .zip(lhs.0.chunks_exact(8))
        {
            *rv = U64x8::from_slice(a) ^ U64x8::from_slice(b);
        }

        // Row layout: `v[k]` lane `i` = canonical word `16i + k`.
        let mut v = transpose_pairs(&r);
        permute(&mut v);

        // Column layout again: `c[2r + b]` lane `i` = word `16r + 2i + b`.
        let mut c = transpose_pairs(&v);
        permute(&mut c);

        let mut q = Self::new();
        for (m, out) in q.0.chunks_exact_mut(8).enumerate() {
            (c[m] ^ r[m]).copy_to_slice(out);
        }
        q
    }

    /// This block's canonical words reordered into the folded layout.
    #[cfg(feature = "ndarray-simd")]
    pub(crate) fn fold(&self) -> Self {
        let mut out = Self::new();
        for (w, &x) in self.0.iter().enumerate() {
            out.0[fold_index(w)] = x;
        }
        out
    }

    /// Inverse of [`Block::fold`]: folded storage back to canonical order.
    #[cfg(feature = "ndarray-simd")]
    pub(crate) fn unfold(&self) -> Self {
        let mut out = Self::new();
        for (w, x) in out.0.iter_mut().enumerate() {
            *x = self.0[fold_index(w)];
        }
        out
    }

    /// Canonical word `w` of this block, whatever the storage order.
    #[inline(always)]
    pub(crate) fn word(&self, w: usize) -> u64 {
        self.0[storage_index(w)]
    }

    /// Mutable canonical word `w` of this block, whatever the storage order.
    #[inline(always)]
    pub(crate) fn word_mut(&mut self, w: usize) -> &mut u64 {
        &mut self.0[storage_index(w)]
    }
}

/// Storage index of canonical word `w` in the folded layout used by
/// [`Block::compress_folded`]: with `w = 16r + 2i + b` (`r` the row, `i` the
/// column pair, `b` the side of the pair), the word sits at lane `i` of
/// storage vector `2r + b`, i.e. index `8(2r + b) + i`.
///
/// Word 0 maps to 0, so the data-dependent addressing read of `prev[0]` is
/// the same in both orders.
#[cfg(feature = "ndarray-simd")]
pub(crate) const fn fold_index(w: usize) -> usize {
    let r = w >> 4;
    let j = w & 15;
    ((2 * r + (j & 1)) << 3) | (j >> 1)
}

#[cfg(feature = "ndarray-simd")]
#[inline(always)]
const fn storage_index(w: usize) -> usize {
    fold_index(w)
}

#[cfg(not(feature = "ndarray-simd"))]
#[inline(always)]
const fn storage_index(w: usize) -> usize {
    w
}

/// The lane round shared by the SIMD compression function.
#[cfg(feature = "ndarray-simd")]
mod lanes {
    use core::array::from_fn;
    pub(super) use ndarray::simd::U64x8;

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

    /// One Blake2b-style permutation in every lane: a column step, then the
    /// diagonal step (register choice only, no data moves).
    #[inline(always)]
    pub(super) fn permute(v: &mut [U64x8; 16]) {
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
    /// is `x[2i + b]` lane `j`. It is its own inverse.
    #[inline(always)]
    pub(super) fn transpose_pairs(x: &[U64x8; 16]) -> [U64x8; 16] {
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

#[cfg(all(test, feature = "ndarray-simd"))]
mod tests {
    use super::Block;

    /// SplitMix64, so the test needs no RNG dependency.
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

    #[test]
    fn compress_folded_matches_scalar_under_the_fold() {
        for seed in 0..64 {
            let (rhs, lhs) = (fill(2 * seed), fill(2 * seed + 1));
            assert_eq!(
                Block::compress_folded(&rhs.fold(), &lhs.fold()).unfold().0,
                Block::compress(&rhs, &lhs).0,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn fold_index_is_a_permutation_fixing_word_zero() {
        let mut seen = [false; 128];
        for w in 0..128 {
            let s = super::fold_index(w);
            assert!(!seen[s], "storage slot {s} reached twice");
            seen[s] = true;
        }
        assert_eq!(super::fold_index(0), 0);
        // Not the identity: a fold that changes nothing proves nothing above.
        assert_ne!(super::fold_index(1), 1);
        let b = fill(7);
        assert_eq!(b.fold().unfold().0, b.0);
        for w in 0..128 {
            assert_eq!(b.fold().word(w), b.0[w]);
        }
    }
}
