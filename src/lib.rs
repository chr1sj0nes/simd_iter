#![no_std]
#![feature(portable_simd)]
#![feature(specialization)]
#![allow(incomplete_features)] // specialization is incomplete

use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

pub use crate::reductions::{IntVectorReductions, NumVectorReductions};
pub use crate::vector::Vector;

mod reductions;
mod scalar;
mod vector;

pub trait SimdIterable<T: SimdElement> {
    // TODO better default num lanes?
    fn simd_iter(&self) -> SimdIter<T, 32> {
        self.simd_iter_with_width::<32>()
    }

    fn simd_iter_with_width<const LANES: usize>(&self) -> SimdIter<T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

pub struct SimdIter<'a, T: SimdElement, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    prefix: &'a [T],
    vectors: &'a [[T; LANES]],
    postfix: &'a [T],
}

pub struct SimdIterPadded<'a, T: SimdElement, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    inner: SimdIter<'a, T, LANES>,
    pad_value: T,
}

impl<'a, T: SimdElement, const LANES: usize> SimdIter<'a, T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn prefix(&self) -> &[T] {
        self.prefix
    }
    pub fn postfix(&self) -> &[T] {
        self.postfix
    }

    pub fn padded_with(self, value: T) -> SimdIterPadded<'a, T, LANES> {
        SimdIterPadded {
            inner: self,
            pad_value: value,
        }
    }
}

// TODO implement `advance_by` and `count`
impl<T: SimdElement, const LANES: usize> Iterator for SimdIter<'_, T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Item = Simd<T, LANES>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((first, rest)) = self.vectors.split_first() {
            self.vectors = rest;
            Some(Simd::from_array(*first))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vectors.len(), Some(self.vectors.len()))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n < self.len() {
            self.vectors = &self.vectors[n..];
            self.next()
        } else {
            self.vectors = &[];
            None
        }
    }
}

impl<T: SimdElement, const LANES: usize> ExactSizeIterator for SimdIter<'_, T, LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}

// TODO override `advance_by` and `count`.
impl<T: SimdElement, const LANES: usize> Iterator for SimdIterPadded<'_, T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Vector<Element = T>,
{
    type Item = Simd<T, LANES>;

    fn next(&mut self) -> Option<Self::Item> {
        let pad_value = self.pad_value;
        let try_take_slice_padded = |values: &mut &[T]| {
            if values.is_empty() {
                None
            } else {
                let vals = *values;
                *values = &[];
                Some(Self::Item::from_slice_padded(vals, pad_value))
            }
        };

        try_take_slice_padded(&mut self.inner.prefix)
            .or_else(|| self.inner.next())
            .or_else(|| try_take_slice_padded(&mut self.inner.postfix))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.inner.len()
            + (!self.inner.prefix.is_empty() as usize)
            + (!self.inner.postfix.is_empty() as usize);
        (n, Some(n))
    }
}

impl<T: SimdElement, const LANES: usize> ExactSizeIterator for SimdIterPadded<'_, T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Vector<Element = T>,
{
}

impl<T: SimdElement, U: AsRef<[T]>> SimdIterable<T> for U {
    fn simd_iter_with_width<const LANES: usize>(&self) -> SimdIter<T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // SAFETY: The transmute is always safe when arrays are aligned.
        let (prefix, vectors, postfix) = unsafe { self.as_ref().align_to::<[T; LANES]>() };
        SimdIter {
            prefix,
            vectors,
            postfix,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::simd::Simd;

    use crate::Vector;

    #[test]
    fn from_slice_padded() {
        assert_eq!(
            [0, 1, 2, 99, 99, 99, 99, 99],
            <Simd::<i32, 8> as Vector>::from_slice_padded(&[0, 1, 2], 99).to_array()
        );
    }
}
