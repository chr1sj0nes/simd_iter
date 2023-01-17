#![no_std]
#![feature(portable_simd)]

use core::simd::{LaneCount, Mask, Simd, SimdElement, SimdPartialOrd, SupportedLaneCount};

use num_traits::{NumCast, One, Zero};

pub use crate::integer::SimdIntegerIterExt;
use crate::min_max_identities::MinMaxIdentities;
pub use crate::num::SimdNumIterExt;
pub use crate::ord::SimdOrdIterExt;

mod integer;
mod min_max_identities;
mod num;
mod ord;

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
    vectors: &'a [Simd<T, LANES>],
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

    /// Returns the sum of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(15., [1., 2., 3., 4., 5.].simd_iter().scalar_sum());
    /// ```
    pub fn scalar_sum(self) -> T
    where
        T: Zero,
        SimdIterPadded<'a, T, LANES>: SimdNumIterExt<Scalar = T>,
    {
        self.padded_with(T::zero()).scalar_sum()
    }

    /// Returns the product of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(120., [1., 2., 3., 4., 5.].simd_iter().scalar_product());
    /// ```
    pub fn scalar_product(self) -> T
    where
        T: One,
        SimdIterPadded<'a, T, LANES>: SimdNumIterExt<Scalar = T>,
    {
        self.padded_with(T::one()).scalar_product()
    }

    /// Returns the min of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(Some(-7), [-1, 1, -2, 3, -7, 5].simd_iter().scalar_min());
    /// ```
    pub fn scalar_min(self) -> Option<T>
    where
        T: MinMaxIdentities,
        SimdIterPadded<'a, T, LANES>: SimdOrdIterExt<Scalar = T>,
    {
        self.padded_with(T::min_identity()).scalar_min()
    }

    /// Returns the max of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(Some(5), [-1, 1, -2, 3, -7, 5].simd_iter().scalar_max());
    /// ```
    pub fn scalar_max(self) -> Option<T>
    where
        T: MinMaxIdentities,
        SimdIterPadded<'a, T, LANES>: SimdOrdIterExt<Scalar = T>,
    {
        self.padded_with(T::max_identity()).scalar_max()
    }

    /// Returns the bit-wise AND (`&`) reduction of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(Some(0b100), [0b111, 0b110, 0b101].simd_iter().scalar_reduce_and());
    /// ```
    pub fn scalar_reduce_and(self) -> Option<T>
    where
        T: Zero + core::ops::Not<Output = T>,
        SimdIterPadded<'a, T, LANES>: SimdIntegerIterExt<Scalar = T>,
    {
        self.padded_with(!T::zero()).scalar_reduce_and()
    }

    /// Returns the bit-wise OR (`|`) reduction of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(Some(0b110), [0b000, 0b110, 0b100].simd_iter().scalar_reduce_or());
    /// ```
    pub fn scalar_reduce_or(self) -> Option<T>
    where
        T: Zero,
        SimdIterPadded<'a, T, LANES>: SimdIntegerIterExt<Scalar = T>,
    {
        self.padded_with(T::zero()).scalar_reduce_or()
    }

    /// Returns the bit-wise XOR (`^`) reduction of all the scalars in the iterator, including the prefix and postfix.
    ///
    /// ```
    /// use simd_iter::SimdIterable;
    /// assert_eq!(Some(0b100), [0b111, 0b110, 0b101].simd_iter().scalar_reduce_xor());
    /// ```
    pub fn scalar_reduce_xor(self) -> Option<T>
    where
        T: Zero,
        SimdIterPadded<'a, T, LANES>: SimdIntegerIterExt<Scalar = T>,
    {
        self.padded_with(T::zero()).scalar_reduce_xor()
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
            Some(*first)
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
    T::Mask: NumCast,
    Simd<T::Mask, LANES>: SimdPartialOrd<Mask = Mask<T::Mask, LANES>>,
{
    type Item = Simd<T, LANES>;

    fn next(&mut self) -> Option<Self::Item> {
        let pad_value = self.pad_value;
        let try_take_slice_padded = |values: &mut &[T]| {
            if values.is_empty() {
                None
            } else {
                let iota = Simd::from_array(core::array::from_fn(|i| {
                    <T::Mask as NumCast>::from(i).unwrap()
                }));
                let mask = iota.simd_lt(Simd::splat(
                    <T::Mask as NumCast>::from(values.len()).unwrap(),
                ));
                // SAFETY: We are reading beyond the end of the slice, but we're going to mask it out.
                let vec = Self::Item::from_slice(unsafe {
                    core::slice::from_raw_parts(values.as_ptr(), LANES)
                });
                *values = &[];
                Some(mask.select(vec, Self::Item::splat(pad_value)))
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
    Self: Iterator,
{
}

impl<T: SimdElement, U: AsRef<[T]>> SimdIterable<T> for U {
    fn simd_iter_with_width<const LANES: usize>(&self) -> SimdIter<T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let (prefix, vectors, postfix) = self.as_ref().as_simd();
        SimdIter {
            prefix,
            vectors,
            postfix,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use crate::SimdIterable;

    proptest! {
        #[test]
        fn test_scalar_sum(xs in prop::collection::vec(0.0..1.0f32, 0..100)) {
            assert_relative_eq!(xs.iter().sum::<f32>(), xs.simd_iter().scalar_sum(), max_relative = 0.00001);
        }

        #[test]
        fn test_scalar_product(xs in prop::collection::vec(0.0..1.0, 0..100)) {
            assert_relative_eq!(xs.iter().product::<f64>(), xs.simd_iter().scalar_product());
        }

        #[test]
        fn test_scalar_min(xs in prop::collection::vec(any::<u32>(), 0..100)) {
            assert_eq!(xs.iter().cloned().min(), xs.simd_iter().scalar_min());
        }

        #[test]
        fn test_scalar_max(xs in prop::collection::vec(any::<i64>(), 0..100)) {
            assert_eq!(xs.iter().cloned().max(), xs.simd_iter().scalar_max());
        }

        #[test]
        fn test_scalar_reduce_and(xs in prop::collection::vec(any::<i8>(), 0..100)) {
            assert_eq!(xs.iter().cloned().reduce(core::ops::BitAnd::bitand), xs.simd_iter().scalar_reduce_and());
        }

        #[test]
        fn test_scalar_reduce_or(xs in prop::collection::vec(any::<i16>(), 0..100)) {
            assert_eq!(xs.iter().cloned().reduce(core::ops::BitOr::bitor), xs.simd_iter().scalar_reduce_or());
        }


        #[test]
        fn test_scalar_reduce_xor(xs in prop::collection::vec(any::<i32>(), 0..1000)) {
            assert_eq!(xs.iter().cloned().reduce(core::ops::BitXor::bitxor), xs.simd_iter().scalar_reduce_xor());
        }
    }
}
