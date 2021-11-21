use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

use num_traits::{Num, One, PrimInt, Zero};

use crate::scalar::MinMaxIdentities;
use crate::vector::{IntVector, NumVector, Vector};
use crate::SimdIter;

pub trait NumVectorReductions {
    type Scalar;

    /// Returns the sum of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{NumVectorReductions, SimdIterable};
    /// assert_eq!(15., [1., 2., 3., 4., 5.].simd_iter().scalar_sum());
    /// ```
    fn scalar_sum(self) -> Self::Scalar;

    /// Returns the product of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{NumVectorReductions, SimdIterable};
    /// assert_eq!(120., [1., 2., 3., 4., 5.].simd_iter().scalar_product());
    /// ```
    fn scalar_product(self) -> Self::Scalar;

    /// Returns the min of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{NumVectorReductions, SimdIterable};
    /// assert_eq!(Some(-7), [-1, 1, -2, 3, -7, 5].simd_iter().scalar_min());
    /// ```
    fn scalar_min(self) -> Option<Self::Scalar>;

    /// Returns the max of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{NumVectorReductions, SimdIterable};
    /// assert_eq!(Some(5), [-1, 1, -2, 3, -7, 5].simd_iter().scalar_max());
    /// ```
    fn scalar_max(self) -> Option<Self::Scalar>;
}

impl<I, V: NumVector> NumVectorReductions for I
where
    I: Iterator<Item = V>,
    V::Element: Num + MinMaxIdentities,
{
    type Scalar = V::Element;

    default fn scalar_sum(self) -> V::Element {
        self.reduce(core::ops::Add::add)
            .map(V::horizontal_sum)
            .unwrap_or(<V::Element as Zero>::zero())
    }

    default fn scalar_product(self) -> V::Element {
        self.reduce(core::ops::Mul::mul)
            .map(V::horizontal_product)
            .unwrap_or(<V::Element as One>::one())
    }

    default fn scalar_min(self) -> Option<V::Element> {
        self.reduce(V::min).map(V::horizontal_min)
    }

    default fn scalar_max(self) -> Option<V::Element> {
        self.reduce(V::max).map(V::horizontal_max)
    }
}

impl<'a, T, const LANES: usize> NumVectorReductions for SimdIter<'a, T, LANES>
where
    T: SimdElement + Num + MinMaxIdentities,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Vector<Element = T> + NumVector,
{
    /// Returns the sum of all the scalars in the iterator, including the prefix and postfix.
    fn scalar_sum(self) -> T {
        self.padded_with(T::zero()).scalar_sum()
    }

    /// Returns the product of all the scalars in the iterator, including the prefix and postfix.
    fn scalar_product(self) -> T {
        self.padded_with(T::one()).scalar_product()
    }

    /// Returns the min of all the scalars in the iterator, including the prefix and postfix.
    fn scalar_min(self) -> Option<T> {
        self.padded_with(T::min_identity()).scalar_min()
    }

    /// Returns the max of all the scalars in the iterator, including the prefix and postfix.
    fn scalar_max(self) -> Option<T> {
        self.padded_with(T::max_identity()).scalar_max()
    }
}

pub trait IntVectorReductions {
    type Scalar;

    /// Returns the bit-wise AND (`&`) reduction of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{IntVectorReductions, SimdIterable};
    /// assert_eq!(Some(0b100), [0b111, 0b110, 0b101].simd_iter().scalar_reduce_and());
    /// ```
    fn scalar_reduce_and(self) -> Option<Self::Scalar>;

    /// Returns the bit-wise OR (`|`) reduction of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{IntVectorReductions, SimdIterable};
    /// assert_eq!(Some(0b110), [0b000, 0b110, 0b100].simd_iter().scalar_reduce_or());
    /// ```
    fn scalar_reduce_or(self) -> Option<Self::Scalar>;

    /// Returns the bit-wise XOR (`^`) reduction of all the scalars in the iterator.
    ///
    /// ```
    /// use simd_iter::{IntVectorReductions, SimdIterable};
    /// assert_eq!(Some(0b100), [0b111, 0b110, 0b101].simd_iter().scalar_reduce_xor());
    /// ```
    fn scalar_reduce_xor(self) -> Option<Self::Scalar>;
}

impl<I, V: IntVector> IntVectorReductions for I
where
    I: Iterator<Item = V>,
    V::Element: PrimInt,
{
    type Scalar = V::Element;

    default fn scalar_reduce_and(self) -> Option<Self::Scalar> {
        self.reduce(core::ops::BitAnd::bitand)
            .map(V::horizontal_and)
    }

    default fn scalar_reduce_or(self) -> Option<Self::Scalar> {
        self.reduce(core::ops::BitOr::bitor).map(V::horizontal_or)
    }

    default fn scalar_reduce_xor(self) -> Option<Self::Scalar> {
        self.reduce(core::ops::BitXor::bitxor)
            .map(V::horizontal_xor)
    }
}

impl<'a, T, const LANES: usize> IntVectorReductions for SimdIter<'a, T, LANES>
where
    T: SimdElement + PrimInt,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Vector<Element = T> + IntVector,
{
    /// Returns the bit-wise AND (`&`) reduction of all the scalars in the iterator,
    /// including the prefix and postfix.
    fn scalar_reduce_and(self) -> Option<Self::Scalar> {
        self.padded_with(!T::zero()).scalar_reduce_and()
    }

    /// Returns the bit-wise OR (`|`) reduction of all the scalars in the iterator,
    /// including the prefix and postfix.
    fn scalar_reduce_or(self) -> Option<Self::Scalar> {
        self.padded_with(T::zero()).scalar_reduce_or()
    }

    /// Returns the bit-wise XOR (`^`) reduction of all the scalars in the iterator,
    /// including the prefix and postfix.
    fn scalar_reduce_xor(self) -> Option<Self::Scalar> {
        self.padded_with(T::zero()).scalar_reduce_xor()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use crate::{IntVectorReductions, NumVectorReductions, SimdIterable};

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
