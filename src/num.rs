use core::simd::{LaneCount, Simd, SimdElement, SimdFloat, SimdInt, SimdUint, SupportedLaneCount};

use num_traits::{Num, NumOps};

/// A SIMD vector with numeric elements (i.e. all of them).
pub trait SimdNum: NumOps + Sized {
    type Scalar: SimdElement + Num;

    fn reduce_sum(self) -> Self::Scalar;
    fn reduce_product(self) -> Self::Scalar;
}

macro_rules! impl_simd_num {
    ($elem:ty, $trait_:ty) => {
        impl<const LANES: usize> SimdNum for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Scalar = $elem;

            fn reduce_sum(self) -> $elem {
                <Self as $trait_>::reduce_sum(self)
            }

            fn reduce_product(self) -> $elem {
                <Self as $trait_>::reduce_product(self)
            }
        }
    };
}

impl_simd_num!(f32, SimdFloat);
impl_simd_num!(f64, SimdFloat);
impl_simd_num!(i8, SimdInt);
impl_simd_num!(i16, SimdInt);
impl_simd_num!(i32, SimdInt);
impl_simd_num!(i64, SimdInt);
impl_simd_num!(isize, SimdInt);
impl_simd_num!(u8, SimdUint);
impl_simd_num!(u16, SimdUint);
impl_simd_num!(u32, SimdUint);
impl_simd_num!(u64, SimdUint);
impl_simd_num!(usize, SimdUint);

/// An extension trait for `Iterator`s over `SimdNum`s.
pub trait SimdNumIterExt {
    type Scalar;

    /// Returns the sum of all the scalars in the iterator.
    fn scalar_sum(self) -> Self::Scalar;

    /// Returns the product of all the scalars in the iterator.
    fn scalar_product(self) -> Self::Scalar;
}

impl<I, T, const LANES: usize> SimdNumIterExt for I
    where
        I: Iterator<Item=Simd<T, LANES>>,
        T: SimdElement + Num,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SimdNum<Scalar=T>,
{
    type Scalar = T;

    fn scalar_sum(self) -> T {
        self.reduce(core::ops::Add::add)
            .map(SimdNum::reduce_sum)
            .unwrap_or_else(T::zero)
    }

    fn scalar_product(self) -> T {
        self.reduce(core::ops::Mul::mul)
            .map(SimdNum::reduce_product)
            .unwrap_or_else(T::one)
    }
}
