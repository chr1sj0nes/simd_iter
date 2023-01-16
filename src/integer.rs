use core::simd::{LaneCount, Simd, SimdElement, SimdInt, SimdUint, SupportedLaneCount};

use num_traits::PrimInt;

/// A SIMD vector with integer elements (i.e. `SimdInt` or `SimdUint`).
pub trait SimdInteger:
core::ops::BitAnd<Output=Self>
+ core::ops::BitOr<Output=Self>
+ core::ops::BitXor<Output=Self>
+ Sized
{
    type Scalar: SimdElement + PrimInt;

    fn reduce_and(self) -> Self::Scalar;
    fn reduce_or(self) -> Self::Scalar;
    fn reduce_xor(self) -> Self::Scalar;
}

macro_rules! impl_simd_integer {
    ($elem:ty, $trait_:ty) => {
        impl<const LANES: usize> SimdInteger for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Scalar = $elem;

            fn reduce_and(self) -> $elem {
                <Self as $trait_>::reduce_and(self)
            }

            fn reduce_or(self) -> $elem {
                <Self as $trait_>::reduce_or(self)
            }

            fn reduce_xor(self) -> $elem {
                <Self as $trait_>::reduce_xor(self)
            }
        }
    };
}

impl_simd_integer!(i8, SimdInt);
impl_simd_integer!(i16, SimdInt);
impl_simd_integer!(i32, SimdInt);
impl_simd_integer!(i64, SimdInt);
impl_simd_integer!(isize, SimdInt);
impl_simd_integer!(u8, SimdUint);
impl_simd_integer!(u16, SimdUint);
impl_simd_integer!(u32, SimdUint);
impl_simd_integer!(u64, SimdUint);
impl_simd_integer!(usize, SimdUint);

/// An extension trait for `Iterator`s over `SimdInteger`s.
pub trait SimdIntegerIterExt {
    type Scalar;

    /// Returns the bit-wise AND (`&`) reduction of all the scalars in the iterator.
    fn scalar_reduce_and(self) -> Option<Self::Scalar>;

    /// Returns the bit-wise OR (`|`) reduction of all the scalars in the iterator.
    fn scalar_reduce_or(self) -> Option<Self::Scalar>;

    /// Returns the bit-wise XOR (`^`) reduction of all the scalars in the iterator.
    fn scalar_reduce_xor(self) -> Option<Self::Scalar>;
}

impl<I, T, const LANES: usize> SimdIntegerIterExt for I
    where
        I: Iterator<Item=Simd<T, LANES>>,
        T: SimdElement + PrimInt,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SimdInteger<Scalar=T>,
{
    type Scalar = T;

    fn scalar_reduce_and(self) -> Option<T> {
        self.reduce(core::ops::BitAnd::bitand)
            .map(SimdInteger::reduce_and)
    }

    fn scalar_reduce_or(self) -> Option<T> {
        self.reduce(core::ops::BitOr::bitor)
            .map(SimdInteger::reduce_or)
    }

    fn scalar_reduce_xor(self) -> Option<T> {
        self.reduce(core::ops::BitXor::bitxor)
            .map(SimdInteger::reduce_xor)
    }
}
