use core::simd::{
    LaneCount, Simd, SimdElement, SimdFloat, SimdInt, SimdOrd, SimdUint, SupportedLaneCount,
};

/// A SIMD vector with `Ord` / `Float` elements.
///
/// While float types do not implement `Ord`, due to NaN weirdness,
/// it can be useful to simply ignore that sometimes!
pub trait SimdOrdOrFloat {
    type Scalar: SimdElement;

    fn simd_min(self, other: Self) -> Self;
    fn simd_max(self, other: Self) -> Self;
    fn reduce_min(self) -> Self::Scalar;
    fn reduce_max(self) -> Self::Scalar;
}

macro_rules! impl_simd_ord_or_float {
    ($elem:ty, $ord_trait:ty, $reduce_trait:ty) => {
        impl<const LANES: usize> SimdOrdOrFloat for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Scalar = $elem;

            fn simd_min(self, other: Self) -> Self {
                <Self as $ord_trait>::simd_min(self, other)
            }

            fn simd_max(self, other: Self) -> Self {
                <Self as $ord_trait>::simd_max(self, other)
            }

            fn reduce_min(self) -> $elem {
                <Self as $reduce_trait>::reduce_min(self)
            }

            fn reduce_max(self) -> $elem {
                <Self as $reduce_trait>::reduce_max(self)
            }
        }
    };
}

impl_simd_ord_or_float!(f32, SimdFloat, SimdFloat);
impl_simd_ord_or_float!(f64, SimdFloat, SimdFloat);
impl_simd_ord_or_float!(i8, SimdOrd, SimdInt);
impl_simd_ord_or_float!(i16, SimdOrd, SimdInt);
impl_simd_ord_or_float!(i32, SimdOrd, SimdInt);
impl_simd_ord_or_float!(i64, SimdOrd, SimdInt);
impl_simd_ord_or_float!(isize, SimdOrd, SimdInt);
impl_simd_ord_or_float!(u8, SimdOrd, SimdUint);
impl_simd_ord_or_float!(u16, SimdOrd, SimdUint);
impl_simd_ord_or_float!(u32, SimdOrd, SimdUint);
impl_simd_ord_or_float!(u64, SimdOrd, SimdUint);
impl_simd_ord_or_float!(usize, SimdOrd, SimdUint);

/// An extension trait for `Iterator`s over `SimdOrdOrFloat`s.
pub trait SimdOrdIterExt {
    type Scalar;

    /// Returns the min of all the scalars in the iterator.
    fn scalar_min(self) -> Option<Self::Scalar>;

    /// Returns the max of all the scalars in the iterator.
    fn scalar_max(self) -> Option<Self::Scalar>;
}

impl<I, T: SimdElement, const LANES: usize> SimdOrdIterExt for I
where
    I: Iterator<Item = Simd<T, LANES>>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SimdOrdOrFloat<Scalar = T>,
{
    type Scalar = T;

    fn scalar_min(self) -> Option<T> {
        self.reduce(SimdOrdOrFloat::simd_min)
            .map(SimdOrdOrFloat::reduce_min)
    }

    fn scalar_max(self) -> Option<T> {
        self.reduce(SimdOrdOrFloat::simd_max)
            .map(SimdOrdOrFloat::reduce_max)
    }
}
