use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

use delegate::delegate;
use num_traits::{Num, PrimInt};

pub trait Vector {
    type Element: SimdElement;
    fn from_slice_padded(values: &[Self::Element], pad_value: Self::Element) -> Self;
}

macro_rules! lane_indices {
    ($idx:ty, $lanes:ident) => {{
        let mut indices = [0 as $idx; $lanes];
        let mut i = 0;
        while i < $lanes {
            indices[i] = i as $idx;
            i += 1;
        }
        Simd::<$idx, $lanes>::from_array(indices)
    }};
}

macro_rules! impl_vector {
    ($elem:ty) => {
        impl<const LANES: usize> Vector for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Element = $elem;

            fn from_slice_padded(values: &[$elem], pad_value: $elem) -> Self {
                // `LaneIndex` could be any integer type with the same width as `Element` and
                // large enough to represent `LANES`. `Mask` always satisfies this.
                type LaneIndex = <$elem as SimdElement>::Mask;
                let mask = lane_indices!(LaneIndex, LANES)
                    .lanes_lt(Simd::splat(values.len() as LaneIndex));
                // SAFETY: We are reading beyond the end of the slice, but we're going to mask it out.
                let values = Self::from_slice(unsafe {
                    core::slice::from_raw_parts(values.as_ptr(), LANES)
                });
                mask.select(values, Self::splat(pad_value))
            }
        }
    };
}

pub trait NumVector:
    Vector + core::ops::Add<Output = Self> + core::ops::Mul<Output = Self> + Sized
where
    <Self as Vector>::Element: Num,
{
    fn horizontal_sum(self) -> <Self as Vector>::Element;
    fn horizontal_product(self) -> <Self as Vector>::Element;
    fn horizontal_min(self) -> <Self as Vector>::Element;
    fn horizontal_max(self) -> <Self as Vector>::Element;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_num_vector {
    ($elem:ty) => {
        impl_vector!($elem);

        impl<const LANES: usize> NumVector for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            delegate! {
                to self {
                    fn horizontal_sum(self) -> $elem;
                    fn horizontal_product(self) -> $elem;
                    fn horizontal_min(self) -> $elem;
                    fn horizontal_max(self) -> $elem;
                }
            }

            fn min(self, other: Self) -> Self {
                self.lanes_lt(other).select(self, other)
            }

            fn max(self, other: Self) -> Self {
                self.lanes_gt(other).select(self, other)
            }
        }
    };
}

impl_num_vector!(f32);
impl_num_vector!(f64);

pub trait IntVector:
    NumVector
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::BitXor<Output = Self>
where
    <Self as Vector>::Element: PrimInt,
{
    fn horizontal_and(self) -> <Self as Vector>::Element;
    fn horizontal_or(self) -> <Self as Vector>::Element;
    fn horizontal_xor(self) -> <Self as Vector>::Element;
}

macro_rules! impl_int_vector {
    ($elem:ty) => {
        impl_num_vector!($elem);

        impl<const LANES: usize> IntVector for Simd<$elem, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            delegate! {
                to self {
                    fn horizontal_and(self) -> $elem;
                    fn horizontal_or(self) -> $elem;
                    fn horizontal_xor(self) -> $elem;
                }
            }
        }
    };
}

impl_int_vector!(i8);
impl_int_vector!(i16);
impl_int_vector!(i32);
impl_int_vector!(i64);
impl_int_vector!(isize);
impl_int_vector!(u8);
impl_int_vector!(u16);
impl_int_vector!(u32);
impl_int_vector!(u64);
impl_int_vector!(usize);
