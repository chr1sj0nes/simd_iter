pub trait MinMaxIdentities {
    fn min_identity() -> Self;
    fn max_identity() -> Self;
}

macro_rules! impl_int_min_max_identities {
    ($int:ty) => {
        impl MinMaxIdentities for $int {
            fn min_identity() -> $int {
                <$int>::MAX
            }

            fn max_identity() -> $int {
                <$int>::MIN
            }
        }
    };
}

impl_int_min_max_identities!(i8);
impl_int_min_max_identities!(i16);
impl_int_min_max_identities!(i32);
impl_int_min_max_identities!(i64);
impl_int_min_max_identities!(isize);
impl_int_min_max_identities!(u8);
impl_int_min_max_identities!(u16);
impl_int_min_max_identities!(u32);
impl_int_min_max_identities!(u64);
impl_int_min_max_identities!(usize);

macro_rules! impl_float_min_max_identities {
    ($float:ty) => {
        impl MinMaxIdentities for $float {
            fn min_identity() -> $float {
                <$float>::INFINITY
            }

            fn max_identity() -> $float {
                <$float>::NEG_INFINITY
            }
        }
    };
}

impl_float_min_max_identities!(f32);
impl_float_min_max_identities!(f64);
