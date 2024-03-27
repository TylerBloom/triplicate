use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// This marker type acts as a compile-time bounds check on the requires of a triplicate, namely
/// that `L > H` and `L <= 64`.
///
/// This type is marked `non_exhaustive` and must be constructed via its const `new` constructor.
/// This constructor checks necessary invariants before providing access to constructor methods for
/// the buffer.
#[non_exhaustive]
pub struct TriplicateBounds<const L: usize = 3, const H: usize = 2>;

impl<const L: usize, const H: usize> TriplicateBounds<L, H> {
    pub const fn new() -> Self {
        if L > 64 {
            panic!("Triplicate can not contain more than 64 elements.");
        }
        if H >= L {
            panic!("Triplicate must have more elements than handles.");
        }
        Self
    }

    /// This method is called at runtime, but we already know that the bound conditions have
    /// already been statisfied because this struct can only be constructed at compile time.
    pub fn construct<T: Default>(self) -> [TriplicateHandle<T, L>; H] {
        self.construct_with(Default::default)
    }

    pub fn construct_with<T, F: FnMut() -> T>(self, f: F) -> [TriplicateHandle<T, L>; H] {
        // SAFETY CHECK:
        // This type can not be constructed without upholding the necessary invariants at compile
        // time; therefore, we can freely construct the buffer.
        let buffer = Arc::new(unsafe { TriplicateInner::<L, T>::new(H, f) });
        std::array::from_fn(|index| TriplicateHandle {
            index,
            buffer: buffer.clone(),
        })
    }

    pub fn construct_with_copies<T: Clone>(self, val: T) -> [TriplicateHandle<T, L>; H] {
        self.construct_with(|| val.clone())
    }
}

/// The main interface into the inner buffer.
pub struct TriplicateHandle<T, const L: usize = 3> {
    index: usize,
    buffer: Arc<TriplicateInner<L, T>>,
}

/// This error is returned during construction of a triplacate buffer whose bounds are checked at
/// runtime instead of compile-time. [TriplicateBounds] can be used to avoid this error during
/// runtime.
pub enum TriplicateConstructionError {
    TooManyElements,
    TooManyHandles,
}

impl<const L: usize, T> TriplicateHandle<T, L> {
    /// Constructs the triplicate buffer and `count` many handles to it.
    ///
    /// SAFETY:
    /// This method is marked as `unsafe` because it constructs the inner buffer. The caller needs
    /// to uphold the same invariants as `TriplicateBuffer::new`.
    unsafe fn construct<F: FnMut() -> T>(count: usize, f: F) -> Vec<Self> {
        let buffer = Arc::new(TriplicateInner::<L, T>::new(count, f));
        let mut index = 0;
        std::iter::from_fn(|| {
            let digest = TriplicateHandle {
                index,
                buffer: buffer.clone(),
            };
            index += 1;
            Some(digest)
        })
        .collect()
    }

    /// Attempts to construct a triplicate buffer by moving to runtime the checks that `TriplicateBounds`
    /// provides at compile-time. This method will create a buffer full of default values.
    pub fn try_construct(count: usize) -> Result<Vec<Self>, TriplicateConstructionError>
    where
        T: Default,
    {
        Self::try_construct_with(count, Default::default)
    }

    /// Attempts to construct a triplicate buffer by moving to runtime the checks that `TriplicateBounds`
    /// provides at compile-time. This method will create a buffer filled with the values returned
    /// from the provided function and in the same order that they are produced.
    pub fn try_construct_with<F: FnMut() -> T>(
        count: usize,
        f: F,
    ) -> Result<Vec<Self>, TriplicateConstructionError> {
        if L > 64 {
            Err(TriplicateConstructionError::TooManyElements)
        } else if count >= L {
            Err(TriplicateConstructionError::TooManyHandles)
        } else {
            // SAFETY CHECK:
            // The above if-elses check the necessary invariants
            Ok(unsafe { Self::construct(count, f) })
        }
    }

    /// Attempts to construct a triplicate buffer by moving to runtime the checks that `TriplicateBounds`
    /// provides at compile-time. This method will create a buffer filled with clones of the given
    /// value.
    pub fn try_construct_with_clone(
        count: usize,
        val: T,
    ) -> Result<Vec<Self>, TriplicateConstructionError>
    where
        T: Clone,
    {
        Self::try_construct_with(count, || val.clone())
    }

    /// Calculates if two handles are associated with the same triplicate buffer.
    pub fn same_buffer(&self, other: &Self) -> bool {
        std::ptr::eq(Arc::as_ptr(&self.buffer), Arc::as_ptr(&other.buffer))
    }

    /// Attempts to move the handle over by one element. If another handle has access to the next element,
    /// this method return `None`; otherwise, a mutable reference to the new item is
    /// returned.
    pub fn try_rotate(&mut self) -> Option<&mut T> {
        if !self.can_rotate() {
            return None;
        }
        let mask = u64::MAX & !(0b1 << self.index);
        self.index = self.next_index();
        let val = 0b1 << self.next_index();
        self.buffer.indices.fetch_or(val, Ordering::Relaxed);
        self.buffer.indices.fetch_add(mask, Ordering::Relaxed);
        Some(&mut *self)
    }

    fn next_index(&self) -> usize {
        if self.index + 1 >= L {
            0
        } else {
            self.index + 1
        }
    }

    /// Checks to see if there is another handle preventing this handle from rotating.
    pub fn can_rotate(&self) -> bool {
        0b1 << self.next_index() & self.buffer.indices.load(Ordering::Relaxed) != 0
    }

    /// Moves the handle over by one element. If another handle has access to the next element,
    /// this method yields until then. Once rotated, a mutable reference to the new item is
    /// returned.
    ///
    /// # Note
    /// If this future is cancelled/dropped, the rotation does not happen. Reads and writes will
    /// affect the original object.
    pub async fn rotate(&mut self) -> &mut T {
        todo!()
    }

    /// Returns a count of the active handles.
    pub fn handle_count(&self) -> u32 {
        self.buffer.indices.load(Ordering::Relaxed).count_ones()
    }

    /// Similar to `Self::create_handle` but will not wait for a spot to become available.
    pub fn try_create_handle(&mut self) -> Result<Self, TryCreateHandleError> {
        if !self.can_rotate() {
            return Err(TryCreateHandleError::SlotInUse);
        }
        if self.handle_count() + 1 == L as u32 {
            return Err(TryCreateHandleError::MaxHandles);
        }
        let digest = Self {
            index: self.next_index(),
            buffer: self.buffer.clone(),
        };
        let val = 0b1 << self.next_index();
        self.buffer.indices.fetch_or(val, Ordering::Relaxed);
        Ok(digest)
    }

    /// Attempts to create a new handle to the triplicate buffer. The new handle will be placed
    /// immediately in front of this handle. Similar to `Self::rotate`, this method yields until
    /// there is an open space.
    ///
    /// The only time this can fail is if the buffer already has its maximum number of handles. In
    /// this case, this method will eagerly fail and will not wait for an empty spot to become
    /// available.
    pub async fn create_handle(&mut self) -> Option<Self> {
        if self.handle_count() + 1 == L as u32 {
            return None;
        }
        todo!()
    }

    /// Swaps the location of the two handles inside the triplicate buffer. This does not affect
    /// the ordering of any other handles.
    pub fn swap_indices(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.index, &mut other.index)
    }
}

/// The error returned when attempting to create a new handle.
pub enum TryCreateHandleError {
    /// The buffer already holds the maximum number of handles into the buffer.
    MaxHandles,
    /// The next slot already has a handle using it.
    SlotInUse,
}

impl<const L: usize, T> Deref for TriplicateHandle<T, L> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY CHECK:
        // On construction, every handle has a different index. This means they "point" to a unique
        // element in the triplicate buffer. A handle can rotate only if there is no handle
        // "pointing" to the next element. This ensure that every handles has a unique index.
        unsafe { &*self.buffer.data[self.index].get() }
    }
}

impl<const L: usize, T> DerefMut for TriplicateHandle<T, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY CHECK:
        // This same safety check that applies to the `Deref` impl also applies here with an
        // additional piece of info.
        //
        // The standard borrowing rules affect shared/exclusive access to the handles. An exclusive
        // reference to the inner data will never be returned from a method that takes a shared
        // reference to the handle. Since this method takes an exclusive reference, it is safe to
        // return an exclusive reference.
        unsafe { &mut *self.buffer.data[self.index].get() }
    }
}

impl<const L: usize, T> Drop for TriplicateHandle<T, L> {
    fn drop(&mut self) {
        let mask = u64::MAX & !(0b1 << self.index);
        // Mask out this handles active index, making it available for other handles to use.
        self.buffer.indices.fetch_and(mask, Ordering::Relaxed);
    }
}

struct TriplicateInner<const L: usize, T> {
    data: [UnsafeCell<T>; L],
    indices: AtomicU64,
}

impl<const L: usize, T> TriplicateInner<L, T> {
    /// SAFETY:
    /// This method is marked as `unsafe` because it does not check that `L` or `count` uphold the
    /// required safety invariants. The caller must ensure:
    ///  - `L` is at most 64
    ///  - `count` is strictly less than `L`
    unsafe fn new<F: FnMut() -> T>(count: usize, mut f: F) -> Self {
        let data = std::array::from_fn(|_| UnsafeCell::new(f()));
        let mut indices = u64::MAX;
        for i in count..64 {
            indices ^= 0b1 << i;
        }
        // TODO: Remove this after testing.
        assert_eq!(indices.count_ones(), count as u32);
        let indices = AtomicU64::new(indices);
        Self { data, indices }
    }
}

// TODO: Need to do extra verification around this...
unsafe impl<const L: usize, T> Sync for TriplicateInner<L, T> where T: Send + Sync {}

#[cfg(test)]
mod tests {
    use crate::{TriplicateBounds, TriplicateHandle};

    fn is_send<T: Send>(_: &T) {}

    fn is_sync<T: Sync>(_: &T) {}

    #[test]
    fn send_sync() {
        let [h1, _h2]: [TriplicateHandle<_, 3>; 2] = TriplicateBounds::new().construct::<Vec<u8>>();
        is_send(&h1);
        is_sync(&h1);
    }
}
