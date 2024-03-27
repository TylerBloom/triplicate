#![allow(dead_code, unused)]
use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::{atomic::{AtomicU64, Ordering}, Arc},
};

/// This marker type acts as a compile-time bounds check on the requires of a triplicate, namely
/// that `L > H` and `L <= 64`.
pub struct TriplicateBounds<const L: usize, const H: usize>;

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
    pub fn construct<T: Default>(self) -> [TriplicateHandle<L, T>; L] {
        self.construct_with(Default::default)
    }

    pub fn construct_with<T, F: FnMut() -> T>(self, f: F) -> [TriplicateHandle<L, T>; L] {
        let inner = Arc::new(TriplicateInner::<L, T>::new(H, f));
        todo!()
    }

    pub fn construct_with_copies<T: Clone>(self, val: T) -> [TriplicateHandle<L, T>; L] {
        self.construct_with(|| val.clone())
    }
}

/// The main interface into the inner buffer.
pub struct TriplicateHandle<const L: usize, T> {
    index: usize,
    buffer: Arc<TriplicateInner<L, T>>,
}

impl<const L: usize, T> TriplicateHandle<L, T> {
    /// Attempts to move the handle over by one element. If another handle has access to the next element,
    /// this method return `None`; otherwise, a mutable reference to the new item is
    /// returned.
    pub fn try_rotate(&mut self) -> Option<&mut T> {
        todo!()
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
    async fn rotate(&mut self) -> &mut T {
        todo!()
    }

    /// Returns a count of the active handles.
    pub fn handle_count(&self) -> u32 {
        self.buffer.indices.load(Ordering::Relaxed).count_ones()
    }

    /// Similar to `Self::create_handle` but will not wait for a spot to become available.
    pub fn try_create_handle(&mut self) -> Result<Self, TryCreateHandleError> {
        todo!()
    }

    /// Attempts to create a new handle to the triplicate buffer. The new handle will be placed
    /// immediately in front of this handle. Similar to `Self::rotate`, this method yields until
    /// there is an open space.
    ///
    /// The only time this can fail is if the buffer already has its maximum number of handles. In
    /// this case, this method will eagerly fail and will not wait for an empty spot to become
    /// available.
    pub async fn create_handle(&mut self) -> Option<Self> {
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

impl<const L: usize, T> Deref for TriplicateHandle<L, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY CHECK:
        // On construction, every handle has a different index. This means they "point" to a unique
        // element in the triplicate buffer. A handle can rotate only if there is no handle
        // "pointing" to the next element. This ensure that every handles has a unique index.
        unsafe { &*self.buffer.data[self.index].get() }
    }
}

impl<const L: usize, T> DerefMut for TriplicateHandle<L, T> {
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

impl<const L: usize, T> Drop for TriplicateHandle<L, T> {
    // This holds the handles active index. That bit needs to be cleared on drop.
    fn drop(&mut self) {
        todo!()
    }
}

struct TriplicateInner<const L: usize, T> {
    data: [UnsafeCell<T>; L],
    indices: AtomicU64,
}

impl<const L: usize, T> TriplicateInner<L, T> {
    // count is the number of handles.
    fn new<F: FnMut() -> T>(count: usize, mut f: F) -> Self {
        // TODO: Surely there is a better way to do this...
        let data = [(); L].map(|()| UnsafeCell::new(f()));
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
