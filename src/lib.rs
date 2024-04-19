//! ## Overview
//! This crate implements a simple, lock-free (but not wait-free) concurrency primitive. This
//! primitive consists of a buffer of N elements and M handles to this buffer. Every handle has an
//! index into the buffer that is gaurnteed to be unique from the other handles. Which elements
//! have indices to them is tracked atomically. This provides ensures that every handle has
//! exclusive access to its element.
//!
//! Note, changes between buffer elements are not transmitted in any way. Rather, handles
//! independently "rotate" around the buffer. Allowing handles to passively communicate changes.
//!
//! This primitive was developed with the primary usecase of a three-element buffer of byte arrays
//! with two handles. One handle performs writes into a given byte array, releases the changes to
//! the reader who releases items after performing any necessary operations. This is the source of
//! the name "triplicate" (a document that has three copies made), but the idea has been
//! generalized beyond this single usecase.
//!
//! A diagram of a buffer in-use might look something like this:
//! > Buffer : [ e0, e1, e2, e3, e4 ]
//! > Handles: [ h0, h1,  _, h2,  _ ]
//! Here, there are handles to the first, second, and fourth elements in the buffer. Each handle is
//! able to make arbitary reads and writes to its element. In order for another handle to gain
//! access to those changes, the handle must reliquish its access to that element. This can happen
//! by dropping the handle, or, more commonly, rotating the handle's index. In this example,
//! both `h1` and `h2` can add one to their index in order to gain access to the next item and
//! release access to their current item. Note that the first handle can not increment its index as
//! there is a handle already there.
//!
//! Let's say both `h1` and `h2` rotate.
//! Note that, rotations happen independently for each handle and that one handle can not force another to rotate.
//! The state of the buffer would now looks like this:
//! > Buffer : [ e0, e1, e2, e3, e4 ]
//! > Handles: [ h0,  _, h1,  _, h2 ]
//! After those rotations occur, `h0` can now rotate and `h1` can rotate yet again. Should `h0`
//! rotate, `h2` could then rotate. This would put `h2` back at the front of the buffer. Once
//! allocated, the buffer's capacity is fixed.
//!
//! At this point, the state of the buffer looks like this:
//! > Buffer : [ e0, e1, e2, e3, e4 ]
//! > Handles: [ h2, h0, h1,  _,  _ ]
//! Note that handles can only move in a single direction. It is possible for two handles to swap
//! indices, but it is not possible to reverse the direction that handles move. Handles will always
//! increase this index (modulo the number of elements).
//!
//! ## Restrictions
//! There are two primary invariants that the buffer must maintain:
//!  - There can not be more than 64 elements
//!  - The number of handles must be strictly less than the number of elements in the buffer.
//!
//! The first requirement is a result of how handle locations are tracked.
//! The second requirement is straightforward. If there were an equal number of handles and
//! elements, no handle could rotate.
//! To uphold these bounds, the triplicate buffer uses compile-time assertions.
//! These are provided through the `TriplicateBounds` type.
//! There is also a runtime-checked API available directly on the `TriplicateHandle` type.

use std::{
    cell::UnsafeCell,
    future::Future,
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll, Waker},
};

/// This marker type acts as a compile-time bounds check on the requires of a triplicate, namely
/// that `L > H` and `L <= 64`.
///
/// This type is marked `non_exhaustive` and must be constructed via its const `new` constructor.
/// This constructor checks necessary invariants before providing access to constructor methods for
/// the buffer.
#[non_exhaustive]
pub struct TriplicateBounds<const L: usize = 3, const H: usize = 2>;

// TODO: Document the reasoning behind these goofy consts and const fns

/// Compile-time bounds checking is a bit wonky in Rust (when the invariants aren't easily encoded
/// in the type system), but it is possible. This ZST is part of that checking. See the docs on
/// `TriplicateBounds::new` for why you might want this. These docs will focus on the how.
///
/// Since [inline-consts](https://rust-lang.github.io/rfcs/2920-inline-const.html) are still
/// unstable, we can not force a `const fn` to be ran at compile time, but that's exactly what we
/// want here. To get around that, we can have the `const fn new` method use a const on
/// `TriplicateBounds`. The usage of this const, even if trivial, forces the check to happen at
/// compile-time.
///
/// This type is private and is only returned by the `bounds_check` `const fn`, which will always
/// return this type (or panic). To ensure that this type is only every constructed at
/// compile-time, `bounds_check` is used to define an associated constant for `TriplicateBounds`.
/// This constant is then used during the `new` method, where it is (trivially) destructured and
/// the bounds object is returned.
struct ValidBounds;

/// This method provides the bounds checks for a triplicate but panics if they are not valid. This
/// method is only meant to be ran at compile time in order to sidestep runtime bounds checking.
/// Runtime bounds checking is available directly on the `TriplicateHandle`. This is only used for
/// the `TriplicateBounds`.
const fn bounds_check<const L: usize, const H: usize>() -> ValidBounds {
    if L > 64 {
        panic!("Triplicate can not contain more than 64 elements.")
    } else if H >= L {
        panic!("Triplicate must have more elements than handles.")
    } else {
        ValidBounds
    }
}

impl<const L: usize, const H: usize> TriplicateBounds<L, H> {
    const CHECK: ValidBounds = bounds_check::<L, H>();

    // TODO: Document this method and explain why the
    pub const fn new() -> Self {
        match Self::CHECK {
            ValidBounds => Self,
        }
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
        let buffer = Arc::new(unsafe { TriplicateInner::<T, L>::new(H, f) });
        std::array::from_fn(|index| TriplicateHandle {
            index,
            buffer: buffer.clone(),
        })
    }

    pub fn construct_with_copies<T: Clone>(self, val: T) -> [TriplicateHandle<T, L>; H] {
        self.construct_with(|| val.clone())
    }
}

impl<const L: usize, const H: usize> Default for TriplicateBounds<L, H> {
    fn default() -> Self {
        Self::new()
    }
}

/// The main interface into the inner buffer.
pub struct TriplicateHandle<T, const L: usize = 3> {
    index: usize,
    buffer: Arc<TriplicateInner<T, L>>,
}

pub struct RotationFailure;

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
        let buffer = Arc::new(TriplicateInner::<T, L>::new(count, f));
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
    pub fn try_rotate(&mut self) -> Result<&mut T, RotationFailure> {
        if !self.can_rotate() {
            return Err(RotationFailure);
        }
        let mask = u64::MAX & !(0b1 << self.index);
        self.index = self.next_index();
        let val = 0b1 << self.next_index();
        self.buffer.indices.fetch_or(val, Ordering::AcqRel);
        self.buffer.indices.fetch_add(mask, Ordering::AcqRel);
        Ok(&mut *self)
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
        0b1 << self.next_index() & self.buffer.indices.load(Ordering::Acquire) != 0
    }

    /// Moves the handle over by one element. If another handle has access to the next element,
    /// this method yields until then. Once rotated, a mutable reference to the new item is
    /// returned.
    ///
    /// # Note
    /// If this future is cancelled/dropped, the rotation does not happen. Reads and writes will
    /// affect the original object.
    pub async fn rotate(&mut self) -> &mut T {
        let fut = RotationFut {
            waker: None,
            handle: self,
        };
        fut.await;
        &mut *self
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
        self.buffer.indices.fetch_or(val, Ordering::AcqRel);
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
        let fut = CreationFut {
            waker: None,
            handle: self,
        };
        Some(fut.await)
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
        self.buffer.indices.fetch_and(mask, Ordering::AcqRel);
    }
}

// TODO: Docs
#[must_use = "The rotate method returns a future that must be `await`ed. Dropping this will leave the handle unrotated"]
struct RotationFut<'a, T, const L: usize> {
    waker: Option<Waker>,
    handle: &'a mut TriplicateHandle<T, L>,
}

impl<'a, T, const L: usize> Future for RotationFut<'a, T, L> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.handle.try_rotate() {
            Ok(_) => Poll::Ready(()),
            Err(_) => {
                match this.waker.as_mut() {
                    None => this.waker = Some(cx.waker().clone()),
                    Some(waker) => waker.clone_from(cx.waker()),
                }
                Poll::Pending
            }
        }
    }
}

// TODO: Docs
#[must_use = "The rotate method returns a future that must be `await`ed. Dropping this will leave the handle unrotated"]
struct CreationFut<'a, T, const L: usize> {
    waker: Option<Waker>,
    handle: &'a mut TriplicateHandle<T, L>,
}

impl<'a, T, const L: usize> Future for CreationFut<'a, T, L> {
    type Output = TriplicateHandle<T, L>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.handle.try_create_handle() {
            Ok(handle) => Poll::Ready(handle),
            Err(_) => {
                match this.waker.as_mut() {
                    None => this.waker = Some(cx.waker().clone()),
                    Some(waker) => waker.clone_from(cx.waker()),
                }
                Poll::Pending
            }
        }
    }
}

// TODO: Docs
struct TriplicateInner<T, const L: usize = 3> {
    data: [UnsafeCell<T>; L],
    indices: AtomicU64,
}

impl<const L: usize, T> TriplicateInner<T, L> {
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
unsafe impl<const L: usize, T> Sync for TriplicateInner<T, L> where T: Send + Sync {}

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

    #[test]
    fn bounds_check() {
        let [h1, _h2, _h3]: [TriplicateHandle<_, 3>; 3] =
            TriplicateBounds::new().construct::<Vec<u8>>();
        assert!(h1.is_empty())
    }
}
