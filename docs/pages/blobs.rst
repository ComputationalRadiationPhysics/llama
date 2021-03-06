.. include:: common.rst

.. _label-blobs:

Blobs
=====

When a :ref:`view <label-view>` is created, it needs to be given an array of blobs.
A blob is an object representing a contiguous region of memory where each byte is accessible using the subscript operator.
The number of blobs and the size of each blob is a property determined by the mapping used by the view.
All this is handled by :cpp:`llama::allocView()`, but I needs to be given a blob allocator to handle the actual allocation of each blob.

Every time a view is copied, it's array of blobs is copied too.
Depending on the type of blobs used, this can have different effects.
If e.g. :cpp:`std::vector<std::byte>` is used, the full storage will be copied.
Contrary, if a :cpp:`std::shared_ptr<std::byte[]>` is used, the storage is shared between each copy of the view.

.. _label-bloballocators:

Blob allocators
---------------

A blob allocator is a callable which returns an appropriately sized blob given a desired allocation size in bytes.
There is a number of a buildin blob allocators:

Shared memory
^^^^^^^^^^^^^

:cpp:`llama::bloballoc::SharedPtr` is a blob allocator creating blobs of type :cpp:`std::shared_ptr<std::byte[]>`.
These blobs will be shared between each copy of the view and only destroyed then the last view is destroyed.

Furthermore a compile time alignment value can be given to the blob allocator to specify the byte alignment of the allocated block of memory.
This has implications on the read/write speed on some CPU architectures and may even lead to CPU exceptions if data is not properly aligned.

.. code-block:: C++

    llama::bloballoc::SharedPtr<256>

Vector
^^^^^^


:cpp:`llama::bloballoc::Vector` is a blob allocator creating blobs of type :cpp:`std::vector<std::byte>`.
This means every time a view is copied, the whole memory is copied
too. When the view is moved, no extra allocation or copy operation happens.

Analogous to :cpp:`llama::bloballoc::SharedPtr` an alignment value may be passed to the blob allocator.

.. code-block:: C++

    llama::bloballoc::Vector<256>

Stack
^^^^^

When working with small amounts of memory or temporary views created often, it is usually beneficial to store the data directly on the stack.

:cpp:`llama::bloballoc::Stack` addresses this issue and creates blobs of type :cpp:`llama::Array<std::byte, N>`, where :cpp:`N` is a compile time value passed to the allocator.
These blobs are copied every time their view is copied.

Creating a small view of :math:`4 \times 4` may look like this:

.. code-block:: C++

    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims miniSize{4, 4};

    using Mapping = /* some simple mapping */;
    using BlobAllocator = llama::bloballoc::Stack<
        miniSize[0] * miniSize[1] * llama::sizeOf<RecordDim>::value
    >;

    auto miniView = llama::allocView(Mapping{miniSize}, BlobAllocator{});

For :math:`N`-dimensional one-element views a shortcut exists, returning a view
with just one element without any padding, aligment, or whatever on the stack:

.. code-block:: C++

    auto tempView = llama::allocViewStack<N, RecordDim>();


Non-owning blobs
----------------

If a view is needed based on already allocated memory, the view can also be directly constructed with an array of blobs,
e.g. an array of :cpp:`std::byte*` pointers or :cpp:`std::span<std::byte>` to the existing memory regions.
Everything works here as long as it can be subscripted by the view like :cpp:`blob[offset]`.
One needs to be careful though, since now the ownership of the blob is decoupled from the view.
It is the responsibility of the user now to ensure that the blobs outlive the views based on them.

Alpaka
^^^^^^

The following descriptions are for alpaka users.
Without an understanding of alpaka they may hard to understand.

LLAMA features some examples using `alpaka <https://github.com/alpaka-group/alpaka>`_ for the abstraction of computation parallelization.
Alpaka has its own memory allocation functions for different memory regions (e.g. host, device and shared memory).
Additionally there are some cuda-inherited rules which make e.g. sharing memory regions hard (e.g. no possibility to use a :cpp:`std::shared_ptr` on a GPU).

Alpaka creates and manages memory using buffers.
However, a pointer to the underlying storage can be obtained, which may be used for a view:

.. code-block:: C++

    auto buffer = alpaka::allocBuf<std::byte, std::size_t>(dev, size);
    auto view = llama::View<Mapping, std::byte*>{mapping, {alpaka::getPtrNative(buffer)}};

Shared memory is created by alpaka using a special function returning a reference to a shared variable.
To allocate storage for LLAMA, we can allocate a shared byte array using alpaka and then pass the address of the first element to a LLAMA view.

.. code-block:: C++

    auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMemSize], __COUNTER__>(acc);
    auto view = llama::View<Mapping, std::byte*>{mapping, {&sharedMem[0]}};
