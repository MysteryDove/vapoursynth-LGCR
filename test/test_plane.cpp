#include "lgcr.h"

#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace lgcr;

namespace {

void testReadOnlyViewAndCopy() {
    constexpr int width = 5, height = 3, stride = 8;
    std::vector<float> storage(size_t(stride) * height, -99.0f);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            storage[size_t(y) * stride + x] = float(10 * y + x);

    Plane view = Plane::readOnlyView(storage.data(), width, height, stride);
    assert(view.isView() && !view.isWritable());
    assert(view.stride == stride && view.retainedBytes() == 0);
    assert(static_cast<const Plane &>(view).row(2)[4] == 24.0f);
    bool rejected = false;
    try {
        (void)view.row(0);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    assert(rejected);

    Plane copy = view;
    assert(!copy.isView() && copy.isWritable() && copy.stride == width);
    assert(copy.at(4, 2) == 24.0f);
    storage[size_t(2) * stride + 4] = 100.0f;
    assert(copy.at(4, 2) == 24.0f);
}

void testWritableMoveAndDetach() {
    constexpr int width = 7, height = 5, stride = 11;
    std::vector<float> storage(size_t(stride) * height, -7.0f);
    Plane view = Plane::writableView(storage.data(), width, height, stride);
    view.fill(2.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x)
            assert(storage[size_t(y) * stride + x] == 2.0f);
        for (int x = width; x < stride; ++x)
            assert(storage[size_t(y) * stride + x] == -7.0f);
    }

    Plane moved = std::move(view);
    assert(moved.isView() && moved.isWritable() && moved.retainedBytes() == 0);
    moved.at(6, 4) = 9.0f;
    assert(storage[size_t(4) * stride + 6] == 9.0f);

    moved.resizeDiscard(3, 9);
    assert(!moved.isView() && moved.stride == 3);
    assert(moved.retainedBytes() >= size_t(3 * 9) * sizeof(float));
    moved.fill(4.0f);
    assert(storage[0] == 2.0f);
    moved.clear();
    assert(moved.w == 0 && moved.h == 0 && moved.retainedBytes() == 0);
}

void testOwnedCopyMoveAndSwap() {
    Plane original(5, 7);
    for (int y = 0; y < original.h; ++y)
        for (int x = 0; x < original.w; ++x)
            original.at(x, y) = float(y * original.w + x);
    const size_t retained = original.retainedBytes();
    Plane copy(original);
    copy.at(0, 0) = -1.0f;
    assert(original.at(0, 0) == 0.0f);

    Plane moved(std::move(copy));
    assert(moved.at(4, 6) == 34.0f && moved.retainedBytes() == retained);
    Plane other(5, 7);
    other.fill(42.0f);
    moved.swapOwnedStorage(other);
    assert(moved.at(0, 0) == 42.0f && other.at(4, 6) == 34.0f);
}

} // namespace

int main() {
    testReadOnlyViewAndCopy();
    testWritableMoveAndDetach();
    testOwnedCopyMoveAndSwap();
}
