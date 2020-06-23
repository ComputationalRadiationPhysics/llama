#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include "common.h"

namespace st {
    struct X {};
    struct Y {};
    struct Z {};
    struct Pos {};
    struct Vel {};
    struct Mom {};
}

using Particle = llama::DS<
    llama::DE<st::Pos, llama::DS<
        llama::DE<st::X, int>,
        llama::DE<st::Y, int>,
        llama::DE<st::Z, int>
    >>,
    llama::DE<st::Mom, int>,
    llama::DE<st::Vel, llama::DS<
        llama::DE<st::Z, int>,
        llama::DE<st::Y, int>,
        llama::DE<st::X, int>
    >>
>;

template<typename T_VirtualDatum>
struct SqrtFunctor {
    template<typename T_OuterCoord, typename T_InnerCoord>
    auto operator()(T_OuterCoord, T_InnerCoord) -> void {
        using Coord = typename T_OuterCoord::template Cat< T_InnerCoord>;
        vd(Coord()) *= std::sqrt(vd(Coord()));
    }
    T_VirtualDatum vd;
};

TEST_CASE("fast virtual view") {
    using UD = llama::UserDomain< 2 >;
    constexpr UD viewSize{4096, 4096};

    using Mapping = llama::mapping::SoA<UD, Particle>;
    auto view = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>::allocView(Mapping(viewSize));

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    llama::VirtualView<decltype(view)> virtualView{
        view,
        {23, 42}, // position
        {13, 37}  // size
    };

    CHECK(virtualView.position == UD{23, 42});
    CHECK(virtualView.size == UD{13, 37});

    CHECK(view(virtualView.position)(st::Pos(), st::X()) == 966);
    CHECK(virtualView({0, 0})(st::Pos(), st::X()) == 966);

    CHECK(view({virtualView.position[0] + 2, virtualView.position[1] + 3})(st::Vel(), st::Z()) == 1125);
    CHECK(virtualView({2, 3})(st::Vel(), st::Z()) == 1125);
}

TEST_CASE("virtual view") {
    using UD = llama::UserDomain<2>;
    constexpr UD viewSize{4096, 4096};
    constexpr UD miniSize{128, 128};
    using Mapping = llama::mapping::SoA<UD, Particle>;
    auto view = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>::allocView(Mapping(viewSize));

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    constexpr UD iterations{
        (viewSize[0] + miniSize[0] - 1) / miniSize[0],
        (viewSize[1] + miniSize[1] - 1) / miniSize[1]
    };

    for (std::size_t x = 0; x < iterations[0]; ++x)
        for (std::size_t y = 0; y < iterations[1]; ++y) {
            const UD validMiniSize{
                (x < iterations[0] - 1) ? miniSize[0] : (viewSize[0] - 1) % miniSize[0] + 1,
                (y < iterations[1] - 1) ? miniSize[1] : (viewSize[1] - 1) % miniSize[1] + 1
            };

            llama::VirtualView<decltype(view)> virtualView(view, {x * miniSize[0], y * miniSize[1]}, miniSize);

            using MiniMapping = llama::mapping::SoA<UD, Particle>;
            auto miniView = llama::Factory<MiniMapping, llama::allocator::Stack<miniSize[0] * miniSize[1] * llama::SizeOf< Particle >::value>>::allocView(MiniMapping(miniSize));

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b)
                    miniView(a, b) = virtualView(a, b);

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b) {
                    SqrtFunctor<decltype(miniView(a, b))> sqrtF{miniView(a, b)};
                    llama::ForEach< Particle >::apply(sqrtF);
                }

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b)
                    virtualView(a, b) = miniView(a, b);
        }

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            CHECK((view(x, y) == x * y * std::sqrt(x * y)));
}