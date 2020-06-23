#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include "common.h"

namespace st {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Options {};
}

using Name = llama::DS<
    llama::DE< st::Pos, llama::DS<
        llama::DE< st::X, float >,
        llama::DE< st::Y, float >,
        llama::DE< st::Z, float >
    > >,
    llama::DE< st::Momentum,llama::DS<
        llama::DE< st::Z, double >,
        llama::DE< st::X, double >
    > >,
    llama::DE< st::Weight, int >,
    llama::DE< st::Options, llama::DA< bool, 4 > >
>;

TEST_CASE("demangleType") {
    const auto str = prettyPrintType(Name());
    CHECK(str == R"(struct boost::mp11::mp_list<
    struct boost::mp11::mp_list<
        struct st::Pos,struct boost::mp11::mp_list<
            struct boost::mp11::mp_list<
                struct st::X,float
                >,struct boost::mp11::mp_list<
                    struct st::Y,float
                    >,struct boost::mp11::mp_list<
                        struct st::Z,float
                    >
                >
                >,struct boost::mp11::mp_list<
                    struct st::Momentum,struct boost::mp11::mp_list<
                        struct boost::mp11::mp_list<
                            struct st::Z,double
                            >,struct boost::mp11::mp_list<
                                struct st::X,double
                            >
                        >
                        >,struct boost::mp11::mp_list<
                            struct st::Weight,int
                            >,struct boost::mp11::mp_list<
                                struct st::Options,struct boost::mp11::mp_list<
                                    struct boost::mp11::mp_list<
                                        struct llama::NoName,bool
                                        >,struct boost::mp11::mp_list<
                                            struct llama::NoName,bool
                                            >,struct boost::mp11::mp_list<
                                                struct llama::NoName,bool
                                                >,struct boost::mp11::mp_list<
                                                    struct llama::NoName,bool
                                                >
                                            >
                                        >
                                    >)");
}

TEST_CASE("AoS address") {
    using UD = llama::UserDomain<2>;
    UD udSize{16, 16};
    const auto address = llama::mapping::AoS<UD, Name>(udSize).getBlobByte<0, 1>({0, 100});
    CHECK(address == 3604);
}

TEST_CASE("SoA address") {
    using UD = llama::UserDomain<2>;
    UD udSize{16, 16};
    const auto address = llama::mapping::SoA<UD, Name>(udSize).getBlobByte<0, 1>({0, 100});
    CHECK(address == 1424);
}

TEST_CASE("SizeOf DatumDomain") {
    const auto size = llama::SizeOf<Name>::value;
    CHECK(size == 36);
}

TEST_CASE("StubType") {
    using NameStub = llama::StubType< Name >;
    static_assert(std::is_same_v<Name, NameStub::type>);
    const auto size = llama::SizeOf<Name>::value;
    const auto stubSize = sizeof(NameStub);
    CHECK(size == stubSize);
}

TEST_CASE("GetCoordFromUID") {
    const auto str = prettyPrintType(llama::GetCoordFromUID< Name, st::Pos, st::X >());
    CHECK(str == R"(struct llama::DatumCoord<
    0,0
>)");
}

TEST_CASE("access") {
    using UD = llama::UserDomain<2>;
    UD udSize{16, 16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);

    const UD pos{0, 0};
    CHECK((view(pos) == view[pos]));
    CHECK((view(pos) == view[{0, 0}]));
    CHECK((view(pos) == view({0, 0})));

    float& x = view(pos).access<0, 0>();
    CHECK(&x == &view(pos).access<0>().access<0>());
    CHECK(&x == &view(pos).access<0>().access<st::X>());
    CHECK(&x == &view(pos).access<st::Pos>().access<0>());
    CHECK(&x == &view(pos).access<st::Pos, st::X>());
    CHECK(&x == &view(pos).access<st::Pos>().access<st::X>());
    CHECK(&x == &view(pos)(st::Pos{}, st::X{}));
    CHECK(&x == &view(pos)(st::Pos{})(st::X{}));
    CHECK(&x == &view(pos).access<st::Pos>()(st::X{}));
    CHECK(&x == &view(pos)(st::Pos{}).access<st::X>());
    CHECK(&x == &view(pos)(llama::DatumCoord<0, 0>{}));
    CHECK(&x == &view(pos)(llama::DatumCoord<0>{})(llama::DatumCoord<0>{}));
    // there are even more combinations
}

TEST_CASE("addresses") {
    using UD = llama::UserDomain<2>;
    UD udSize{16, 16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping,llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);

    const UD pos{0, 0};
    auto& x = view(pos).access<st::Pos, st::X>();
    auto& y = view(pos).access<st::Pos, st::Y>();
    auto& z = view(pos).access<st::Pos, st::Z>();
    auto& mz = view(pos).access<st::Momentum, st::Z>();
    auto& mx = view(pos).access<st::Momentum, st::X>();
    auto& w = view(pos)(llama::DatumCoord<2>());
    auto& o0 = view(pos).access<st::Options>().access<0>();
    auto& o1 = view(pos).access<st::Options>().access<1>();
    auto& o2 = view(pos).access<st::Options>().access<2>();
    auto& o3 = view(pos).access<st::Options>().access<3>();

    CHECK((size_t)&y - (size_t)&x == 1024);
    CHECK((size_t)&z - (size_t)&x == 2048);
    CHECK((size_t)&mz - (size_t)&x == 3072);
    CHECK((size_t)&mx - (size_t)&x == 5120);
    CHECK((size_t)&w - (size_t)&x == 7168);
    CHECK((size_t)&o0 - (size_t)&x == 8192);
    CHECK((size_t)&o1 - (size_t)&x == 8448);
    CHECK((size_t)&o2 - (size_t)&x == 8704);
    CHECK((size_t)&o3 - (size_t)&x == 8960);
}

template<typename T_VirtualDatum>
struct SetZeroFunctor {
    template<typename T_OuterCoord, typename T_InnerCoord>
        void operator()(T_OuterCoord, T_InnerCoord) {
        vd(typename T_OuterCoord::template Cat< T_InnerCoord >()) = 0;
    }
    T_VirtualDatum vd;
};

TEST_CASE("iteration and access") {
    using UD = llama::UserDomain< 2 >;
    UD udSize{16, 16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);

    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y) {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::ForEach<Name, llama::DatumCoord<0, 0>>::apply(szf);
            llama::ForEach<Name, st::Momentum >::apply(szf);
            view({x, y}) = double(x + y) / double(udSize[0] + udSize[1]);
        }

    double sum = 0.0;
    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y)
            sum += view(x, y).access< 1, 0 >();
    CHECK(sum == 120.0);
}

TEST_CASE("Datum access") {
    using UD = llama::UserDomain< 2 >;
    UD udSize{16, 16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);

    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y) {
            auto datum = view(x, y);
            datum.access<st::Pos, st::X>() += datum.access<llama::DatumCoord<1, 0>>();
            datum.access(st::Pos(), st::Y()) += datum.access(llama::DatumCoord<1, 1>());
            datum(st::Pos(), st::Z()) += datum(llama::DatumCoord<2>());
            datum(st::Pos()) += datum(st::Momentum());
        }

    double sum = 0.0;
    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y)
            sum += view(x, y).access< 1, 0 >();

    CHECK(sum == 120.0);
}