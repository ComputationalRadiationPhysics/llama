#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Flags {};
}

// clang-format off
using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;
// clang-format on

TEST_CASE("address.AoS")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::AoS<UserDomain, Particle>{userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("address.AoS.fortran")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::
        AoS<UserDomain, Particle, llama::LinearizeUserDomainAdressLikeFortran>{
            userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }
}

TEST_CASE("address.SoA")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::SoA<UserDomain, Particle>{userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }
}

TEST_CASE("address.SoA.fortran")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::
        SoA<UserDomain, Particle, llama::LinearizeUserDomainAdressLikeFortran>{
            userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }
}