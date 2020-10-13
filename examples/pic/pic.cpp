/*
 * main.cpp
 *
 *  Created on: Aug 20, 2020
 *      Author: Jiri Vyskocil
 */
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <sstream>
#include <vector>

using Real = double;

// clang-format off
struct X{};
struct Y{};
struct Z{};

using V3Real = llama::Record<
    llama::Field<X, Real>,
    llama::Field<Y, Real>,
    llama::Field<Z, Real>
>;

struct R{};
struct U{};
struct Q{};
struct M{};

using Particle = llama::Record<
    llama::Field<R, V3Real>,
    llama::Field<U, V3Real>,
    llama::Field<Q, Real>,
    llama::Field<M, Real>
>;
// clang-format on

const Real PI = M_PI;

auto makeV3Real(Real x, Real y, Real z)
{
    llama::One<V3Real> r;
    r(X{}) = x;
    r(Y{}) = y;
    r(Z{}) = z;
    return r;
}

/* Grid operators */
template<typename FieldView>
auto rot(const FieldView& field, size_t i, size_t j, Real inv_dx, int direction)
{
    const size_t ii = i + direction;
    const size_t jj = j + direction;

    return makeV3Real(
        (field(i, j)(Z{}) - field(i, jj)(Z{})) * inv_dx,
        (field(ii, j)(Z{}) - field(i, j)(Z{})) * inv_dx,
        (field(i, j)(Y{}) - field(ii, j)(Y{})) * inv_dx - (field(i, j)(X{}) - field(i, jj)(X{})) * inv_dx);
}

Real inv_dx;
size_t X_ = 100;
size_t Y_ = 150;
Real dx = 0.01;
Real dt = dx / sqrt(2);
int ppc = 5;
int R_ = 5;
int L = 40;
Real density = 1;
auto B0 = 1e6 * makeV3Real(1, 0.5, 0.1);
Real v_drift = 0.01;
Real v_therm = 0.01;
Real alpha = PI / 4;
int NSTEPS = 400;
int outputInterval = 20;
auto Rc = makeV3Real(R_ * dx, R_* dx, 0);

void read_input_file()
{
    if(auto infile = std::ifstream("input.txt"))
    {
        double B0x = 0;
        double B0y = 0;
        double B0z = 0;
        double R0x = 0;
        double R0y = 0;
        std::string line;
        while(infile.good())
        {
            std::getline(infile, line);
            std::string name;
            std::string value;
            std::stringstream stream(line);
            std::getline(stream, name, '=');
            std::getline(stream, value, '=');
            if(name == "X")
                sscanf(value.c_str(), "%zu", &X_);
            else if(name == "Y")
                sscanf(value.c_str(), "%zu", &Y_);
            else if(name == "ppc")
                sscanf(value.c_str(), "%d", &ppc);
            else if(name == "R")
                sscanf(value.c_str(), "%d", &R_);
            else if(name == "L")
                sscanf(value.c_str(), "%d", &L);
            else if(name == "NSTEPS")
                sscanf(value.c_str(), "%d", &NSTEPS);
            else if(name == "outputInterval")
                sscanf(value.c_str(), "%d", &outputInterval);
            else if(name == "dx")
                sscanf(value.c_str(), "%lf", &dx);
            else if(name == "B0x")
                sscanf(value.c_str(), "%lf", &B0x);
            else if(name == "B0y")
                sscanf(value.c_str(), "%lf", &B0y);
            else if(name == "R0x")
                sscanf(value.c_str(), "%lf", &R0x);
            else if(name == "R0y")
                sscanf(value.c_str(), "%lf", &R0y);
            else if(name == "B0z")
                sscanf(value.c_str(), "%lf", &B0z);
            else if(name == "v_drift")
                sscanf(value.c_str(), "%lf", &v_drift);
            else if(name == "v_therm")
                sscanf(value.c_str(), "%lf", &v_therm);
            else if(name == "alpha")
                sscanf(value.c_str(), "%lf", &alpha);
            else if(name == "density")
                sscanf(value.c_str(), "%lf", &density);
        }
        B0 = makeV3Real(B0x, B0y, B0z);
        Rc = makeV3Real(R_ * dx + R0x, R_ * dx + R0y, 0);
        std::cout << "--- input file processed ---" << '\n';
    }
    else
        std::cout << "--- no input file found ---" << '\n';

    std::cout << "X = " << X_ << '\n'
              << "Y = " << Y_ << '\n'
              << "NSTEPS = " << NSTEPS << '\n'
              << "outputInterval = " << outputInterval << '\n'
              << "dx = " << dx << '\n'
              << "ppc = " << ppc << '\n'
              << "R = " << R_ << '\n'
              << "L = " << L << '\n'
              << "B0 = " << B0(X{}) << " " << B0(Y{}) << " " << B0(Z{}) << '\n'
              << "v_drift = " << v_drift << '\n'
              << "v_therm = " << v_therm << '\n'
              << "alpha = " << alpha << '\n'
              << "density = " << density << '\n'
              << "Rc = " << Rc(X{}) << " " << Rc(Y{}) << " " << Rc(Z{}) << '\n';
}

float swapBytes(float t)
{
    uint32_t a;
    std::memcpy(&a, &t, sizeof(a));
    a = ((a & 0x000000FF) << 24) | ((a & 0x0000FF00) << 8) | ((a & 0x00FF0000) >> 8) | ((a & 0xFF000000) >> 24);
    std::memcpy(&t, &a, sizeof(a));
    return t;
}

template<typename VectorField2D>
void output(int n, const std::string& name, const VectorField2D& field)
{
    std::ofstream f("out-" + name + "-" + std::to_string(n) + ".vtk", std::ios::binary);

    f << "# vtk DataFile Version 3.0\nvtkfile\n"
      << "BINARY\nDATASET STRUCTURED_POINTS\n"
      << "DIMENSIONS 1 " << Y_ << " " << X_ << "\n"
      << "ORIGIN 0 0 0\n"
      << "SPACING " << dx << " " << dx << " " << dx << "\n"
      << "POINT_DATA " << X_ * Y_ << "\n"
      << "VECTORS " << name << " float\n";

    std::vector<float> buffer;
    buffer.reserve(X_ * Y_ * 3);
    for(size_t i = 0; i < X_; i++)
    {
        for(size_t j = 0; j < Y_; j++)
        {
            const auto& v = field(i, j);
            buffer.push_back(swapBytes(v(X{})));
            buffer.push_back(swapBytes(v(Y{})));
            buffer.push_back(swapBytes(v(Z{})));
        }
    }
    f.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
}

template<typename ParticleView>
void output(int n, const ParticleView& particles)
{
    std::ofstream outP("out-particles-" + std::to_string(n) + ".vtk", std::ios::binary);
    outP << "# vtk DataFile Version 3.0\nvtkfile\nBINARY\nDATASET POLYDATA\n";

    std::vector<float> buffer;
    auto flushBuffer = [&] { outP.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float)); };
    auto addFloat = [&](float f) { buffer.push_back(swapBytes(f)); };

    const auto pointCount = particles.mapping.arrayDims()[0];
    outP << "POINTS " << pointCount << " float\n";
    buffer.reserve(pointCount * 3);
    for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
    {
        auto p = particles(i);
        addFloat(0);
        addFloat(p(U{}, Y{}));
        addFloat(p(U{}, X{}));
    }
    flushBuffer();

    outP << "POINT_DATA " << pointCount << "\nVECTORS velocity float\n";
    buffer.clear();
    for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
    {
        auto p = particles(i);
        addFloat(p(U{}, Z{}));
        addFloat(p(U{}, Y{}));
        addFloat(p(U{}, X{}));
    }
    flushBuffer();

    outP << "SCALARS q float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
        addFloat(particles(i)(Q{}));
    flushBuffer();

    outP << "SCALARS m float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
        addFloat(particles(i)(M{}));
    flushBuffer();
}

auto setup()
{
    read_input_file();

    inv_dx = 1 / dx;
    dt = 0.99 * dx / sqrt(2);

    std::cout << "dt = " << dt << '\n';

    auto fieldAd = llama::ArrayDims{X_, Y_};
    auto fieldMapping = llama::mapping::AoS<decltype(fieldAd), V3Real>(fieldAd);

    // TODO: how about a llama::forEach that passes an auto& to each leave node instead of the coord?
    auto E = llama::allocView(fieldMapping);
    auto B = llama::allocView(fieldMapping);
    auto J = llama::allocView(fieldMapping);
    for(auto i : llama::ArrayDimsIndexRange{fieldAd})
    {
        E(i) = 0;
        B(i) = B0;
        J(i) = 0;
    }

    const Real r = R_ * dx;
    const Real l = L * dx;

    const size_t numpart = 2 * ppc * R_ * L;

    auto particleAd = llama::ArrayDims{numpart};
    auto particleMapping = llama::mapping::SoA<decltype(particleAd), Particle>{particleAd};

    auto particles = llama::allocView(particleMapping);

    std::default_random_engine engine;

    auto uniform = [&] { return std::uniform_real_distribution<Real>{}(engine); };
    auto gauss = [&] { return sqrt(-2.0 * log(1.0 - uniform())) * sin(2 * PI * uniform()); };

    /* Beam initialization. Plasma must be neutral in each cell, so let's just
     * create an electron and a proton at the same spot. (I know...) */
    for(size_t i = 0; i < numpart; i += 2)
    {
        const Real rp = 2 * r * (uniform() - 0.5);
        const Real lp = l * uniform();
        const auto r = Rc + makeV3Real(lp * sin(alpha) + rp * cos(alpha), lp * cos(alpha) - rp * sin(alpha), 0)
            + dx * makeV3Real(1, 1, 0); // ghost cell
        const auto u = makeV3Real(v_drift * sin(alpha), v_drift * cos(alpha), 0);

        /* Electron */
        auto p = particles(i);
        p(Q{}) = -1;
        p(M{}) = 1;
        p(R{}) = r;
        p(U{}) = u + v_therm * makeV3Real(gauss(), gauss(), gauss());

        /* Proton */
        auto p2 = particles(i + 1);
        p2(Q{}) = 1;
        p2(M{}) = 1386;
        p2(R{}) = r;
        p2(U{}) = u + v_therm * makeV3Real(gauss(), gauss(), gauss());
    }

    std::cout << "Initialized " << numpart << " particles\n";

    return std::tuple{E, B, J, particles};
}

template<typename Vec3F>
auto squaredNorm(const Vec3F& vec) -> Real
{
    return vec(X{}) * vec(X{}) + vec(Y{}) * vec(Y{}) + vec(Z{}) * vec(Z{});
}

template<typename Vec3F>
auto cross(const Vec3F& a, const Vec3F& b)
{
    return makeV3Real(
        a(Y{}) * b(Z{}) - a(Z{}) * b(Y{}),
        a(Z{}) * b(Z{}) - a(Z{}) * b(Z{}),
        a(Z{}) * b(Y{}) - a(Y{}) * b(Z{}));
}

template<typename VirtualParticle, typename FieldView>
void advance_particle(VirtualParticle part, const FieldView& E, const FieldView& B)
{
    /* Interpolate fields (nearest-neighbor) */
    /*FIXME: These are wrong. Even NGP interpolation must respect staggering! */
    const auto i = static_cast<size_t>(part(R{}, X{}) * inv_dx);
    const auto j = static_cast<size_t>(part(R{}, Y{}) * inv_dx);
    const auto El = E(i, j);
    const auto Bl = B(i, j);

    /* q/m [SI] -> 2*PI*q/m in dimensionless */
    const Real qmpidt = part(Q{}) / part(M{}) * PI * dt;
    const Real igamma = 1 / sqrt(1 + squaredNorm(part(U{})));

    const auto F_E = qmpidt * El;
    const auto F_B = igamma * qmpidt * Bl;

    /* Buneman-Boris scheme */
    const auto u_m = part(U{}) + F_E;
    const auto u_0 = u_m + cross(u_m, F_B);
    const auto u_p = u_m + 2 / (1 + squaredNorm(F_B)) * cross(u_0, F_B);
    const auto u = u_p + F_E;

    /* Update stored velocity and advance particle */
    const Real igamma2 = 1 / sqrt(1 + squaredNorm(part(U{})));
    part(U{}) = u;
    part(R{}) += dt * igamma2 * u;
    part(R{}, Z{}) = 0;
}

template<typename ParticleView>
void particle_boundary_conditions(ParticleView& particles)
{ //* Periodic boundary condition
    for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
    {
        auto p = particles(i);
        if(p(R{}, X{}) < dx)
            p(R{}, X{}) += (X_ - 2) * dx;
        else if(p(R{}, X{}) > (X_ - 1) * dx)
            p(R{}, X{}) -= (X_ - 2) * dx;
        if(p(R{}, Y{}) < dx)
            p(R{}, Y{}) += (Y_ - 2) * dx;
        else if(p(R{}, Y{}) > (Y_ - 1) * dx)
            p(R{}, Y{}) -= (Y_ - 2) * dx;
    }
}

template<typename VirtualParticle, typename FieldView>
void deposit_current(const VirtualParticle& part, FieldView& J)
{
    /* Calculation of previous position */
    const Real igamma = 1 / sqrt(1 + squaredNorm(part(U{})));
    const auto r_old = part(R{}) - igamma * dt * part(U{}); // Position should be a 2D vector

    /* Particle position to cell coordinates */
    const auto i = static_cast<size_t>(part(R{}, X{}) * inv_dx);
    const auto j = static_cast<size_t>(part(R{}, Y{}) * inv_dx);

    const auto i_old = static_cast<size_t>(r_old(X{}) * inv_dx);
    const auto j_old = static_cast<size_t>(r_old(Y{}) * inv_dx);

    /* Current density deposition */
    const auto F = density * part(Q{}) * igamma * part(U{});

    /*FIXME: J is defined mid-edge, so this is not NGP.*/
    J(i_old, j_old) -= F;
    J(i, j) += F;
}

template<typename FieldView>
void current_boundary_condition(FieldView& J)
{
    for(size_t i = 0; i < X_; ++i)
    {
        J(i, Y_ - 2) += J(i, size_t{0});
        J(i, size_t{1}) += J(i, Y_ - 1);
    }
    for(size_t j = 1; j < Y_ - 1; j++)
    {
        J(X_ - 2, j) += J(size_t{0}, j);
        J(size_t{1}, j) += J(X_ - 1, j);
    }
}

/* Half-step in B */
template<typename FieldView>
void advance_B_half(const FieldView& E, FieldView& B, size_t i, size_t j)
{
    B(i, j) += 0.5 * dt * rot(E, i, j, inv_dx, 1);
}

/* Full step in E */
template<typename FieldView>
void advance_E(FieldView& E, const FieldView& B, const FieldView& J, size_t i, size_t j)
{
    E(i, j) += dt * rot(B, i, j, inv_dx, -1) - 2 * PI * dt * J(i, j); // full step i n E
}

template<typename FieldView>
void field_boundary_condition(FieldView& field)
{
    for(size_t i = 0; i < X_; ++i)
    {
        field(i, size_t{0}) = field(i, Y_ - 2);
        field(i, Y_ - 1) = field(i, size_t{1});
    }
    for(size_t j = 1; j < Y_ - 1; j++)
    {
        field(size_t{0}, j) = field(X_ - 2, j);
        field(X_ - 1, j) = field(size_t{1}, j);
    }
}

constexpr auto reportInterval = 10;

int main()
{
    auto [E, B, J, particles] = setup();

    std::cout << "Running N = " << NSTEPS << " steps\n";
    /* Timestep */
    using clock = std::chrono::high_resolution_clock;
    clock::duration timeAcc = {};
    for(int n = 0; n <= NSTEPS; n++)
    {
        if(n % reportInterval != 0)
            std::cout << "." << std::flush;
        else if(n > 0)
        {
            std::cout << " " << n << " iterations, "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(timeAcc).count() / reportInterval
                      << "ms per step\n"
                      << std::flush;
            timeAcc = {};
        }
        auto start = clock::now();

        /* Clear charge/current density fields */
        for(size_t i = 0; i < X_; i++)
            for(size_t j = 0; j < Y_; j++)
                J(i, j) = makeV3Real(0, 0, 0);

        /* Integrate equations of motion */
        for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
            advance_particle(particles(i), E, B);
        particle_boundary_conditions(particles);

        /* Deposit charge/current density */
        for(auto i : llama::ArrayDimsIndexRange{particles.mapping.arrayDims()})
            deposit_current(particles(i), J);
        current_boundary_condition(J);

        /* Integrate Maxwell's equations */
        // This could be a view or something
        for(size_t i = 1; i < X_ - 1; i++)
            for(size_t j = 1; j < Y_ - 1; j++)
                advance_B_half(E, B, i, j);
        field_boundary_condition(B); // This could be a view as well

        for(size_t i = 1; i < X_ - 1; i++)
            for(size_t j = 1; j < Y_ - 1; j++)
                advance_E(E, B, J, i, j);
        field_boundary_condition(E);

        for(size_t i = 1; i < X_ - 1; i++)
            for(size_t j = 1; j < Y_ - 1; j++)
                advance_B_half(E, B, i, j);
        field_boundary_condition(B);

        timeAcc += clock::now() - start;

        /* Output data */
        if(n % outputInterval == 0)
        {
            output(n, "E", E);
            output(n, "B", B);
            output(n, "J", J);
            output(n, particles);
        }
    }

    std::cout << "\nSimulation finished\n";
}
