/*
 * main.cpp
 *
 *  Created on: Aug 20, 2020
 *      Author: Jiri Vyskocil
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Geometry>

using Real = double;

/* Vector types shorthands */
using V3Real = Eigen::Matrix<Real, 3, 1>;
using V3Unsigned = Eigen::Matrix<unsigned int, 3, 1>;
using V3Int = Eigen::Matrix<int, 3, 1>;

/* Grid types shorthand */
using VectorField1D = std::vector<V3Real>;
using VectorField2D = std::vector<VectorField1D>;
using VectorField3D = std::vector<VectorField2D>;

/* Basic functions shorthands */
template<typename T> inline T sqr(T x) { return x * x; }
template<typename T> inline T cub(T x) { return x * x * x; }
const Real PI = std::numbers::pi;

/* Grid operators */
V3Real rot(const VectorField2D& field, size_t i, size_t j, Real inv_dx, int direction) {
    const int ii = i + direction;
    const int jj = j + direction;

    return V3Real(
        (field[i][j].z() - field[i][jj].z()) * inv_dx,

        (field[ii][j].z() - field[i][j].z()) * inv_dx,

        (field[i][j].y() - field[ii][j].y()) * inv_dx
        - (field[i][j].x() - field[i][jj].x()) * inv_dx
    );
}

struct Particle {
    V3Real r;
    V3Real u;
    Real q;
    Real m;
};
using ParticlePool = std::vector<Particle>;

VectorField2D E;
VectorField2D B;
VectorField2D J;
ParticlePool particles;

Real inv_dx;
int X = 100;
int Y = 150;
Real dx = 0.01;
Real dt = dx / sqrt(2);
int ppc = 5;
int R = 5;
int L = 40;
Real density = 1;
V3Real B0 = 1e6 * V3Real(1, 0.5, 0.1);
Real v_drift = 0.01;
Real v_therm = 0.01;
Real alpha = PI / 4;
int NSTEPS = 400;
int outputInterval = 20;
V3Real Rc = V3Real(R * dx, R * dx, 0);

void read_input_file() {
    std::string line;
    std::ifstream infile("input.txt");
    if (infile) {
        double B0x = 0;
        double B0y = 0;
        double B0z = 0;
        double R0x = 0;
        double R0y = 0;
        while (infile.good()) {
            std::getline(infile, line);
            std::string name;
            std::string value;
            std::stringstream stream(line);
            std::getline(stream, name, '=');
            std::getline(stream, value, '=');
            if (name == "X")                   sscanf(value.c_str(), "%d", &X);
            else if (name == "Y")              sscanf(value.c_str(), "%d", &Y);
            else if (name == "ppc")            sscanf(value.c_str(), "%d", &ppc);
            else if (name == "R")              sscanf(value.c_str(), "%d", &R);
            else if (name == "L")              sscanf(value.c_str(), "%d", &L);
            else if (name == "NSTEPS")         sscanf(value.c_str(), "%d", &NSTEPS);
            else if (name == "outputInterval") sscanf(value.c_str(), "%d", &outputInterval);
            else if (name == "dx")  	       sscanf(value.c_str(), "%lf", &dx);
            else if (name == "B0x")            sscanf(value.c_str(), "%lf", &B0x);
            else if (name == "B0y")            sscanf(value.c_str(), "%lf", &B0y);
            else if (name == "R0x")            sscanf(value.c_str(), "%lf", &R0x);
            else if (name == "R0y")            sscanf(value.c_str(), "%lf", &R0y);
            else if (name == "B0z")            sscanf(value.c_str(), "%lf", &B0z);
            else if (name == "v_drift")        sscanf(value.c_str(), "%lf", &v_drift);
            else if (name == "v_therm")        sscanf(value.c_str(), "%lf", &v_therm);
            else if (name == "alpha")          sscanf(value.c_str(), "%lf", &alpha);
            else if (name == "density")        sscanf(value.c_str(), "%lf", &density);
        }
        B0 = V3Real(B0x, B0y, B0z);
        Rc = V3Real(R * dx + R0x, R * dx + R0y, 0);
    }
    std::cout << "X = " << X << '\n'
        << "Y = " << Y << '\n'
        << "NSTEPS = " << NSTEPS << '\n'
        << "outputInterval = " << outputInterval << '\n'
        << "dx = " << dx << '\n'
        << "ppc = " << ppc << '\n'
        << "R = " << R << '\n'
        << "L = " << L << '\n'
        << "B0 = " << B0.transpose() << '\n'
        << "v_drift = " << v_drift << '\n'
        << "v_therm = " << v_therm << '\n'
        << "alpha = " << alpha << '\n'
        << "density = " << density << '\n'
        << "Rc = " << Rc.transpose() << '\n'
        << "--- input file processed ---" << '\n';
}

float swapBytes(float t) {
    uint32_t a;
    std::memcpy(&a, &t, sizeof(a));
    a = ((a & 0x000000FF) << 24) |
        ((a & 0x0000FF00) << 8) |
        ((a & 0x00FF0000) >> 8) |
        ((a & 0xFF000000) >> 24);
    std::memcpy(&t, &a, sizeof(a));
    return t;
}

void output(int n, const std::string& name, const VectorField2D& field) {
    std::ofstream f("out-" + name + "-" + std::to_string(n) + ".vtk", std::ios::binary);

    f << "# vtk DataFile Version 3.0\nvtkfile\n"
        << "BINARY\nDATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS 1 " << Y << " " << X << "\n"
        << "ORIGIN 0 0 0\n"
        << "SPACING " << dx << " " << dx << " " << dx << "\n"
        << "POINT_DATA " << X * Y << "\n"
        << "VECTORS " << name << " float\n";

    std::vector<float> buffer;
    buffer.reserve(X * Y * 3);
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            const auto& v = field[i][j];
            buffer.push_back(swapBytes(v.x()));
            buffer.push_back(swapBytes(v.y()));
            buffer.push_back(swapBytes(v.z()));
        }
    }
    f.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
}

void output(int n, const ParticlePool& particles) {
    std::ofstream outP("out-particles-" + std::to_string(n) + ".vtk", std::ios::binary);
    outP << "# vtk DataFile Version 3.0\nvtkfile\nBINARY\nDATASET POLYDATA\n";

    std::vector<float> buffer;
    auto flushBuffer = [&] {
        outP.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
    };
    auto addFloat = [&](float f) {
        buffer.push_back(swapBytes(f));
    };

    outP << "POINTS " << particles.size() << " float\n";
    buffer.reserve(particles.size() * 3);
    for (const auto& p : particles) {
        addFloat(0);
        addFloat(p.r.y());
        addFloat(p.r.x());
    }
    flushBuffer();

    outP << "POINT_DATA " << particles.size() << "\nVECTORS velocity float\n";
    buffer.clear();
    for (const auto& p : particles) {
        addFloat(p.u.z());
        addFloat(p.u.y());
        addFloat(p.u.x());
    }
    flushBuffer();

    outP << "SCALARS q float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for (const auto& p : particles) {
        addFloat(p.q);
    }
    flushBuffer();

    outP << "SCALARS m float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for (const auto& p : particles) {
        addFloat(p.m);
    }
    flushBuffer();
}

void setup() {
    read_input_file();

    inv_dx = 1 / dx;
    dt = 0.99 * dx / sqrt(2);

    std::cout << "dt = " << dt << '\n';

    E = VectorField2D(X, VectorField1D(Y, V3Real(0, 0, 0)));
    B = VectorField2D(X, VectorField1D(Y, B0));
    J = VectorField2D(X, VectorField1D(Y, V3Real(0, 0, 0)));
    particles = {};

    const Real r = R * dx;
    const Real l = L * dx;

    const size_t numpart = 2 * ppc * R * L;
    particles.reserve(numpart);

    std::default_random_engine engine;

    auto uniform = [&] {
        return std::uniform_real_distribution<Real>{}(engine);
    };
    auto gauss = [&] {
        return sqrt(-2.0 * log(1.0 - uniform())) * sin(2 * PI * uniform());
    };

    /* Beam initialization. Plasma must be neutral in each cell, so let's just
     * create an electron and a proton at the same spot. (I know...) */
    for (int i = 0; i < numpart; i += 2) {
        const Real rp = 2 * r * (uniform() - 0.5);
        const Real lp = l * uniform();
        const V3Real r = Rc + V3Real(lp * sin(alpha) + rp * cos(alpha),
            lp * cos(alpha) - rp * sin(alpha), 0)
            + dx * V3Real(1, 1, 0); // ghost cell
        const V3Real u = V3Real(v_drift * sin(alpha), v_drift * cos(alpha), 0);

        /* Electron */
        auto& p = particles.emplace_back();
        p.q = -1;
        p.m = 1;
        p.r = r;
        p.u = u + v_therm * V3Real(gauss(), gauss(), gauss());;

        /* Proton */
        auto& p2 = particles.emplace_back();
        p2.q = 1;
        p2.m = 1386;
        p2.r = r;
        p2.u = u + v_therm * V3Real(gauss(), gauss(), gauss());
    }

    std::cout << "Initialized " << particles.size() << " particles\n";
}

void advance_particle(Particle& part, const VectorField2D& E, const VectorField2D& B) {
    /* Interpolate fields (nearest-neighbor) */
    /*FIXME: These are wrong. Even NGP interpolation must respect staggering! */
    const auto i = static_cast<size_t>(part.r.x() * inv_dx);
    const auto j = static_cast<size_t>(part.r.y() * inv_dx);
    const V3Real El = E[i][j];
    const V3Real Bl = B[i][j];

    /* q/m [SI] -> 2*PI*q/m in dimensionless */
    const Real qmpidt = part.q / part.m * PI * dt;
    const Real igamma = 1 / sqrt(1 + part.u.squaredNorm());

    const V3Real F_E = qmpidt * El;
    const V3Real F_B = igamma * qmpidt * Bl;

    /* Buneman-Boris scheme */
    const V3Real u_m = part.u + F_E;
    const V3Real u_0 = u_m + u_m.cross(F_B);
    const V3Real u_p = u_m + 2 / (1 + F_B.squaredNorm()) * (u_0.cross(F_B));
    const V3Real u = u_p + F_E;

    /* Update stored velocity and advance particle */
    const Real igamma2 = 1 / sqrt(1 + part.u.squaredNorm());
    part.u = u;
    part.r += dt * igamma2 * u;
    part.r.z() = 0;
}

void particle_boundary_conditions(ParticlePool& particles) {  //* Periodic boundary condition
    for (auto& p : particles) {
        if (p.r.x() < dx) { p.r.x() += (X - 2) * dx; }
        else if (p.r.x() > (X - 1) * dx) { p.r.x() -= (X - 2) * dx; }
        if (p.r.y() < dx) { p.r.y() += (Y - 2) * dx; }
        else if (p.r.y() > (Y - 1) * dx) { p.r.y() -= (Y - 2) * dx; }
    }
}

void deposit_current(const Particle& part, VectorField2D& J) {
    /* Calculation of previous position */
    const Real igamma = 1 / sqrt(1 + part.u.squaredNorm());
    const V3Real r_old = part.r - igamma * dt * part.u; // Position should be a 2D vector

    /* Particle position to cell coordinates */
    const int i = static_cast<size_t>(part.r[0] * inv_dx);
    const int j = static_cast<size_t>(part.r[1] * inv_dx);

    const int i_old = static_cast<size_t>(r_old[0] * inv_dx);
    const int j_old = static_cast<size_t>(r_old[1] * inv_dx);

    /* Current density deposition */
    const V3Real F = density * part.q * igamma * part.u;

    /*FIXME: J is defined mid-edge, so this is not NGP.*/
    J[i_old][j_old] -= F;
    J[i][j] += F;
}

void current_boundary_condition(VectorField2D& J) {
    for (size_t i = 0; i < X; ++i) {
        J[i][Y - 2] += J[i][0];
        J[i][1] += J[i][Y - 1];
    }
    for (size_t j = 1; j < Y - 1; j++) {
        J[X - 2][j] += J[0][j];
        J[1][j] += J[X - 1][j];
    }
}

/* Half-step in B */
void advance_B_half(const VectorField2D& E, VectorField2D& B, size_t i, size_t j) {
    B[i][j] += 0.5 * dt * rot(E, i, j, inv_dx, 1);
}

/* Full step in E */
void advance_E(VectorField2D& E, const VectorField2D& B, const VectorField2D& J, size_t i, size_t j) {
    E[i][j] += dt * rot(B, i, j, inv_dx, -1) - 2 * PI * dt * J[i][j]; // full step i n E
}

void field_boundary_condition(VectorField2D& field) {
    for (size_t i = 0; i < X; ++i) {
        field[i][0] = field[i][Y - 2];
        field[i][Y - 1] = field[i][1];
    }
    for (size_t j = 1; j < Y - 1; j++) {
        field[0][j] = field[X - 2][j];
        field[X - 1][j] = field[1][j];
    }
}

constexpr auto reportInterval = 10;

int main() {
    setup();

    std::cout << "Running N = " << NSTEPS << " steps\n";
    /* Timestep */
    using clock = std::chrono::high_resolution_clock;
    clock::duration timeAcc = {};
    for (int n = 0; n <= NSTEPS; n++) {
        if (n % reportInterval != 0) {
            std::cout << "." << std::flush;
        }
        else if (n > 0) {
            std::cout << " " << n << " iterations, " << std::chrono::duration_cast<std::chrono::milliseconds>(timeAcc).count() / reportInterval << "ms per step\n" << std::flush;
            timeAcc = {};
        }
        auto start = clock::now();

        /* Clear charge/current density fields */
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                J[i][j] = V3Real(0, 0, 0);
            }
        }

        /* Integrate equations of motion */
        for (auto& particle : particles) {
            advance_particle(particle, E, B);
        }
        particle_boundary_conditions(particles);

        /* Deposit charge/current density */
        for (auto& particle : particles) {
            deposit_current(particle, J);
        }
        current_boundary_condition(J);

        /* Integrate Maxwell's equations */
        for (int i = 1; i < X - 1; i++) {            // This could be a view or something
            for (int j = 1; j < Y - 1; j++) {
                advance_B_half(E, B, i, j);
            }
        }
        field_boundary_condition(B);  // This could be a view as well

        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++) {
                advance_E(E, B, J, i, j);
            }
        }
        field_boundary_condition(E);

        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++) {
                advance_B_half(E, B, i, j);
            }
        }
        field_boundary_condition(B);

        timeAcc += clock::now() - start;

        /* Output data */
        if (n % outputInterval == 0) {
            output(n, "E", E);
            output(n, "B", B);
            output(n, "J", J);
            output(n, particles);
        }
    }

    std::cout << "\nSimulation finished\n";
}
