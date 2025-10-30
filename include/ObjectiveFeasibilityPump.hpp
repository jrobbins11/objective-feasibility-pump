#ifndef OBJECTIVE_FEASIBILITY_PUMP_HPP_
#define OBJECTIVE_FEASIBILITY_PUMP_HPP_

#include <cassert>
#include <iostream>
#include <random>
#include <chrono>
#include <utility>
#include <algorithm>
#include <limits>

#include "Highs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace ObjectiveFeasibilityPump
{
    struct OFP_Settings
    {
        int max_iter = 10000;
        int max_restarts = 100;
        double tol = 1e-6;
        double alpha0 = 1.0;
        double phi = 0.9;
        double delta_alpha = 0.005;
        double t_max = 60.0;
        int lp_threads = 1;
        int buffer_size = 10;
        int T = 20;
        unsigned int rng_seed = 0;
        bool verbose = false;
        int verbosity_interval = 100;
    };

    struct OFP_Info
    {
        int iter = 0;
        int restarts = 0;
        int perturbations = 0;
        double runtime = 0.0;
        bool feasible = false;
        double alpha = 0.0;
        double objective = 0.0;
    };

    class OFP_Solver
    {
    public:

        OFP_Solver() = default;
        OFP_Solver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const std::vector<int>& bins, const OFP_Settings& settings = OFP_Settings(), double b=0.0) {
            setup(c, A, l_A, u_A, l_x, u_x, bins, settings, b);
        }

        void setup(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const std::vector<int>& bins,
            const OFP_Settings& settings = OFP_Settings(), double b=0.0);

        bool solve();

        Eigen::VectorXd get_solution() const;

        OFP_Info get_info() const { return info_; }

    private:
        OFP_Info info_;
        OFP_Settings settings_;
        double b_;
        Eigen::VectorXd c_, l_A_, u_A_, l_x_, u_x_, solution;
        Eigen::SparseMatrix<double> A_;
        std::vector<int> bins_;
        int n = 0, m = 0;
        std::mt19937 rand_gen;

        bool check_dimensions() const;
        static bool check_settings(const OFP_Settings& settings);
        static void sparse_eigen_2_highs(Eigen::SparseMatrix<double>& eigen_matrix, HighsSparseMatrix& highs_matrix);
        bool vectors_equal(const std::vector<double>& a, const std::vector<double>& b) const;
        void restart(const std::vector<double>& x_star, std::vector<double>& x_tilde);
        bool check_feasible(const std::vector<double>& x) const;
        double dist_to_LP_polyhedron(const std::vector<double>& x) const;

        template <typename T>
        static void eigen_vector_2_std_vector(const Eigen::Vector<T, -1>& eigen_vec, std::vector<T>& std_vec) {
            std_vec.clear();
            for (int i=0; i<eigen_vec.size(); ++i)
                std_vec.push_back(eigen_vec(i));
        }

        template <typename T>
        static void std_vector_2_eigen_vector(const std::vector<T>& std_vec, Eigen::Vector<T, -1>& eigen_vec) {
            eigen_vec.resize(std_vec.size());
            for (size_t i=0; i<std_vec.size(); ++i) {
                eigen_vec(i) = std_vec[i];
            }
        }
    };

    template <typename T, typename EqualsComparator>
    class cycle_buffer
    {
    public:
        cycle_buffer(const size_t N, const EqualsComparator& comp): N(N), comp(comp) {
            buffer.reserve(N);
        }

        // check containment
        bool contains(const T& val) const {
            for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
                if (comp(*it, val)) {
                    return true;
                }
            }
            return false;
        }

        // returns false if cycle length > 1 detected
        void insert(const T& val) {
            if (buffer.size() == N) {
                buffer.erase(buffer.begin());
            }
            buffer.push_back(val);
        }

        // empty buffer
        void clear() {
            buffer.clear();
        }

        // get methods
        const std::vector<T>& get_vals() const {
            return buffer;
        }

        const T& get_back() const {
            return buffer.back();
        }

    private:
        const size_t N;
        std::vector<T> buffer;
        EqualsComparator comp;
    };
}


#endif