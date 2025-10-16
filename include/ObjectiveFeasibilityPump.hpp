#ifndef OBJECTIVE_FEASIBILITY_PUMP_HPP_
#define OBJECTIVE_FEASIBILITY_PUMP_HPP_

#include <cassert>
#include <iostream>
#include <random>
#include <chrono>

#include "Highs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace ObjectiveFeasibilityPump
{
    struct OFP_Settings
    {
        int max_iter = 10000;
        int max_stalls = 1000;
        double tol = Eigen::NumTraits<double>::dummy_precision();
        double alpha0 = 0.9;
        double phi = 0.9;
        double t_max = 60.0;
    };

    struct OFP_Info
    {
        int iter = 0;
        int stalls = 0;
        double runtime = 0.0;
        bool feasible = false;
    };



    class OFP_Solver
    {
    public:

        OFP_Solver() = default;
        OFP_Solver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const OFP_Settings& settings = OFP_Settings()) {
            setup(c, A, l_A, u_A, l_x, u_x, settings);
        }

        void setup(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const OFP_Settings& settings = OFP_Settings());

        bool solve(Eigen::VectorXd& sol);

    private:
        OFP_Info info_;
        OFP_Settings settings_;
        Eigen::VectorXd c_, l_A_, u_A_, l_x_, u_x_;
        Eigen::SparseMatrix<double> A_;
        int n, m;

        bool check_dimensions() const;
        static bool check_settings(const OFP_Settings& settings);
        static void sparse_eigen_2_highs(Eigen::SparseMatrix<double>& eigen_matrix, HighsSparseMatrix& highs_matrix);

        template <typename T>
        static void eigen_vector_2_std_vector(const Eigen::Vector<T, -1>& eigen_vec, std::vector<T>& std_vec) {
            std_vec.clear();
            for (int i=0; i<eigen_vec.size(); ++i)
                std_vec.push_back(eigen_vec(i));
        }
    };
}


#endif