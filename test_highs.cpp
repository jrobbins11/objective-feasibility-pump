#include <cassert>
#include <iostream>
#include <random>
#include <chrono>

#include "Highs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

void sparse_eigen_2_highs(Eigen::SparseMatrix<double> &eigen_matrix, HighsSparseMatrix &highs_matrix) {
    if (!eigen_matrix.isCompressed()) {
        eigen_matrix.makeCompressed();
    }

    highs_matrix.format_ = MatrixFormat::kColwise;

    int nnz = 0;
    for (int k = 0; k < eigen_matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_matrix, k); it; ++it) {
            highs_matrix.index_.push_back(static_cast<HighsInt>(it.row()));
            highs_matrix.value_.push_back(it.value());
            nnz++;
        }
        highs_matrix.start_.push_back(nnz);
    }
}

Eigen::SparseMatrix<double> random_sparse_matrix(const int m, const int n, const double density, const double range)
{
    assert(density > 0.0 && density <= 1.0);
    assert(m > 0 && n > 0);

    // 1. Calculate the number of non-zero entries (nnz)
    // We use std::round to get the closest integer number of elements.
    const int max_nnz = m*n;
    const int nnz = static_cast<int>(std::round(static_cast<double>(max_nnz) * density));

    // 2. Initialize the result matrix
    // Eigen::SparseMatrix stores entries in compressed format, but we build it from triplets first.
    Eigen::SparseMatrix<double> mat(m, n);

    // 3. Setup Random Number Generators for high-quality randomness
    // Seed the engine using the current time
    const unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // Define distributions for indices (0 to m-1 or 0 to n-1) and values (-10.0 to 10.0)
    std::uniform_int_distribution<int> dist_row(0, m - 1);
    std::uniform_int_distribution<int> dist_col(0, n - 1);
    std::uniform_real_distribution<double> dist_val(-range, range);

    // 4. Create Triplets
    // Eigen::Triplet is (row index, column index, value).
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nnz); // Reserve space to avoid reallocations

    for (size_t k = 0; k < nnz; ++k) {
        // Generate random row, column, and value
        int i = dist_row(generator);
        int j = dist_col(generator);
        double v = dist_val(generator);

        // Add the triplet to the list
        tripletList.emplace_back(i, j, v);
    }

    // 5. Assemble the matrix from the triplet list
    // setFromTriplets efficiently builds the matrix and handles duplicate entries
    // by summing their values.
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    // Optional: Compress the matrix for optimal storage and arithmetic performance
    mat.makeCompressed();

    return mat;
}

std::vector<double> random_vector(const int m, const double range) {
    const unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> dist_val(-range, range);

    std::vector<double> vec;
    vec.reserve(m);
    for (int i=0; i<m; ++i) {
        vec.push_back(dist_val(generator));
    }

    return vec;
}

std::vector<double> const_vector(const int m, const double val) {
    std::vector<double> v;
    v.reserve(m);
    for (int i=0; i<m; ++i) {
        v.push_back(val);
    }
    return v;
}

// following https://github.com/ERGO-Code/HiGHS/blob/master/examples/call_highs_from_cpp.cpp

int main()
{
    // dimensions
    const int m = 1000;
    const int n = 2300;
    const double density = 0.003;

    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = m;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = m;
    model.lp_.col_cost_ = random_vector(n, 10.0);
    model.lp_.col_lower_ = const_vector(n, -10.0);
    model.lp_.col_upper_ = const_vector(n, 10.0);
    model.lp_.row_lower_ = const_vector(m, -10.0);
    model.lp_.row_upper_ = const_vector(m, 10.0);

    // make matrix in Eigen
    Eigen::SparseMatrix<double> A = random_sparse_matrix(m, n, density, 1.0);

    // copy to model matrix
    sparse_eigen_2_highs(A, model.lp_.a_matrix_);

    // create highs instance
    Highs highs;
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", 1);

    HighsStatus return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();

    auto run_and_print = [&]() {
        const auto start = std::chrono::high_resolution_clock::now();

        // solve LP
        return_status = highs.run();
        assert(return_status == HighsStatus::kOk);

        const double run_time = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count());

        // get solution
        const HighsSolution& solution = highs.getSolution();
        const auto info = highs.getInfo();
        const bool has_values = info.primal_solution_status;
        assert(has_values);
        // std::cout << "Solution: [ ";
        // for (int col = 0; col < lp.num_col_; col++)
        // {
        //     std::cout << solution.col_value[col] << " ";
        // }
        // std::cout << "]" << std::endl << std::endl;

        // display runtime
        std::cout << "Runtime = " << run_time << " sec" << std::endl;
    };

    run_and_print();

    // modify objective and resolve
    std::vector<double> delta_cost = random_vector(n, 0.1);
    for (int i=0; i<n; ++i)
        model.lp_.col_cost_[i] += delta_cost[i];
    highs.passModel(model); // should automatically warm-start
    assert(return_status == HighsStatus::kOk);

    run_and_print();

    return 0;
}
