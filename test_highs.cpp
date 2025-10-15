#include <cassert>
#include <iostream>
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

// following https://github.com/ERGO-Code/HiGHS/blob/master/examples/call_highs_from_cpp.cpp

int main()
{
    HighsModel model;
    model.lp_.num_col_ = 2;
    model.lp_.num_row_ = 3;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = 3;
    model.lp_.col_cost_ = {1.0, 1.0};
    model.lp_.col_lower_ = {0.0, 1.0};
    model.lp_.col_upper_ = {4.0, 1.0e30};
    model.lp_.row_lower_ = {-1.0e30, 5.0, 6.0};
    model.lp_.row_upper_ = {7.0, 15.0, 1.0e30};

    // make matrix in Eigen
    Eigen::SparseMatrix<double> A(3, 2);
    std::vector<Eigen::Triplet<double> > triplets;
    triplets.emplace_back(1, 0, 1.0);
    triplets.emplace_back(2, 0, 3.0);
    triplets.emplace_back(0, 1, 1.0);
    triplets.emplace_back(1, 1, 2.0);
    triplets.emplace_back(2, 1, 2.0);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // copy to model matrix
    sparse_eigen_2_highs(A, model.lp_.a_matrix_);

    // create highs instance
    Highs highs;
    highs.setOptionValue("log_to_console", false);
    HighsStatus return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();

    auto run_and_print = [&]() {
        // solve LP
        return_status = highs.run();
        assert(return_status == HighsStatus::kOk);

        // get solution
        const HighsSolution& solution = highs.getSolution();
        const auto info = highs.getInfo();
        const bool has_values = info.primal_solution_status;
        assert(has_values);
        std::cout << "Solution: [ ";
        for (int col = 0; col < lp.num_col_; col++)
        {
            std::cout << solution.col_value[col] << " ";
        }
        std::cout << "]" << std::endl;
    };

    run_and_print();

    // modify objective and resolve
    model.lp_.col_cost_[1] = 10.0;
    highs.passModel(model); // should automatically warm-start
    assert(return_status == HighsStatus::kOk);

    run_and_print();

    return 0;
}
