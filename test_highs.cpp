#include <cassert>
#include <iostream>
#include "Highs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

void sparse_eigen_2_highs(Eigen::SparseMatrix<double>& eigen_matrix, HighsSparseMatrix& highs_matrix)
{
    if (!eigen_matrix.isCompressed())
    {
        eigen_matrix.makeCompressed();
    }

    highs_matrix.format_ = MatrixFormat::kColwise;

    int nnz = 0;
    for (int k = 0; k < eigen_matrix.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_matrix, k); it; ++it)
        {
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


    // model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    // // a_start_ has num_col_1 entries, and the last entry is the number
    // // of nonzeros in A, allowing the number of nonzeros in the last
    // // column to be defined
    // model.lp_.a_matrix_.start_ = {0, 2, 5};
    // model.lp_.a_matrix_.index_ = {1, 2, 0, 1, 2};
    // model.lp_.a_matrix_.value_ = {1.0, 3.0, 1.0, 2.0, 2.0};
    // std::cout << "A rows: " << model.lp_.a_matrix_.num_row_ << ", cols: " << model.lp_.a_matrix_.num_col_ << std::endl;

    // make matrix in Eigen
    Eigen::SparseMatrix<double> A(3, 2);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.emplace_back(1, 0, 1.0);
    triplets.emplace_back(2, 0, 3.0);
    triplets.emplace_back(0, 1, 1.0);
    triplets.emplace_back(1, 1, 2.0);
    triplets.emplace_back(2, 1, 2.0);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // copy to model matrix
    sparse_eigen_2_highs(A, model.lp_.a_matrix_);

    // Create a Highs instance
    Highs highs;
    HighsStatus return_status;
    //
    // Pass the model to HiGHS
    return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);
    // If a user passes a model with entries in
    // model.lp_.a_matrix_.value_ less than (the option)
    // small_matrix_value in magnitude, they will be ignored. A logging
    // message will indicate this, and passModel will return
    // HighsStatus::kWarning
    //
    // Get a const reference to the LP data in HiGHS
    const HighsLp& lp = highs.getLp();
    //
    // Solve the model
    return_status = highs.run();
    assert(return_status == HighsStatus::kOk);
    //
    // Get the model status
    const HighsModelStatus& model_status = highs.getModelStatus();
    assert(model_status == HighsModelStatus::kOptimal);
    std::cout << "Model status: " << highs.modelStatusToString(model_status) << std::endl;
    //
    // Get the solution information
    const HighsInfo& info = highs.getInfo();
    std::cout << "Simplex iteration count: " << info.simplex_iteration_count << std::endl;
    std::cout << "Objective function value: " << info.objective_function_value << std::endl;
    std::cout << "Primal  solution status: "
        << highs.solutionStatusToString(info.primal_solution_status) << std::endl;
    std::cout << "Dual    solution status: "
        << highs.solutionStatusToString(info.dual_solution_status) << std::endl;
    std::cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << std::endl;
    const bool has_values = info.primal_solution_status;
    const bool has_duals = info.dual_solution_status;
    const bool has_basis = info.basis_validity;
    //
    // Get the solution values and basis
    const HighsSolution& solution = highs.getSolution();
    const HighsBasis& basis = highs.getBasis();
    //
    // Report the primal and solution values and basis
    for (int col = 0; col < lp.num_col_; col++)
    {
        std::cout << "Column " << col;
        if (has_values) std::cout << "; value = " << solution.col_value[col];
        if (has_duals) std::cout << "; dual = " << solution.col_dual[col];
        if (has_basis)
            std::cout << "; status: " << highs.basisStatusToString(basis.col_status[col]);
        std::cout << std::endl;
    }
    for (int row = 0; row < lp.num_row_; row++)
    {
        std::cout << "Row    " << row;
        if (has_values) std::cout << "; value = " << solution.row_value[row];
        if (has_duals) std::cout << "; dual = " << solution.row_dual[row];
        if (has_basis)
            std::cout << "; status: " << highs.basisStatusToString(basis.row_status[row]);
        std::cout << std::endl;
    }

    // Now indicate that all the variables must take integer values
    model.lp_.integrality_.resize(lp.num_col_);
    for (int col = 0; col < lp.num_col_; col++)
        model.lp_.integrality_[col] = HighsVarType::kInteger;

    highs.passModel(model);
    // Solve the model
    return_status = highs.run();
    assert(return_status == HighsStatus::kOk);
    // Report the primal solution values
    for (int col = 0; col < lp.num_col_; col++)
    {
        std::cout << "Column " << col;
        if (info.primal_solution_status)
            std::cout << "; value = " << solution.col_value[col];
        std::cout << std::endl;
    }
    for (int row = 0; row < lp.num_row_; row++)
    {
        std::cout << "Row    " << row;
        if (info.primal_solution_status)
            std::cout << "; value = " << solution.row_value[row];
        std::cout << std::endl;
    }


    return 0;
}
