#include <cassert>
#include "Highs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

void sparse_eigen_2_highs(Eigen::SparseMatrix<double>& eigen_matrix, HighsSparseMatrix& highs_matrix)
{
    if (!eigen_matrix.isCompressed()) {
        eigen_matrix.makeCompressed();
    }

    highs_matrix.format_ = MatrixFormat::kColwise;
    highs_matrix.num_row_ = eigen_matrix.rows();
    highs_matrix.num_col_ = eigen_matrix.cols();
    highs_matrix.num_nz_ = eigen_matrix.nonZeros();
    highs_matrix.a_.resize(highs_matrix.num_nz_);
    highs_matrix.row_index_.resize(highs_matrix.num_nz_);
    highs_matrix.col_ptr_.resize(highs_matrix.num_col_ + 1);

    int nz_index = 0;
    for (int col = 0; col < eigen_matrix.outerSize(); ++col) {
        highs_matrix.col_ptr_[col] = nz_index;
        for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_matrix, col); it; ++it) {
            highs_matrix.a_[nz_index] = it.value();
            highs_matrix.row_index_[nz_index] = it.row();
            ++nz_index;
        }
    }
    highs_matrix.col_ptr_[highs_matrix.num_col_] = nz_index;
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
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.emplace_back(1, 1, 1.0);
    triplets.emplace_back(1, 2, 3.0);
    triplets.emplace_back(2, 0, 1.0);
    triplets.emplace_back(2, 1, 2.0);
    triplets.emplace_back(2, 2, 2.0);
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
    cout << "Model status: " << highs.modelStatusToString(model_status) << endl;
    //
    // Get the solution information
    const HighsInfo& info = highs.getInfo();
    cout << "Simplex iteration count: " << info.simplex_iteration_count << endl;
    cout << "Objective function value: " << info.objective_function_value << endl;
    cout << "Primal  solution status: "
         << highs.solutionStatusToString(info.primal_solution_status) << endl;
    cout << "Dual    solution status: "
         << highs.solutionStatusToString(info.dual_solution_status) << endl;
    cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << endl;
    const bool has_values = info.primal_solution_status;
    const bool has_duals = info.dual_solution_status;
    const bool has_basis = info.basis_validity;
    //
    // Get the solution values and basis
    const HighsSolution& solution = highs.getSolution();
    const HighsBasis& basis = highs.getBasis();
    //
    // Report the primal and solution values and basis
    for (int col = 0; col < lp.num_col_; col++) {
      cout << "Column " << col;
      if (has_values) cout << "; value = " << solution.col_value[col];
      if (has_duals) cout << "; dual = " << solution.col_dual[col];
      if (has_basis)
        cout << "; status: " << highs.basisStatusToString(basis.col_status[col]);
      cout << endl;
    }
    for (int row = 0; row < lp.num_row_; row++) {
      cout << "Row    " << row;
      if (has_values) cout << "; value = " << solution.row_value[row];
      if (has_duals) cout << "; dual = " << solution.row_dual[row];
      if (has_basis)
        cout << "; status: " << highs.basisStatusToString(basis.row_status[row]);
      cout << endl;
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
    for (int col = 0; col < lp.num_col_; col++) {
      cout << "Column " << col;
      if (info.primal_solution_status)
        cout << "; value = " << solution.col_value[col];
      cout << endl;
    }
    for (int row = 0; row < lp.num_row_; row++) {
      cout << "Row    " << row;
      if (info.primal_solution_status)
        cout << "; value = " << solution.row_value[row];
      cout << endl;
    }



    return 0;
}