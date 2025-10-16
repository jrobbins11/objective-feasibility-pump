#include "ObjectiveFeasibilityPump.hpp"

using namespace ObjectiveFeasibilityPump;

bool OFP_Solver::check_settings(const OFP_Settings& settings)
{
    return settings.alpha0 >= 0 && settings.alpha0 <= 1 && settings.phi > 0 && settings.phi < 1 &&
        settings.max_iter > 0 && settings.max_stalls > 0 && settings.t_max > 0;
}

void OFP_Solver::sparse_eigen_2_highs(Eigen::SparseMatrix<double>& eigen_matrix, HighsSparseMatrix& highs_matrix)
{
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

void OFP_Solver::setup(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const OFP_Settings& settings)
{
    this->c_ = c;
    this->A_ = A;
    this->l_A_ = l_A;
    this->u_A_ = u_A;
    this->l_x_ = l_x;
    this->u_x_ = u_x;
    this->settings_ = settings;
    this->n = static_cast<int>(this->c_.size());
    this->m = static_cast<int>(this->A_.rows());

    if (!check_dimensions())
        throw std::invalid_argument("OFP_Solver setup: dimensions mismatch");
    if (!check_settings(settings))
        throw std::invalid_argument("OFP_Solver setup: settings invalid");

}

bool OFP_Solver::solve(Eigen::VectorXd& sol)
{


    // build HiGHS LP model
    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = m;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = m;
    eigen_vector_2_std_vector<double>(this->c_, model.lp_.col_cost_);
    eigen_vector_2_std_vector<double>(this->l_x_, model.lp_.col_lower_);
    eigen_vector_2_std_vector<double>(this->u_x_, model.lp_.col_upper_);
    eigen_vector_2_std_vector<double>(this->l_A_, model.lp_.row_lower_);
    eigen_vector_2_std_vector<double>(this->u_A_, model.lp_.col_upper_);



}