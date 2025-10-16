#include "ObjectiveFeasibilityPump.hpp"

using namespace ObjectiveFeasibilityPump;

bool OFP_Solver::check_settings(const OFP_Settings& settings)
{
    return settings.alpha0 >= 0 && settings.alpha0 <= 1 && settings.phi > 0 && settings.phi < 1 &&
        settings.max_iter > 0 && settings.max_stalls > 0 && settings.t_max > 0 && settings.lp_threads >= 1 &&
        settings.buffer_size > 0 && settings.T_frac >= 0 && settings.T_frac < 1;
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

bool OFP_Solver::vectors_equal(const std::vector<double>& a, const std::vector<double>& b, const double tol)
{
    if (a.size() != b.size()) return false;

    for (int i=0; i<a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol)
            return false;
    }

    return true;
}

bool OFP_Solver::check_dimensions() const
{
    if (this->c_.size() != this->n || this->l_x_.size() != this->n || this->u_x_.size() != this->n || this->A_.cols() != this->n)
        return false;
    if (this->l_A_.size() != this->m || this->u_A_.size() != this->m || this->A_.rows() != this->m)
        return false;
    for (const int i : this->bins_) {
        if (i < 0 || i >= this->n)
            return false;
    }
    return true;
}

void OFP_Solver::perturb_binaries(const std::vector<double>& x_star, const std::vector<double>& x_tilde)
{
    constexpr double dist_min = -0.3;
    constexpr double dist_max = 0.7;
}

void OFP_Solver::setup(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const std::vector<int>& bins, const OFP_Settings& settings)
{
    this->c_ = c;
    this->A_ = A;
    this->l_A_ = l_A;
    this->u_A_ = u_A;
    this->l_x_ = l_x;
    this->u_x_ = u_x;
    this->bins_ = bins;
    this->settings_ = settings;
    this->n = static_cast<int>(this->c_.size());
    this->m = static_cast<int>(this->A_.rows());
    this->rand_gen = std::mt19937(this->settings_.rng_seed);

    if (!check_dimensions())
        throw std::invalid_argument("OFP_Solver setup: dimensions mismatch");
    if (!check_settings(settings))
        throw std::invalid_argument("OFP_Solver setup: settings invalid");
}

bool OFP_Solver::solve(Eigen::VectorXd& sol)
{
    // start timer
    const auto start_time = std::chrono::high_resolution_clock::now();

    // build HiGHS LP model
    HighsModel model;
    model.lp_.num_col_ = this->n;
    model.lp_.num_row_ = this->m;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = this->m;
    eigen_vector_2_std_vector<double>(this->c_, model.lp_.col_cost_);
    eigen_vector_2_std_vector<double>(this->l_x_, model.lp_.col_lower_);
    eigen_vector_2_std_vector<double>(this->u_x_, model.lp_.col_upper_);
    eigen_vector_2_std_vector<double>(this->l_A_, model.lp_.row_lower_);
    eigen_vector_2_std_vector<double>(this->u_A_, model.lp_.col_upper_);
    sparse_eigen_2_highs(this->A_, model.lp_.a_matrix_);

    Highs highs;
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", this->settings_.lp_threads);

    HighsStatus return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();

    // solve root relaxation
    return_status = highs.run();
    assert(return_status == HighsStatus::kOk); // make sure optimization ran ok

    const HighsSolution& solution = highs.getSolution();
    assert(highs.getInfo().primal_solution_status); // make sure solution exists

    // (x_tilde, alpha) comparison
    auto x_tilde_alpha_comp = [&](const std::pair<std::vector<double>, double>& a, const std::pair<std::vector<double>, double>& b) -> bool {
        const bool x_tilde_eq = vectors_equal(a.first, b.first, this->settings_.tol);
        const bool alpha_eq = std::abs(a.second - b.second) <= this->settings_.delta_alpha;
        return x_tilde_eq && alpha_eq;
    };

    // fractionality comparison
    auto frac_comp = [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
        return a.second >= b.second;
    };

    // initialize
    std::vector<double> x_star_k = solution.col_value;
    std::vector<double> x_tilde_k, x_tilde_km1;
    int iter = 0;
    int restarts = 0;
    double alpha = this->settings_.alpha0;
    cycle_buffer<std::pair<std::vector<double>, double>, decltype(x_tilde_alpha_comp)> L (this->settings_.buffer_size, x_tilde_alpha_comp);
    const int T = static_cast<int>(std::round(this->settings_.T_frac * n));

    // OFP loop
    this->info_.feasible = true; // set false if early termination
    while (!vectors_equal(x_star_k, x_tilde_k, this->settings_.tol))
    {
        // check for early termination
        const double elapsed_time = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count());
        if (elapsed_time > this->settings_.t_max || iter > this->settings_.max_iter || restarts > this->settings_.max_stalls)
        {
            this->info_.feasible = false;
            break;
        }

        // check for cycle of length 1 and perturb
        if (iter > 1 && vectors_equal(x_tilde_k, x_tilde_km1, this->settings_.tol))
        {
            // find T most fractional variables in x_star
            std::vector<std::pair<int, double>> frac_vec;
            frac_vec.reserve(this->bins_.size());
            for (int i : this->bins_) {
                frac_vec.emplace_back(i, std::min(x_star_k[i], 1.0 - x_star_k[i]));
            }

            // sort with most fractional first
            std::sort(frac_vec.begin(), frac_vec.end(), frac_comp);

            // flip binaries
            for (int i=0; i<T; ++i) {
                const int ib = frac_vec[i].first;
                x_tilde_k[ib] = x_star_k[ib] < 1.0 - x_star_k[ib] ? 1.0 : 0.0;
            }

            // clear buffer
            L.clear();
        }

        // push to buffer and check for cycle
        if (!L.insert(std::make_pair(x_tilde_k, alpha)))
        {


            ++restarts;
        }




        // increment
        ++iter;
    }

    // log info
    this->info_.iter = iter;
    this->info_.restarts = restarts;


}