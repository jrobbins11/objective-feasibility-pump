#include "ObjectiveFeasibilityPump.hpp"

using namespace ObjectiveFeasibilityPump;

bool OFP_Solver::check_settings(const OFP_Settings& settings)
{
    return settings.alpha0 >= 0 && settings.alpha0 <= 1 && settings.phi > 0 && settings.phi < 1 &&
        settings.max_iter > 0 && settings.max_stalls > 0 && settings.t_max > 0 && settings.lp_threads >= 1 &&
        settings.buffer_size > 0 && settings.T >= 0;
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

void OFP_Solver::restart(const std::vector<double>& x_star, std::vector<double>& x_tilde)
{
    std::uniform_real_distribution<double> distr(-0.3, 0.7);
    for (const int ib : this->bins_)
    {
        const double frac = std::min(x_star[ib], 1.0 - x_star[ib]);
        if (frac + std::max(distr(this->rand_gen), 0.0) > 0.5) {
            x_tilde[ib] = x_star[ib] < 1.0 - x_star[ib] ? 1.0 : 0.0;
        }
    }
}

void OFP_Solver::setup(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& l_A, const Eigen::VectorXd& u_A,
            const Eigen::VectorXd& l_x, const Eigen::VectorXd& u_x, const std::vector<int>& bins, const OFP_Settings& settings, double b)
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
    this->b_ = b;

    if (!check_dimensions())
        throw std::invalid_argument("OFP_Solver setup: dimensions mismatch");
    if (!check_settings(settings))
        throw std::invalid_argument("OFP_Solver setup: settings invalid");
}

bool OFP_Solver::solve()
{
    // start timer
    const auto start_time = std::chrono::high_resolution_clock::now();

    // build HiGHS LP model
    HighsModel model;
    model.lp_.num_col_ = this->n;
    model.lp_.num_row_ = this->m;
    model.lp_.sense_ = ObjSense::kMinimize;
    eigen_vector_2_std_vector<double>(this->c_, model.lp_.col_cost_);
    eigen_vector_2_std_vector<double>(this->l_x_, model.lp_.col_lower_);
    eigen_vector_2_std_vector<double>(this->u_x_, model.lp_.col_upper_);
    eigen_vector_2_std_vector<double>(this->l_A_, model.lp_.row_lower_);
    eigen_vector_2_std_vector<double>(this->u_A_, model.lp_.row_upper_);
    sparse_eigen_2_highs(this->A_, model.lp_.a_matrix_);

    Highs highs;
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", this->settings_.lp_threads);

    HighsStatus return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();

    // run highs to solve LP
    auto solve_LP = [&]() -> std::vector<double> {
        return_status = highs.run();
        assert(return_status == HighsStatus::kOk); // make sure optimization ran ok

        const HighsSolution& solution = highs.getSolution();
        assert(highs.getInfo().primal_solution_status); // make sure solution exists

        return solution.col_value;
    };

    // (x_tilde, alpha) comparison
    auto x_tilde_alpha_comp = [&](const std::pair<std::vector<double>, double>& a, const std::pair<std::vector<double>, double>& b) -> bool {
        const bool x_tilde_eq = vectors_equal(a.first, b.first, this->settings_.tol);
        const bool alpha_eq = std::abs(a.second - b.second) <= this->settings_.delta_alpha;
        return x_tilde_eq && alpha_eq;
    };

    // fractionality comparison
    auto frac_comp = [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
        return a.second > b.second;
    };

    // initialize
    std::vector<double> x_star_k = solve_LP(); // root relaxation
    std::vector<double> x_tilde_k, x_tilde_km1;
    int iter = 0;
    int restarts = 0;
    int perturbations = 0;
    double alpha = this->settings_.alpha0;
    Eigen::VectorXd Delta_S (this->n);
    Delta_S.setZero();
    cycle_buffer<std::pair<std::vector<double>, double>, decltype(x_tilde_alpha_comp)> L (this->settings_.buffer_size, x_tilde_alpha_comp);

    const int T_min = this->settings_.T/2;
    const int T_max = std::min(3*this->settings_.T/2, static_cast<int>(this->bins_.size()));
    std::uniform_int_distribution<int> T_dist(T_min, T_max);

    // OFP loop
    while (!check_feasible(x_star_k))
    {
        // check for early termination
        const double elapsed_time = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count());
        if (elapsed_time > this->settings_.t_max || iter > this->settings_.max_iter || restarts > this->settings_.max_stalls)
        {
            break;
        }

        // round x_star to get x_tilde
        x_tilde_k = x_star_k;
        for (const int ib : this->bins_)
        {
            x_tilde_k[ib] = std::round(x_tilde_k[ib]);
        }

        // check for cycle of length 1 and perturb
        if (vectors_equal(x_tilde_k, x_tilde_km1, this->settings_.tol))
        {
            // find T most fractional variables in x_star
            std::vector<std::pair<int, double>> frac_vec;
            frac_vec.reserve(this->bins_.size());
            for (const int i : this->bins_) {
                frac_vec.emplace_back(i, std::min(x_star_k[i], 1.0 - x_star_k[i]));
            }

            // sort with most fractional first
            std::sort(frac_vec.begin(), frac_vec.end(), frac_comp);

            // flip binaries
            for (int i=0; i<T_dist(rand_gen); ++i) {
                const int ib = frac_vec[i].first;
                x_tilde_k[ib] = x_star_k[ib] < 1.0 - x_star_k[ib] ? 1.0 : 0.0;
            }

            // clear buffer
            L.clear();

            ++perturbations;
        }

        // push to buffer and check for cycle
        if (!L.insert(std::make_pair(x_tilde_k, alpha)))
        {
            restart(x_star_k, x_tilde_k);
            ++restarts;
        }

        // update objective for LP and resolve
        for (const int ib : this->bins_)
        {
            Delta_S(ib) = (x_tilde_k[ib] < 1.0 - x_tilde_k[ib]) ? 1.0 : -1.0;
        }

        Eigen::VectorXd Delta_S_alpha;
        if (this->c_.norm() > this->settings_.tol) // divide by 0 protection
        {
            Delta_S_alpha = (1.0 - alpha)*Delta_S + alpha*(Delta_S.norm() / this->c_.norm())*this->c_;
        }
        else
        {
            Delta_S_alpha = Delta_S;
        }
        eigen_vector_2_std_vector(Delta_S_alpha, model.lp_.col_cost_);
        highs.passModel(model); // should automatically warm-start
        assert(return_status == HighsStatus::kOk);

        x_star_k = solve_LP();

        // increment
        ++iter;
        alpha *= this->settings_.phi;
        x_tilde_km1 = x_tilde_k;
    }

    // get solution
    std_vector_2_eigen_vector(x_tilde_k, this->solution);

    // log info
    this->info_.iter = iter;
    this->info_.restarts = restarts;
    this->info_.perturbations = perturbations;
    this->info_.runtime = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count());
    this->info_.alpha = alpha;
    this->info_.feasible = check_feasible(x_tilde_k);
    this->info_.objective = this->c_.dot(this->solution) + b_;

    // return success flag
    return this->info_.feasible;
}

Eigen::VectorXd OFP_Solver::get_solution() const
{
    return solution;
}

bool OFP_Solver::check_feasible(const std::vector<double>& x_std) const
{
    Eigen::VectorXd x;
    std_vector_2_eigen_vector(x_std, x);

    if (x.size() != n)
        throw std::invalid_argument("OFP_Solver::check_feasible: x.size() != n");

    for (int i=0; i < n; ++i) {
        if (x(i) < this->l_x_(i) - this->settings_.tol || x(i) > this->u_x_(i) + this->settings_.tol)
            return false;
    }

    Eigen::VectorXd Ax = this->A_*x;
    for (int i=0; i < m; ++i) {
        if (Ax(i) < this->l_A_(i) - this->settings_.tol || Ax(i) > this->u_A_(i) + this->settings_.tol)
            return false;
    }

    for (const int ib : this->bins_) {
        if (std::abs(x(ib) - std::round(x(ib))) > this->settings_.tol)
            return false;
    }

    return true;
}