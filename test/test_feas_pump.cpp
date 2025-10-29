#include "ObjectiveFeasibilityPump.hpp"

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

Eigen::VectorXd random_vector(const int m, const double range) {
    const unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> dist_val(-range, range);

    Eigen::VectorXd vec (m);
    for (int i=0; i<m; ++i) {
        vec(i) = dist_val(generator);
    }

    return vec;
}



int main()
{
    // dimensions
    const int m = 100;
    const int n = 2300;
    const int nb = 600;
    const double density = 0.003;

    // problem
    Eigen::VectorXd c = random_vector(n, 10.0);
    Eigen::VectorXd x_l (n), x_u (n);
    x_l.setZero();
    x_u.setConstant(1.0);
    const Eigen::SparseMatrix<double> A = random_sparse_matrix(m, n, density, 1.0);
    Eigen::VectorXd A_l (m), A_u (m);
    A_l.setConstant(-0.0);
    A_u.setConstant(0.0);
    std::vector<int> bins;
    for (int i=n-nb; i<n; ++i) {
        bins.push_back(i);
    }

    ObjectiveFeasibilityPump::OFP_Settings settings;
    settings.max_iter = 10000;
    settings.max_restarts = 1000;
    settings.alpha0 = 1.0;
    settings.t_max = 10.0;
    settings.verbose = true;
    settings.verbosity_interval = 1;
    settings.tol = 1e-6;
    ObjectiveFeasibilityPump::OFP_Solver OFP (c, A, A_l, A_u, x_l, x_u, bins, settings);
    const bool success = OFP.solve();
    Eigen::VectorXd sol = OFP.get_solution();
    std::cout << "successful? " << (success ? "true" : "false") << std::endl;

    const ObjectiveFeasibilityPump::OFP_Info info = OFP.get_info();
    std::cout << "iter: " << info.iter << std::endl;
    std::cout << "restarts: " << info.restarts << std::endl;
    std::cout << "perturbations: " << info.perturbations << std::endl;
    std::cout << "runtime: " << info.runtime << std::endl;
    std::cout << "feasible: " << info.feasible << std::endl;
    std::cout << "alpha: " << info.alpha << std::endl;
    std::cout << "objective: " << info.objective << std::endl;

    return 0;
}