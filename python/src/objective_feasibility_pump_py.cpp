#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include "ObjectiveFeasibilityPump.hpp"
using namespace ObjectiveFeasibilityPump;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Python bindings for objective_feasibility_pump";

    py::class_<OFP_Settings>(m, "OFP_Settings")
        .def(py::init<>())
        .def_readwrite("max_iter", &OFP_Settings::max_iter)
        .def_readwrite("max_stalls", &OFP_Settings::max_stalls)
        .def_readwrite("tol", &OFP_Settings::tol)
        .def_readwrite("alpha0", &OFP_Settings::alpha0)
        .def_readwrite("phi", &OFP_Settings::phi)
        .def_readwrite("delta_alpha", &OFP_Settings::delta_alpha)
        .def_readwrite("t_max", &OFP_Settings::t_max)
        .def_readwrite("lp_threads", &OFP_Settings::lp_threads)
        .def_readwrite("buffer_size", &OFP_Settings::buffer_size)
        .def_readwrite("T", &OFP_Settings::T)
        .def_readwrite("rng_seed", &OFP_Settings::rng_seed)
    ;

    py::class_<OFP_Info>(m, "OFP_Info")
        .def(py::init<>())
        .def_readonly("iter", &OFP_Info::iter)
        .def_readonly("restarts", &OFP_Info::restarts)
        .def_readonly("perturbations", &OFP_Info::perturbations)
        .def_readonly("runtime", &OFP_Info::runtime)
        .def_readonly("feasible", &OFP_Info::feasible)
        .def_readonly("alpha", &OFP_Info::alpha)
        .def_readonly("objective", &OFP_Info::objective)
    ;

    py::class_<OFP_Solver>(m, "OFP_Solver")
        .def(py::init<const Eigen::VectorXd&, const Eigen::SparseMatrix<double>&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, const std::vector<int>&, const OFP_Settings&>(),
             py::arg("c"), py::arg("A"), py::arg("l_A"), py::arg("u_A"),
             py::arg("l_x"), py::arg("u_x"), py::arg("bins"), py::arg("settings") = OFP_Settings())
        .def("solve", &OFP_Solver::solve)
        .def("get_info", &OFP_Solver::get_info)
        .def("get_solution", &OFP_Solver::get_solution)
    ;
}