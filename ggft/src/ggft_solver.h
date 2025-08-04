#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"
#include "unsupported/Eigen/IterativeSolvers"
// #include "config.h"
// #include "geometry/point.h"
// #include "utils/logger.h"


namespace deeprwcap {

// using namespace Eigen;

/*
    |z
    |
    |_____ y
   /
 x/
 index starts from zero, x + y*N + z*N*N
*/
class index {
public:
    // degree of discretization
    static constexpr int N = 23;
    // width of each dielectric block, MAKE SURE N%block_width=0
    static constexpr int block_width = 1;
    static constexpr int blockN = (N/block_width);
    static constexpr int blockNN = blockN*blockN;
    static constexpr int blockNNN = blockN*blockN*blockN;
    static constexpr int NN = N*N;
    static constexpr int NNN = NN*N;

    static constexpr int N1 = NNN;
    static constexpr int N2 = 3*NN*(N-1);
    static constexpr int N3 = 6*NN;


    // for N^3 internal cells
    static constexpr int idx(int i, int j, int k) { return i + j * N + k * NN; }
    
    // for 3(N^3-N^2) dielectric interface panels
    // z-(i,j,k), the panel under panel (i,j,k), requires k > 0
    static constexpr int idx_z_minus(int i, int j, int k) { return i + j * N + (k-1) * NN; }
    // z+(i,j,k) requires k < N-1
    static constexpr int idx_z_plus(int i, int j, int k) { return idx_z_minus(i, j, k+1); }
    // y-(i,j,k) requires j > 0
    static constexpr int idx_y_minus(int i, int j, int k) { return NNN - NN + i + k * N + (j-1) * NN; }
    // y+(i,j,k) requires j < N-1
    static constexpr int idx_y_plus(int i, int j, int k) { return idx_y_minus(i, j+1, k); }
    // x-(i,j,k) requires i > 0
    static constexpr int idx_x_minus(int i, int j, int k) { return 2*(NNN-NN) + j + k * N + (i-1) * NN; }
    // x+(i,j,k) requires i < N-1
    static constexpr int idx_x_plus(int i, int j, int k) { return idx_x_minus(i+1, j, k); }

    // for 6N^2 boundary panels
    static constexpr int idx_b_z_minus(int i, int j /*int k*/) { return i + j * N; }
    static constexpr int idx_b_z_plus(int i, int j /*int k*/) { return NN + i + j * N; }
    static constexpr int idx_b_y_minus(int i, /*int j,*/int k) { return 2*NN + i + k * N; }
    static constexpr int idx_b_y_plus(int i, /*int j,*/int k) { return 3*NN + i + k * N; }
    static constexpr int idx_b_x_minus(/*int i,*/ int j, int k) { return 4*NN + j + k * N; }
    static constexpr int idx_b_x_plus(/*int i,*/ int j, int k) { return 5*NN + j + k * N; }

    static constexpr int idx_b_z_minus(int i, int j, int k) { return i + j * N; }
    static constexpr int idx_b_z_plus(int i, int j, int k) { return NN + i + j * N; }
    static constexpr int idx_b_y_minus(int i, int j, int k) { return 2*NN + i + k * N; }
    static constexpr int idx_b_y_plus(int i, int j, int k) { return 3*NN + i + k * N; }
    static constexpr int idx_b_x_minus(int i, int j, int k) { return 4*NN + j + k * N; }
    static constexpr int idx_b_x_plus(int i, int j, int k) { return 5*NN + j + k * N; }

    static constexpr int center_idx_zzz() { return idx(N/2, N/2, N/2); }
    static constexpr int center_idx_pzz() { return idx(N/2 + 1, N/2, N/2); }
    static constexpr int center_idx_zpz() { return idx(N/2, N/2 + 1, N/2); }
    static constexpr int center_idx_zzp() { return idx(N/2, N/2, N/2 + 1); }
    static constexpr int center_idx_mzz() { return idx(N/2 - 1, N/2, N/2); }
    static constexpr int center_idx_zmz() { return idx(N/2, N/2 - 1, N/2); }
    static constexpr int center_idx_zzm() { return idx(N/2, N/2, N/2 - 1); }
    static constexpr int center_idx_mmz() { return idx(N/2 - 1, N/2 - 1, N/2); }
    static constexpr int center_idx_mzm() { return idx(N/2 - 1, N/2, N/2 - 1); }
    static constexpr int center_idx_zmm() { return idx(N/2, N/2 - 1, N/2 - 1); }
    static constexpr int center_idx_mmm() { return idx(N/2 - 1, N/2 - 1, N/2 - 1); }

    // blocked cell index
    static constexpr int b_idx(int i, int j, int k) { 
        i = i / block_width;
        j = j / block_width;
        k = k / block_width;
        return i + j * blockN + k * blockNN; 
    }

    static constexpr int b_center_idx_zzz() { return b_idx(N/2, N/2, N/2); }
    static constexpr int b_center_idx_pzz() { return b_idx(N/2 + 1, N/2, N/2); }
    static constexpr int b_center_idx_zpz() { return b_idx(N/2, N/2 + 1, N/2); }
    static constexpr int b_center_idx_zzp() { return b_idx(N/2, N/2, N/2 + 1); }
    static constexpr int b_center_idx_mzz() { return b_idx(N/2 - 1, N/2, N/2); }
    static constexpr int b_center_idx_zmz() { return b_idx(N/2, N/2 - 1, N/2); }
    static constexpr int b_center_idx_zzm() { return b_idx(N/2, N/2, N/2 - 1); }
    static constexpr int b_center_idx_mmz() { return b_idx(N/2 - 1, N/2 - 1, N/2); }
    static constexpr int b_center_idx_mzm() { return b_idx(N/2 - 1, N/2, N/2 - 1); }
    static constexpr int b_center_idx_zmm() { return b_idx(N/2, N/2 - 1, N/2 - 1); }
    static constexpr int b_center_idx_mmm() { return b_idx(N/2 - 1, N/2 - 1, N/2 - 1); }
};

class ggft_solver {
   public:
    typedef double real;
    typedef Eigen::SparseMatrix<real, Eigen::ColMajor> sparse_matrix;
    typedef Eigen::SparseMatrix<real, Eigen::RowMajor> sparse_matrix_rowmaj;
    typedef Eigen::SparseVector<real, Eigen::ColMajor> sparse_vector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vectorx;
    enum class solver_code { GFT = 0, WVTZ, WVTX, WVTY };

    // typedef Eigen::SparseLU<sparse_matrix, Eigen::COLAMDOrdering<int>>
    // sparse_solver;
    typedef Eigen::BiCGSTAB<sparse_matrix_rowmaj, Eigen::IdentityPreconditioner>
    // typedef Eigen::MINRES<sparse_matrix_rowmaj, Eigen::IdentityPreconditioner>
    // typedef Eigen::BiCGSTAB<sparse_matrix_rowmaj, Eigen::IncompleteLUT<real>>
        sparse_solver;
    // typedef Eigen::BiCGSTAB<sparse_matrix_rowmaj, Eigen::IncompleteLUT<real>> sparse_solver_stable;
    typedef Eigen::SparseLU<sparse_matrix, Eigen::COLAMDOrdering<int>> sparse_solver_lu;
    std::vector<real> DIEL;
    std::vector<real> STRUCTURE;
    std::vector<bool> sign_is_positive;
    real wvt_sum;
    real center_diel;
    vectorx gft;
    std::ofstream file_gft;
    std::ofstream file_wvtx;
    std::ofstream file_wvty;
    std::ofstream file_wvtz;
    int solve_count = 0;

    // if high-order, we use 2nd-order difference for dielectric interface equation, GFT=ek*(6I+A12*A22^{-1}*A21)^{-1}*(A13-A12*A22^{-1}*A23)
    // else, use 1st-order, GFT=ek*(6I+A12*A21)^{-1}*A13
    static constexpr bool highorder = false;

    ggft_solver(const std::string write_to = "") {
        if (highorder) {
            std::cout << "[Warning] Highorder is not well supported. Very slow and prone to error." << std::endl;
        }
        // std::cout << "N1=" << index::N1 << ", N2=" << index::N2 << ", N3=" << index::N3 << std::endl;
        DIEL_ratio.resize(index::N2, 0);
        // STRUCTURE.resize((N_STRUCTURES+1) * 7, 0);
        DIEL.resize(index::blockNNN, 1);
        sign_is_positive.resize(index::N3, 1);
        build_matrix();
        set_tolerance(1e-8);
        // solverA.set_restart(100);
        bootstrap_solve();
        prepare_file(write_to);
        // file_wvtx.open("wvtx.bin", std::ios::out | std::ios::binary);
        // file_wvty.open("wvty.bin", std::ios::out | std::ios::binary);
        // file_wvtz.open("wvtz.bin", std::ios::out | std::ios::binary);
    }

    ~ggft_solver() {
        if (file_gft.is_open()) {
            file_gft.close();
        }
        // outfile.close();
    }

    void quit() {
        file_gft.close();
        file_wvtx.close();
        file_wvty.close();
        file_wvtz.close();
        exit(1);
    }

    bool solve(const solver_code code);
    void dump_matrix(std::ofstream& outfile) {
        if (!outfile.is_open()) return;
        // sparse_matrix mat(rows,cols);
        // std::ofstream outfile(filename, std::ios::out | std::ios::binary |
        // std::ios::app);
        // real nstructure = STRUCTURE.size();
        // outfile.write(reinterpret_cast<const char*>(&nstructure), sizeof(real));
        outfile.write(reinterpret_cast<const char*>(DIEL.data()), DIEL.size() * sizeof(real));
        outfile.write(reinterpret_cast<const char*>(STRUCTURE.data()), STRUCTURE.size() * sizeof(real));
        outfile.write(reinterpret_cast<const char*>(gft.data()), gft.size() * sizeof(real));
        // outfile.close();
    }

    void set_tolerance(double tol) {
        tolerance = tol;
        solverA22.setTolerance(tol);
        solverA.setTolerance(tol);
        // solverA_stable.setTolerance(tol);
    }
   private:
    double tolerance = 1e-6;

    sparse_matrix A12, A13, A21, A23, Ainv;  // actually all TRANSPOSED!
    sparse_matrix_rowmaj A, A22;                       // actually all TRANSPOSED!
    // sparse_vector ek, ek_dx, ek_dy, ek_dz;
    vectorx ek, ek_dx, ek_dy, ek_dz;
    vectorx inner;
    vectorx single_gft_inner, single_wvtz_inner, single_wvtx_inner, single_wvty_inner;


    sparse_solver solverA22, solverA;
    // sparse_solver_stable solverA_stable;
    sparse_solver_lu solverA22_lu, solverA_lu;
    std::vector<double> DIEL_ratio;
    // std::ofstream outfile;
    // A11 = -6I
    void build_matrix();
    void bootstrap_solve();
    void postprocess(const solver_code code);

    void prepare_file(const std::string &fname);

    void update_matrix();
    void update_ek(const solver_code code);

    void dump_matrix(const sparse_matrix& mat, std::string outf) {
        // sparse_matrix mat(rows,cols);
        std::ofstream outfile(outf, std::ios::out);
        if (!outfile.is_open()) return;
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (sparse_matrix::InnerIterator it(mat, k); it; ++it) {
                outfile << it.row() + 1 << " " << it.col() + 1 << " "
                        << std::scientific << std::setprecision(7) << it.value()
                        << std::endl;
            }
        }
    }


    void laplace(int i, int j, int k);
    void dielectric(int i, int j, int k);
    void dielectric_opt(int i, int j, int k);
    inline bool diel_ieq(int row_id, double ratio) {
        return true;
        return std::abs(DIEL_ratio[row_id] - ratio) > 1e-6;
    }
};

}  // namespace deeprwcap
