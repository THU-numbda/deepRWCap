/**
 * @file ggft_solver.cpp
 * @author huangjc
 * @date 2023-10-24
 */
#include "ggft_solver.h"
#include <chrono>
#include <sys/stat.h>

namespace rwcap {

bool create_directory_if_not_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        // Directory does not exist. Create it
        if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
            return false; // Failed to create directory
        }
    } else if (!S_ISDIR(st.st_mode)) {
        return false; // Path exists, but it's not a directory
    }
    return true;
}

std::string get_date_string() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    // Convert to a tm struct
    std::tm* localTime = std::localtime(&now);
    // Create a string stream to format the date
    std::ostringstream dateStream;
    dateStream << std::put_time(localTime, "%Y%m%d");
    // Convert to a string
    return dateStream.str();
}

void ggft_solver::prepare_file(const std::string &write_to) {
    std::string fname = "data"+get_date_string()+"/gft_" + std::to_string(index::N) + "_" + std::to_string(index::block_width) + ".bin";
    if (!write_to.empty()) {
        fname = write_to;
    }
    std::string dir = fname.substr(0, fname.find_last_of("/\\"));
    if (!create_directory_if_not_exists(dir)) {
        std::cerr << "Failed to create directory: " << strerror(errno) << std::endl;
    }
    file_gft.open(fname, std::ios::out | std::ios::binary);
    if (file_gft.is_open()) {
        std::cout << "Write to " << fname << std::endl;
        real dN = index::N;
        file_gft.write((char*)&dN, sizeof(real));
        real dW = index::block_width;
        file_gft.write((char*)&dW, sizeof(real));
    }
}

void ggft_solver::build_matrix() {
    ek.setZero(index::N1);
    ek_dx.setZero(index::N1);
    ek_dy.setZero(index::N1);
    ek_dz.setZero(index::N1);
    update_ek(solver_code::GFT);
    update_ek(solver_code::WVTZ);
    update_ek(solver_code::WVTX);
    update_ek(solver_code::WVTY);

    A12.resize(index::N2, index::N1);
    A13.resize(index::N3, index::N1);
    A21.resize(index::N1, index::N2);
    A22.resize(index::N2, index::N2);
    A23.resize(index::N3, index::N2);
    A.resize(index::N1, index::N1);
    for (int k = 0; k < index::N; k++) {
        for (int j = 0; j < index::N; j++) {
            for (int i = 0; i < index::N; i++) {
                laplace(i, j, k);
                dielectric(i, j, k);
            }
        }
    }
    A12.makeCompressed();
    A13.makeCompressed();
    A21.makeCompressed();
    A22.makeCompressed();
    A23.makeCompressed();
    return;
}

void ggft_solver::bootstrap_solve() {
    // solve single
    if (highorder) {
        solverA22.analyzePattern(A22);
        solverA22_lu.analyzePattern(A22);
        solverA22.factorize(A22);
        Ainv = solverA22.solve(A12);
        A = A21 * Ainv;
        for (int i = 0; i < index::N1; i++) {
            A.coeffRef(i, i) += 6;
        }
        solverA.analyzePattern(A);
        solverA_lu.analyzePattern(A);
        solverA.factorize(A);
        single_gft_inner = solverA.solve(ek);
        // gft = A13 * inner - A23 * (Ainv * inner);
        
        single_wvtz_inner = solverA.solve(ek_dz);
        // gft = A13 * inner - A23 * (Ainv * inner);
        
        single_wvtx_inner = solverA.solve(ek_dx);
        // gft = A13 * inner - A23 * (Ainv * inner);
        
        single_wvty_inner = solverA.solve(ek_dy);
        // gft = A13 * inner - A23 * (Ainv * inner);
    } else {  // A22 = I
        A = A21 * A12;
        for (int i = 0; i < index::N1; i++) {
            A.coeffRef(i, i) += 6;
        }
        solverA.analyzePattern(A);
        solverA.factorize(A);
        single_gft_inner = solverA.solve(ek);
        std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        single_wvtz_inner = solverA.solve(ek_dz);
        std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        single_wvtx_inner = solverA.solve(ek_dx);
        std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        single_wvty_inner = solverA.solve(ek_dy);
        std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        // solverA_stable.analyzePattern(A);
        solverA_lu.analyzePattern(A);

        // single_gft = A13 * solverA.solve(ek);
        // single_wvtz = A13 * solverA.solve(ek_dz);
        // std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        // single_wvtx = A13 * solverA.solve(ek_dx);
        // std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
        // single_wvty = A13 * solverA.solve(ek_dy);
        // std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;
    }
    // dump_matrix(gft, "gft_sol.bin", true);
}

void ggft_solver::update_ek(const solver_code code) {
    double h = 2. / index::N;
    if (index::N % 2) {
        // N == 31
        // N/2 = 15
        if (code == solver_code::GFT) {
            ek.coeffRef(index::center_idx_zzz()) = 1;
        } else if (code == solver_code::WVTZ) {
            double d_plus =
                DIEL[index::b_center_idx_zzp()] + DIEL[index::b_center_idx_zzz()];
            double d_minus = -(DIEL[index::b_center_idx_zzz()] +
                               DIEL[index::b_center_idx_zzm()]);
            ek_dz.coeffRef(index::center_idx_zzp()) =
                DIEL[index::b_center_idx_zzp()] / d_plus / h;
            ek_dz.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] * (1. / d_plus + 1. / d_minus) / h;
            ek_dz.coeffRef(index::center_idx_zzm()) =
                DIEL[index::b_center_idx_zzm()] / d_minus / h;
        } else if (code == solver_code::WVTX) {
            double d_plus =
                DIEL[index::b_center_idx_pzz()] + DIEL[index::b_center_idx_zzz()];
            double d_minus = -(DIEL[index::b_center_idx_zzz()] +
                               DIEL[index::b_center_idx_mzz()]);
            ek_dx.coeffRef(index::center_idx_pzz()) =
                DIEL[index::b_center_idx_pzz()] / d_plus / h;
            ek_dx.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] * (1. / d_plus + 1. / d_minus) / h;
            ek_dx.coeffRef(index::center_idx_mzz()) =
                DIEL[index::b_center_idx_mzz()] / d_minus / h;
        } else if (code == solver_code::WVTY) {
            double d_plus =
                DIEL[index::b_center_idx_zpz()] + DIEL[index::b_center_idx_zzz()];
            double d_minus = -(DIEL[index::b_center_idx_zzz()] +
                               DIEL[index::b_center_idx_zmz()]);
            ek_dy.coeffRef(index::center_idx_zpz()) =
                DIEL[index::b_center_idx_zpz()] / d_plus / h;
            ek_dy.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] * (1. / d_plus + 1. / d_minus) / h;
            ek_dy.coeffRef(index::center_idx_zmz()) =
                DIEL[index::b_center_idx_zmz()] / d_minus / h;
        }
    } else {
        // N == 32
        // N/2 = 16
        double dsum =
            DIEL[index::b_center_idx_zzz()] + DIEL[index::b_center_idx_zmz()] +
            DIEL[index::b_center_idx_zzm()] + DIEL[index::b_center_idx_zmm()] +
            DIEL[index::b_center_idx_mzz()] + DIEL[index::b_center_idx_mmz()] +
            DIEL[index::b_center_idx_mzm()] + DIEL[index::b_center_idx_mmm()];
        if (code == solver_code::GFT) {
            ek.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] / dsum;
            ek.coeffRef(index::center_idx_mzz()) =
                DIEL[index::b_center_idx_mzz()] / dsum;
            ek.coeffRef(index::center_idx_zmz()) =
                DIEL[index::b_center_idx_zmz()] / dsum;
            ek.coeffRef(index::center_idx_zzm()) =
                DIEL[index::b_center_idx_zzm()] / dsum;
            ek.coeffRef(index::center_idx_mmz()) =
                DIEL[index::b_center_idx_mmz()] / dsum;
            ek.coeffRef(index::center_idx_mzm()) =
                DIEL[index::b_center_idx_mzm()] / dsum;
            ek.coeffRef(index::center_idx_zmm()) =
                DIEL[index::b_center_idx_zmm()] / dsum;
            ek.coeffRef(index::center_idx_mmm()) =
                DIEL[index::b_center_idx_mmm()] / dsum;
        } else if (code == solver_code::WVTZ) {
            double d_plus =
                (DIEL[index::b_center_idx_zzz()] + DIEL[index::b_center_idx_mzz()] +
                 DIEL[index::b_center_idx_zmz()] + DIEL[index::b_center_idx_mmz()]);
            double d_minus = -(
                DIEL[index::b_center_idx_zzm()] + DIEL[index::b_center_idx_mzm()] +
                DIEL[index::b_center_idx_zmm()] + DIEL[index::b_center_idx_mmm()]);
            ek_dz.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] / dsum / h * 2;
            ek_dz.coeffRef(index::center_idx_mzz()) =
                DIEL[index::b_center_idx_mzz()] / dsum / h * 2;
            ek_dz.coeffRef(index::center_idx_zmz()) =
                DIEL[index::b_center_idx_zmz()] / dsum / h * 2;
            ek_dz.coeffRef(index::center_idx_zzm()) =
                DIEL[index::b_center_idx_zzm()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dz.coeffRef(index::center_idx_mmz()) =
                DIEL[index::b_center_idx_mmz()] / dsum / h * 2;
            ek_dz.coeffRef(index::center_idx_mzm()) =
                DIEL[index::b_center_idx_mzm()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dz.coeffRef(index::center_idx_zmm()) =
                DIEL[index::b_center_idx_zmm()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dz.coeffRef(index::center_idx_mmm()) =
                DIEL[index::b_center_idx_mmm()] * (1. / dsum + 1. / d_minus) / h * 2;
        } else if (code == solver_code::WVTX) {
            double d_plus =
                (DIEL[index::b_center_idx_zzz()] + DIEL[index::b_center_idx_zmz()] +
                 DIEL[index::b_center_idx_zzm()] + DIEL[index::b_center_idx_zmm()]);
            double d_minus = -(
                DIEL[index::b_center_idx_mzz()] + DIEL[index::b_center_idx_mmz()] +
                DIEL[index::b_center_idx_mzm()] + DIEL[index::b_center_idx_mmm()]);
            ek_dx.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] / dsum / h * 2;
            ek_dx.coeffRef(index::center_idx_mzz()) =
                DIEL[index::b_center_idx_mzz()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dx.coeffRef(index::center_idx_zmz()) =
                DIEL[index::b_center_idx_zmz()] / dsum / h * 2;
            ek_dx.coeffRef(index::center_idx_zzm()) =
                DIEL[index::b_center_idx_zzm()] / dsum / h * 2;
            ek_dx.coeffRef(index::center_idx_mmz()) =
                DIEL[index::b_center_idx_mmz()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dx.coeffRef(index::center_idx_mzm()) =
                DIEL[index::b_center_idx_mzm()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dx.coeffRef(index::center_idx_zmm()) =
                DIEL[index::b_center_idx_zmm()] / dsum / h * 2;
            ek_dx.coeffRef(index::center_idx_mmm()) =
                DIEL[index::b_center_idx_mmm()] * (1. / dsum + 1. / d_minus) / h * 2;
        } else if (code == solver_code::WVTY) {
            double d_plus =
                (DIEL[index::b_center_idx_zzz()] + DIEL[index::b_center_idx_mzz()] +
                 DIEL[index::b_center_idx_zzm()] + DIEL[index::b_center_idx_mzm()]);
            double d_minus = -(
                DIEL[index::b_center_idx_zmz()] + DIEL[index::b_center_idx_mmz()] +
                DIEL[index::b_center_idx_zmm()] + DIEL[index::b_center_idx_mmm()]);
            ek_dy.coeffRef(index::center_idx_zzz()) =
                DIEL[index::b_center_idx_zzz()] / dsum / h * 2;
            ek_dy.coeffRef(index::center_idx_mzz()) =
                DIEL[index::b_center_idx_mzz()] / dsum / h * 2;
            ek_dy.coeffRef(index::center_idx_zmz()) =
                DIEL[index::b_center_idx_zmz()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dy.coeffRef(index::center_idx_zzm()) =
                DIEL[index::b_center_idx_zzm()] / dsum / h * 2;
            ek_dy.coeffRef(index::center_idx_mmz()) =
                DIEL[index::b_center_idx_mmz()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dy.coeffRef(index::center_idx_mzm()) =
                DIEL[index::b_center_idx_mzm()] / dsum / h * 2;
            ek_dy.coeffRef(index::center_idx_zmm()) =
                DIEL[index::b_center_idx_zmm()] * (1. / dsum + 1. / d_minus) / h * 2;
            ek_dy.coeffRef(index::center_idx_mmm()) =
                DIEL[index::b_center_idx_mmm()] * (1. / dsum + 1. / d_minus) / h * 2;
        }
    }
}

void ggft_solver::update_matrix() {
    if (!highorder) {
        for (int i = 0; i < index::N1; i++) {
            A.coeffRef(i, i) = 0;
        }
    }
    for (int k = 0; k < index::N; k++) {
        for (int j = 0; j < index::N; j++) {
            for (int i = 0; i < index::N; i++) {
                if (highorder) dielectric(i, j, k);
                else dielectric_opt(i, j, k);
            }
        }
    }
}

void ggft_solver::laplace(int i, int j, int k) {
    int row_id = index::idx(i, j, k);
    if (k > 0) {
        A12.insert(index::idx_z_minus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_z_minus(i, j, k), row_id) = 1;
    }
    if (k < index::N - 1) {
        A12.insert(index::idx_z_plus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_z_plus(i, j, k), row_id) = 1;
    }

    if (j > 0) {
        A12.insert(index::idx_y_minus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_y_minus(i, j, k), row_id) = 1;
    }
    if (j < index::N - 1) {
        A12.insert(index::idx_y_plus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_y_plus(i, j, k), row_id) = 1;
    }

    if (i > 0) {
        A12.insert(index::idx_x_minus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_x_minus(i, j, k), row_id) = 1;
    }
    if (i < index::N - 1) {
        A12.insert(index::idx_x_plus(i, j, k), row_id) = 1;
    } else {
        A13.insert(index::idx_b_x_plus(i, j, k), row_id) = 1;
    }
}

void ggft_solver::dielectric(int i, int j, int k) {
    // for z-
    if (k > 0) {
        double d_minus = DIEL[index::b_idx(i, j, k - 1)],
               d_plus = DIEL[index::b_idx(i, j, k)];
        {
            double sum = d_minus + d_plus;
            d_minus /= sum;
            d_plus /= sum;
        }
        int row_id = index::idx_z_minus(i, j, k);
        // if (diel_ieq(row_id, d_minus / d_plus)) {
            if (highorder) {
                A22.coeffRef(row_id, row_id) = 1.5;
                A21.coeffRef(index::idx(i, j, k - 1), row_id) = -2 * d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -2 * d_plus;
                if (k > 1) {
                    A22.coeffRef(index::idx_z_minus(i, j, k - 1), row_id) =
                        0.5 * d_minus;
                } else {
                    A23.coeffRef(index::idx_b_z_minus(i, j, k - 1), row_id) =
                        0.5 * d_minus;
                }
                if (k < index::N - 1) {
                    A22.coeffRef(index::idx_z_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                } else {
                    A23.coeffRef(index::idx_b_z_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                }
            } else {
                // A22.coeffRef(row_id, row_id) = 1;
                A21.coeffRef(index::idx(i, j, k - 1), row_id) = -d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -d_plus;
            }
            // DIEL_ratio[row_id] = d_minus / d_plus;
        // }
    }

    // for y-
    if (j > 0) {
        double d_minus = DIEL[index::b_idx(i, j - 1, k)],
               d_plus = DIEL[index::b_idx(i, j, k)];
        {
            double sum = d_minus + d_plus;
            d_minus /= sum;
            d_plus /= sum;
        }
        int row_id = index::idx_y_minus(i, j, k);
        // if (diel_ieq(row_id, d_minus / d_plus)) {
            if (highorder) {
                A22.coeffRef(row_id, row_id) = 1.5;
                A21.coeffRef(index::idx(i, j - 1, k), row_id) = -2 * d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -2 * d_plus;
                if (j > 1) {
                    A22.coeffRef(index::idx_y_minus(i, j - 1, k), row_id) =
                        0.5 * d_minus;
                } else {
                    A23.coeffRef(index::idx_b_y_minus(i, j - 1, k), row_id) =
                        0.5 * d_minus;
                }
                if (j < index::N - 1) {
                    A22.coeffRef(index::idx_y_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                } else {
                    A23.coeffRef(index::idx_b_y_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                }
            } else {
                // A22.coeffRef(row_id, row_id) = 1;
                A21.coeffRef(index::idx(i, j - 1, k), row_id) = -d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -d_plus;
            }
            // DIEL_ratio[row_id] = d_minus / d_plus;
        // }
    }

    // for x-
    if (i > 0) {
        double d_minus = DIEL[index::b_idx(i - 1, j, k)],
               d_plus = DIEL[index::b_idx(i, j, k)];
        {
            double sum = d_minus + d_plus;
            d_minus /= sum;
            d_plus /= sum;
        }
        int row_id = index::idx_x_minus(i, j, k);
        // if (diel_ieq(row_id, d_minus / d_plus)) {
            if (highorder) {
                A22.coeffRef(row_id, row_id) = 1.5;
                A21.coeffRef(index::idx(i - 1, j, k), row_id) = -2 * d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -2 * d_plus;
                if (i > 1) {
                    A22.coeffRef(index::idx_x_minus(i - 1, j, k), row_id) =
                        0.5 * d_minus;
                } else {
                    A23.coeffRef(index::idx_b_x_minus(i - 1, j, k), row_id) =
                        0.5 * d_minus;
                }
                if (i < index::N - 1) {
                    A22.coeffRef(index::idx_x_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                } else {
                    A23.coeffRef(index::idx_b_x_plus(i, j, k), row_id) =
                        0.5 * d_plus;
                }
            } else {
                // A22.coeffRef(row_id, row_id) =
                // (d_minus+d_plus);
                A21.coeffRef(index::idx(i - 1, j, k), row_id) = -d_minus;
                A21.coeffRef(index::idx(i, j, k), row_id) = -d_plus;
            }
            // DIEL_ratio[row_id] = d_minus / d_plus;
        // }
    }
}



void ggft_solver::dielectric_opt(int i, int j, int k) {
    double diag = 0;
    double d_self = DIEL[index::b_idx(i, j, k)];
    int rowid = index::idx(i, j, k);
    // // z-
    // if (k > 0) {
    //     double d_other = DIEL[index::b_idx(i, j, k-1)];
    //     double e = 1/(1+d_other/d_self);
    //     int other = index::idx(i, j, k-1);
    //     A.coeffRef(other, rowid) = e-1;
    //     diag += e;
    //     A.coeffRef(rowid, other) = -e;
    //     A.coeffRef(other, other) += 1-e;
    // }
    // z+
    if (k < index::N - 1) {
        double d_other = DIEL[index::b_idx(i, j, k+1)];
        double e = 1/(1+d_other/d_self);
        int other = index::idx(i, j, k+1);
        // double e = d_other/(d_self+d_other);
        A.coeffRef(other, rowid) = e-1;
        diag += e;
        A.coeffRef(rowid, other) = -e;
        A.coeffRef(other, other) += 1-e;
    }
    // // y-
    // if (j > 0) {
    //     double d_other = DIEL[index::b_idx(i, j-1, k)];
    //     double e = 1/(1+d_other/d_self);
    //     // double e = d_other/(d_self+d_other);
    //     int other = index::idx(i, j-1, k);
    //     A.coeffRef(other, rowid) = e-1;
    //     diag += e;
    //     A.coeffRef(rowid, other) = -e;
    //     A.coeffRef(other, other) += 1-e;
    // }
    // y+
    if (j < index::N - 1) {
        double d_other = DIEL[index::b_idx(i, j+1, k)];
        double e = 1/(1+d_other/d_self);
        int other = index::idx(i, j+1, k);
        // double e = d_other/(d_self+d_other);
        A.coeffRef(other, rowid) = e-1;
        diag += e;
        A.coeffRef(rowid, other) = -e;
        A.coeffRef(other, other) += 1-e;
    }
    // // x-
    // if (i > 0) {
    //     double d_other = DIEL[index::b_idx(i-1, j, k)];
    //     double e = 1/(1+d_other/d_self);
    //     // double e = d_other/(d_self+d_other);
    //     int other = index::idx(i-1, j, k);
    //     A.coeffRef(other, rowid) = e-1;
    //     diag += e;
    //     A.coeffRef(rowid, other) = -e;
    //     A.coeffRef(other, other) += 1-e;
    // }
    // x+
    if (i < index::N - 1) {
        double d_other = DIEL[index::b_idx(i+1, j, k)];
        // double e = d_other/(d_self+d_other);
        double e = 1/(1+d_other/d_self);
        int other = index::idx(i+1, j, k);
        A.coeffRef(other, rowid) = e-1;
        diag += e;
        A.coeffRef(rowid, other) = -e;
        A.coeffRef(other, other) += 1-e;
    }
    // std::cout << "final " << rowid << " " << diag << " previous " << A.coeff(rowid, rowid) << std::endl;
    A.coeffRef(rowid, rowid) = 6 - diag - A.coeff(rowid, rowid);
}


bool ggft_solver::solve(const solver_code code) {
    update_matrix();
    update_ek(code);
    if (highorder) {
        solverA22.factorize(A22);
        Ainv = solverA22.solve(A12);
        if (solverA22.info() != Eigen::Success || std::isnan(solverA22.error())) {
            solverA22_lu.factorize(A22);
            Ainv = solverA22_lu.solve(A12);
            real residual = (A22 * Ainv - A12).norm() / A12.norm();
            if (solverA22_lu.info() != Eigen::Success || std::isnan(residual)) {
                std::cout << "[Error] Failed to solve A22." << std::endl;
                return false;
            }
            if (residual > tolerance) {
                std::cout << "[Warning] Large residual " << residual << std::endl;
            }
        }
        A = A21 * Ainv;
        for (int i = 0; i < index::N1; i++) {
            A.coeffRef(i, i) += 6;
        }
    } else {  // A22 = I
        // A = A21 * A12;
        // for (int i = 0; i < index::N1; i++) {
        //     A.coeffRef(i, i) += 6;
        // }
        // std::cout << "#iterations:     " << solverA.iterations() <<
        // std::endl; std::cout << "estimated error: " << solverA.error()  << "
        // "  << solverA.tolerance()    << std::endl;
    }
    solverA.factorize(A);
    if (code == solver_code::GFT) {
        inner = solverA.solveWithGuess(ek, single_gft_inner);
    } else if (code == solver_code::WVTZ) {
        inner = solverA.solveWithGuess(ek_dz, single_wvtz_inner);
    } else if (code == solver_code::WVTX) {
        inner = solverA.solveWithGuess(ek_dx, single_wvtx_inner);
    } else if (code == solver_code::WVTY) {
        inner = solverA.solveWithGuess(ek_dy, single_wvty_inner);
    }

    bool dump = false;

    if (solverA.info() != Eigen::Success || std::isnan(solverA.error())) {
        solverA_lu.factorize(A);
        real residual = 0;
        if (code == solver_code::GFT) {
            inner = solverA_lu.solve(ek);
            residual = (A*inner - ek).norm() / ek.norm();
        } else if (code == solver_code::WVTZ) {
            inner = solverA_lu.solve(ek_dz);
            residual = (A*inner - ek_dz).norm() / ek_dz.norm();
        } else if (code == solver_code::WVTX) {
            inner = solverA_lu.solve(ek_dx);
            residual = (A*inner - ek_dx).norm() / ek_dx.norm();
        } else if (code == solver_code::WVTY) {
            inner = solverA_lu.solve(ek_dy);
            residual = (A*inner - ek_dy).norm() / ek_dy.norm();
        }
        if (solverA_lu.info() != Eigen::Success || std::isnan(residual)) {
            std::cout << "[Error] Failed to solve." << std::endl;
            return false;
        }
        if (residual > tolerance) {
            std::cout << "[Warning] Large residual " << residual << std::endl;
            dump = true;
        }
    }
    // std::cout << "iter " << solverA.iterations() << " error " << solverA.error() << std::endl;

    if (highorder) {
        gft = A13 * inner - A23 * (Ainv * inner);
    } else {
        gft = A13 * inner;
    }

    if (code == solver_code::GFT) {
        bool wrong_distribution = std::abs(gft.sum()-1) > 0.01 || (gft.array() < 0).any();
        if (wrong_distribution) {
            solverA_lu.factorize(A);
            real residual = 0;
            inner = solverA_lu.solve(ek);
            residual = (A*inner - ek).norm() / ek.norm();
            if (solverA_lu.info() != Eigen::Success || std::isnan(residual)) {
                std::cout << "[Error] Failed to solve." << std::endl;
                return false;
            }
            if (residual > tolerance) {
                std::cout << "[Warning] Large residual " << residual << std::endl;
                dump = true;
            }
            if (highorder) {
                gft = A13 * inner - A23 * (Ainv * inner);
            } else {
                gft = A13 * inner;
            }
        }
        double gsum = gft.sum();
        if (std::abs(gsum-1) > 0.01) {
            std::cout << "[Error] Encountered a GF with sum=" << gsum << std::endl;
        }
        if ((gft.array() < 0).any()) {
            std::cout << "[Error] Encountered a GF with negative values." << std::endl;
        }
    }


    // if (dump) {
    {
        dump_matrix(file_gft);
    }
    // postprocess(code);
    return true;
    // if (code == solver_code::GFT) {
    //     dump_matrix(file_gft, gft);
    // } else if (code == solver_code::WVTZ) {
    //     dump_matrix(file_wvtz, gft);
    // } else if (code == solver_code::WVTX) {
    //     dump_matrix(file_wvtx, gft);
    // } else if (code == solver_code::WVTY) {
    //     dump_matrix(file_wvty, gft);
    // }
    // solve_count++;
    // if (solve_count == 1000) {
    //     quit();
    // }
    
    // dump_matrix(gft, "gft_sol.bin");
    // dump_matrix(gft);
}

void ggft_solver::postprocess(const solver_code code) {
    if (code != solver_code::GFT) {
        for (int i = 0; i < index::N3; i++) {
            if (gft(i) >= 0) {
                sign_is_positive[i] = true;
            } else {
                gft(i) = -gft(i);
                sign_is_positive[i] = false;
            }
        }

        if (index::N % 2) {
            center_diel = DIEL[index::b_center_idx_zzz()];
        } else {
            if (code == solver_code::WVTZ) {
                center_diel = (DIEL[index::b_center_idx_zzm()] +
                               DIEL[index::b_center_idx_mmm()] +
                               DIEL[index::b_center_idx_zmm()] +
                               DIEL[index::b_center_idx_mzm()]) /
                              4;
            } else if (code == solver_code::WVTX) {
                center_diel = (DIEL[index::b_center_idx_mzz()] +
                               DIEL[index::b_center_idx_mmz()] +
                               DIEL[index::b_center_idx_mzm()] +
                               DIEL[index::b_center_idx_mmm()]) /
                              4;
            } else if (code == solver_code::WVTY) {
                center_diel = (DIEL[index::b_center_idx_zmz()] +
                               DIEL[index::b_center_idx_mmz()] +
                               DIEL[index::b_center_idx_zmm()] +
                               DIEL[index::b_center_idx_mmm()]) /
                              4;
            }
        }
    }
    // std::ofstream outfile("oct.bin", std::ios::out | std::ios::binary);
    // outfile.write((char*)(gft.data()), gft.size() * sizeof(double));
    // outfile.close();
    // exit(1);


    // to cumulative distribution function
    wvt_sum = gft.sum();
    gft /= wvt_sum;
    for (int i = 1; i < index::N3; i++) {
        gft(i) += gft(i - 1);
    }
}

}  // namespace rwcap
