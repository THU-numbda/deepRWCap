#pragma once
#include <random>

#include "ggft_solver.h"

namespace rwcap {

	struct dielectric_block {
		// center
		double x=0, y=0, z=0;
		// length
		double x_radius=0, y_radius=0, z_radius=0;
		double value=0;
		bool contain(double _x, double _y, double _z) {
			return std::abs(_x - x) < x_radius && std::abs(_y - y) < y_radius && std::abs(_z - z) < z_radius;
		}
		dielectric_block expand(double r, double eps) {
			double dr = r * std::max(x_radius, std::max(y_radius, z_radius));
			return {x, y, z, x_radius+dr, y_radius+dr, z_radius+dr, eps};
		}
	};
	
	class dielectric_generator {
		unsigned seed = 0;
		std::mt19937 eng;
	
		std::vector<dielectric_block> random_blocks(int block_count) {
			double left = -2, right = 2;
			double min_radius = 0, max_radius = 2;
			double min_diel = 2, max_diel = 20;
			// x,y,z ~ U(-2,2)
			// 
			
			std::uniform_real_distribution<> center(left, right);
			std::uniform_real_distribution<> radius(min_radius, max_radius);
			std::uniform_real_distribution<> diel(min_diel, max_diel);
			double background_diel = diel(eng);
	
			std::vector<dielectric_block> blocks;
			for (int i = 0; i < block_count; i++) {
				blocks.push_back({center(eng), center(eng), center(eng), radius(eng), radius(eng), radius(eng), diel(eng)});
			}
			blocks.push_back({0, 0, 0, right, right, right, background_diel});
			return blocks;
		}
	
		std::vector<dielectric_block> random_blocks_practical(int block_count) {
			double left = -2, right = 2;
			double min_radius = 0, max_radius = 2;
			
			std::uniform_real_distribution<> center(left, right);
			std::uniform_real_distribution<> radius(min_radius, max_radius);
			std::uniform_real_distribution<> lowk(2, 10);
			std::uniform_real_distribution<> highk(10, 80);
			std::uniform_real_distribution<> unit(0, 1);
	
			double lowk_proba = 0.8;
			double nested_proba = 0.2;
			double expand_ratio = 0.1;
	
			auto random_diel = [&]() {
				return unit(eng) <= lowk_proba ? lowk(eng) : highk(eng);
			};
			std::vector<dielectric_block> blocks;
			for (int i = 0; i < block_count; i++) {
				dielectric_block b = {center(eng), center(eng), center(eng), radius(eng), radius(eng), radius(eng), random_diel()};
				blocks.push_back(b);
				while (unit(eng) <= nested_proba) {
					blocks.push_back(b.expand(expand_ratio, random_diel()));
				}
			}
			if (blocks.size() > MAX_BLOCK_COUNT) {
				std::cout << "Warning: " << blocks.size() << " block count exceeds " << MAX_BLOCK_COUNT << ", truncating." << std::endl;
			}
			blocks.resize(MAX_BLOCK_COUNT);
			// background
			blocks.push_back({0, 0, 0, right, right, right, random_diel()});
			return blocks;
		}
		
	public:
		static constexpr int MAX_BLOCK_COUNT = 15;
		dielectric_generator(unsigned _seed = 42): seed(_seed) { eng.seed(_seed); }
		
		void iid_uniform(std::vector<ggft_solver::real> &DIEL) {
			std::uniform_real_distribution<> dielectric_value(1, 10);
			for (int i = 0; i < DIEL.size(); i++) {
				DIEL[i] = dielectric_value(eng);
			}
			return;
		}
	
		void random_generate(std::vector<ggft_solver::real> &DIEL, std::vector<ggft_solver::real> &STRUCTURE, int block_count = 10) {
			// auto blocks = random_blocks(block_count);
			auto blocks = random_blocks_practical(block_count);
	
			// Populate STRUCTURE with block data
			STRUCTURE.clear();
			for (const auto& block : blocks) {
				STRUCTURE.push_back(block.x);
				STRUCTURE.push_back(block.y);
				STRUCTURE.push_back(block.z);
				STRUCTURE.push_back(block.x_radius);
				STRUCTURE.push_back(block.y_radius);
				STRUCTURE.push_back(block.z_radius);
				STRUCTURE.push_back(block.value);
			}
	
			double cube_radius = 1;
			for (int i = 0; i < index::blockN; i++) {
				double x = cube_radius*(2.*i+1.)/index::blockN - cube_radius;
				for (int j = 0; j < index::blockN; j++) {
					double y = cube_radius*(2.*j+1.)/index::blockN - cube_radius;
					for (int k = 0; k < index::blockN; k++) {
						double z = cube_radius*(2.*k+1.)/index::blockN - cube_radius;
						for (auto b : blocks) {
							if (b.contain(x, y, z)) {
								DIEL[i + j * index::blockN + k * index::blockNN] = b.value;
								// std::cout << b.value << " ";
								break;
							}
						}
					}
					// std::cout << std::endl;
				}
				// std::cout << std::endl;
			}
		}
	
	};
	
};