// CPU expert checks on real, possibly sharded GGUF weights. No GPU allocations.
#include "runtime.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
using namespace strata;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    poly::Shards shards(argv[1]);
    poly::Workers workers(12);
    std::set<std::pair<int,int>> seen;
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, .1f);
    double worst = 0;
    int cases = 0;
    shards.each([&](const TensorInfo& gate) {
        const auto at = gate.name.find(".ffn_gate_exps.weight");
        if (at == std::string::npos || gate.shape.size() != 3) return;
        const auto prefix = gate.name.substr(0, at);
        const auto* up = shards.find(prefix + ".ffn_up_exps.weight");
        const auto* down = shards.find(prefix + ".ffn_down_exps.weight");
        if (!up || !down || up->type != gate.type || !seen.emplace(gate.type, down->type).second) return;
        const int embd = gate.shape[0], ff = gate.shape[1], ne = gate.shape[2];
        const size_t gb = shards.payload(gate) / ne, db = shards.payload(*down) / ne;
        for (int rows = 1; rows <= 8; ++rows) for (bool gelu : {false,true}) for (float limit : {INFINITY, .01f}) {
            std::vector<float> x((size_t) rows * embd), out(x.size()), ref(x.size());
            for (auto& v : x) v = normal(rng);
            std::vector<poly::CpuExpert> e;
            for (int t = 0; t < rows; ++t) for (int k = 0; k < 3; ++k) {
                // Two experts shared by every token; one varies with its token.
                const int id = k < 2 ? k : (2+t)%ne;
                e.push_back({shards.data(gate)+id*gb, shards.data(*up)+id*gb,
                             shards.data(*down)+id*db, 1.f/3, t});
            }
            poly::cpu_experts(workers,e.data(),e.size(),rows,gate.type,down->type,embd,ff,x.data(),limit,gelu,out.data());
            for (int t = 0; t < rows; ++t) {
                auto solo = std::vector<poly::CpuExpert>(e.begin()+3*t,e.begin()+3*t+3);
                for (auto& v: solo) v.row=0;
                poly::cpu_experts(workers,solo.data(),3,1,gate.type,down->type,embd,ff,
                                  x.data()+(size_t)t*embd,limit,gelu,ref.data()+(size_t)t*embd);
            }
            double num=0,den=0;
            for(size_t i=0;i<out.size();++i){num+=std::pow(double(out[i])-ref[i],2);den+=double(ref[i])*ref[i];}
            const double err=std::sqrt(num/std::max(den,1e-30));
            if(!std::isfinite(err) || err>.005) throw std::runtime_error("CPU IQ relative error above .005");
            worst=std::max(worst,err); ++cases;
        }
        std::printf("types %u/%u, geometry %d/%d checked\n",gate.type,down->type,embd,ff);
    });
    if (!cases) return 2;
    std::printf("CPU IQ: %d cases passed, max relative L2 error %.8g\n",cases,worst);
}
