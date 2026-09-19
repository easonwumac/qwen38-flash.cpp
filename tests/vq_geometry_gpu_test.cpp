#include "qwen38/mlx_backend.hpp"
#include "qwen38/quantization_geometry.hpp"
#include "../src/vq_metal_kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace qwen38;

namespace {
struct Bank {
    MlxArray codes, book, scales;
    std::vector<float> dense;
};

Bank bank(int in, int out, int experts, int d, int bits, int seed) {
    const int k = d == 8 ? 16384 : 256;
    const int nsub = in / d;
    const int words = bits ? static_cast<int>(packed_vq_word_count(nsub, bits)) : nsub;
    std::vector<std::int32_t> codes(experts * out * words);
    std::vector<float> book(k * d), scales(experts * out * (in / 64));
    std::vector<float> dense(experts * out * in);
    for (int i = 0; i < k * d; ++i) book[i] = ((i * 17 + seed) % 31 - 15) / 64.F;
    for (int i = 0; i < static_cast<int>(scales.size()); ++i)
        scales[i] = (1 + (i * 7 + seed) % 4) / 16.F;
    for (int r = 0; r < experts * out; ++r) {
        for (int s = 0; s < nsub; ++s) {
            const std::uint32_t code = (r * 73 + s * 251 + seed) % k;
            if (!bits) codes[r * words + s] = static_cast<std::int32_t>(code);
            else {
                const int bit = (s % 32) * bits, w = (s / 32) * bits + bit / 32;
                const std::uint64_t shifted = static_cast<std::uint64_t>(code) << (bit % 32);
                auto low = std::bit_cast<std::uint32_t>(codes[r * words + w]);
                codes[r * words + w] = std::bit_cast<std::int32_t>(low | static_cast<std::uint32_t>(shifted));
                if (bit % 32 + bits > 32) {
                    low = std::bit_cast<std::uint32_t>(codes[r * words + w + 1]);
                    codes[r * words + w + 1] = std::bit_cast<std::int32_t>(low | static_cast<std::uint32_t>(shifted >> 32));
                }
            }
            for (int e = 0; e < d; ++e)
                dense[r * in + s * d + e] = book[code * d + e] *
                    scales[r * (in / 64) + (s * d + e) / 64];
        }
    }
    return {MlxArray::from_int32(codes, std::array<int,3>{experts,out,words}).astype(bits ? MLX_UINT32 : MLX_UINT8),
        MlxArray::from_float32(book, std::array<int,2>{k,d}).astype(MLX_FLOAT16),
        MlxArray::from_float32(scales, std::array<int,3>{experts,out,in/64}).astype(MLX_FLOAT16),
        std::move(dense)};
}

void near(const MlxArray& actual, const MlxArray& expected, const char* label) {
    const auto a = actual.astype(MLX_FLOAT32).to_float32();
    const auto b = expected.astype(MLX_FLOAT32).to_float32();
    if (a.size() != b.size()) throw std::runtime_error("shape mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > 0.0005F)
            throw std::runtime_error(std::string(label) + " mismatch at " + std::to_string(i) +
                ": " + std::to_string(a[i]) + " vs " + std::to_string(b[i]));
}

void segmented(int d, int bits, int tile) {
    constexpr int in = 640, out = 32;
    auto gate = bank(in,out,2,d,bits,3), up = bank(in,out,2,d,bits,11);
    std::vector<float> input(tile * in);
    std::vector<std::int32_t> rowmap(tile);
    for (int i = 0; i < tile * in; ++i) input[i] = ((i * 13) % 17 - 8) / 32.F;
    for (int i = 0; i < tile; ++i) rowmap[i] = (i * 7) % tile;
    const auto x = MlxArray::from_float32(input, std::array<int,2>{tile,in}).astype(MLX_FLOAT16);
    const auto rows = MlxArray::from_int32(rowmap, std::array<int,1>{tile});
    // Two experts; exercise a nearly full tile and a one-row tail, with no
    // padded output allocation. Incorrect tail writes will overlap/go past it.
    const auto meta = MlxArray::from_int32(std::array<std::int32_t,6>{1,0,tile-1,0,tile-1,1},
        std::array<int,1>{6});
    const auto expected = [&](const Bank& b) {
        std::vector<float> output(tile * out);
        for (int r=0;r<tile;++r) for(int o=0;o<out;++o) {
            float value=0;
            for(int c=0;c<in;++c)
                value += input[rowmap[r]*in+c] * b.dense[((r<tile-1 ? 1:0)*out+o)*in+c];
            output[r*out+o]=value;
        }
        return MlxArray::from_float32(output,std::array<int,2>{tile,out}).astype(MLX_FLOAT16);
    };
    const std::array<MlxMetalIntTemplate,7> ints{{{"OUT",out},{"IN",in},{"NGRP",in/64},
        {"RTILE",tile},{"OTILE",32},{"D",d},{"BITS",bits}}};
    const std::array<MlxMetalDtypeTemplate,0> types{};
    const std::array<MlxMetalOutputSpec,1> outputs{{{.shape={tile,out},.dtype=MLX_FLOAT16}}};
    const char* names[]{"codes","codebook","scales","xsrc","srcrows","tmeta"};
    const MlxMetalKernel gemm("vq_geometry_segmented", names,"y",vq_metal::gemmseg_d8,vq_metal::header);
    const MlxArray* inputs[]{&gate.codes,&gate.book,&gate.scales,&x,&rows,&meta};
    const auto g=expected(gate), u=expected(up);
    const std::array<int,3> grid{32,8,1}, group{32,4,1};
    near(gemm.apply(inputs,outputs,grid,group,types,ints).front(),g,"segmented CPU oracle");
    const char* twin_names[]{"gate_codes","gate_codebook","gate_scales",
        "up_codes","up_codebook","up_scales","xsrc","srcrows","tmeta"};
    const MlxMetalKernel twin("vq_geometry_gate_up",twin_names,"h",vq_metal::gemmseg_gate_up_d8,vq_metal::header);
    const MlxArray* twin_inputs[]{&gate.codes,&gate.book,&gate.scales,&up.codes,&up.book,&up.scales,&x,&rows,&meta};
    near(twin.apply(twin_inputs,outputs,grid,group,types,ints).front(),
        MlxArray::multiply(g.silu(),u),"segmented SwiGLU oracle");
}

void down(int d, int bits, int batch, mlx_dtype type, bool add_shared) {
    constexpr int in=640,out=32,slots=10;
    auto weights=bank(in,out,12,d,bits,19);
    std::vector<float> xv(batch*slots*in), rv(batch*slots);
    std::vector<std::int32_t> ev(batch*slots);
    for(int i=0;i<static_cast<int>(xv.size());++i) xv[i]=((i*11)%29-14)/32.F;
    for(int i=0;i<batch*slots;++i) {rv[i]=(i%7+1)/64.F;ev[i]=(i*7)%12;}
    const auto x=MlxArray::from_float32(xv,std::array<int,2>{batch*slots,in}).astype(type);
    const auto routes=MlxArray::from_float32(rv,std::array<int,2>{batch,slots});
    const auto experts=MlxArray::from_int32(ev,std::array<int,2>{batch,slots});
    const auto shared=MlxArray::from_float32(
        std::span<const float>(xv.data(), batch*out),std::array<int,2>{batch,out}).astype(type);
    const char* all_names[]{"x","codes","codebook","cb_scales","cb_biases","scales","experts","route_weights","shared"};
    const auto names=std::span(all_names).first(add_shared ? 9 : 8);
    const std::string prefix=add_shared ? "#define QWEN38_VQ_ADD_SHARED 1\n" : "";
    const MlxMetalKernel packed("vq_geometry_down_pack",names,"y",
        prefix+std::string(vq_metal::down_reduce),vq_metal::header);
    const MlxMetalKernel scalar("vq_geometry_down_scalar",names,"y",
        "#define QWEN38_VQ_TEST_SCALAR_DOWN 1\n"+prefix+std::string(vq_metal::down_reduce),vq_metal::header);
    const MlxArray* all_inputs[]{&x,&weights.codes,&weights.book,&weights.book,&weights.book,&weights.scales,&experts,&routes,&shared};
    const auto inputs=std::span(all_inputs).first(add_shared ? 9 : 8);
    const std::array<MlxMetalIntTemplate,8> ints{{{"OUT",out},{"IN",in},{"D",d},{"GROUP",64},
        {"BITS",bits},{"SLOTS",slots},{"BATCH",batch},{"CBQ",0}}};
    const std::array<MlxMetalDtypeTemplate,1> types{{{"T",type}}};
    const std::array<MlxMetalOutputSpec,1> outputs{{{.shape={batch,out},.dtype=type}}};
    const std::array<int,3> grid{batch*out/8*256,1,1}, group{256,1,1};
    auto packed_output=packed.apply(inputs,outputs,grid,group,types,ints);
    auto scalar_output=scalar.apply(inputs,outputs,grid,group,types,ints);
    auto a=std::move(packed_output.front());
    auto b=std::move(scalar_output.front());
    const auto av=a.astype(MLX_FLOAT32).to_float32(),bv=b.astype(MLX_FLOAT32).to_float32();
    for(std::size_t i=0;i<av.size();++i)
        if(!std::isfinite(av[i]) || std::bit_cast<std::uint32_t>(av[i])!=std::bit_cast<std::uint32_t>(bv[i]))
            throw std::runtime_error("slot-packed down differs from scalar bits");
}
} // namespace

int main() {
    try {
        for(const int tile:{8,16,24}) {
            segmented(2,0,tile); segmented(4,8,tile); segmented(8,14,tile);
        }
        for(const int batch:{1,2,5}) for(const auto type:{MLX_FLOAT16,MLX_BFLOAT16}) {
            for(const bool shared:{false,true}) {
                down(4,8,batch,type,shared); down(8,14,batch,type,shared);
            }
        }
        std::cout << "9 segmented geometry/tail oracles and 24 slot-packed down parity cases passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
