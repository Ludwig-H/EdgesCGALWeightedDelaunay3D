// EdgesCGALWeightedDelaunay3D.cpp
// Fast extraction of the 1-skeleton of the 3D Regular Triangulation (Weighted Delaunay) using CGAL.
// Input: two .npy files: points.npy with shape (N,3) and weights.npy with shape (N,).
// Output: edges.npy containing Nx2 uint32_t pairs of sorted indices of each finite edge.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <limits>
#include <utility>
#include <algorithm>
#include <thread> // hardware_concurrency

#ifdef CGAL_LINKED_WITH_TBB
  #include <tbb/global_control.h>
#endif

struct BBox {
    double xmin =  std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();
    void update(double x, double y, double z) {
        xmin = std::min(xmin, x); ymin = std::min(ymin, y); zmin = std::min(zmin, z);
        xmax = std::max(xmax, x); ymax = std::max(ymax, y); zmax = std::max(zmax, z);
    }
    bool valid() const { return xmin <= xmax && ymin <= ymax && zmin <= zmax; }
};

static bool load_npy_double(const char* path, std::vector<double>& out,
                           size_t& rows, size_t& cols) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("fopen"); return false; }
    unsigned char magic[6];
    if (std::fread(magic,1,6,f)!=6 || magic[0]!=0x93 || std::memcmp(magic+1,"NUMPY",5)!=0) {
        std::fprintf(stderr, "%s: not a .npy file\n", path);
        std::fclose(f);
        return false;
    }
    unsigned char ver[2];
    if (std::fread(ver,1,2,f)!=2) { std::fclose(f); return false; }
    size_t header_len;
    if (ver[0] <= 1) {
        uint16_t hl;
        if (std::fread(&hl,2,1,f)!=1) { std::fclose(f); return false; }
        header_len = hl;
    } else {
        uint32_t hl;
        if (std::fread(&hl,4,1,f)!=1) { std::fclose(f); return false; }
        header_len = hl;
    }

    std::string header(header_len, '\0');
    if (std::fread(header.data(),1,header_len,f)!=header_len) { std::fclose(f); return false; }

    auto get_field = [&](const char* key) -> std::string {
        std::string k = std::string("'") + key + "'";
        size_t pos = header.find(k);
        if (pos == std::string::npos) return {};
        pos = header.find('\'', pos + k.size());
        if (pos == std::string::npos) return {};
        size_t end = header.find('\'', pos + 1);
        if (end == std::string::npos) return {};
        return header.substr(pos + 1, end - pos - 1);
    };

    std::string descr = get_field("descr");
    size_t word_size = 0;
    if (descr == "<f8") word_size = 8;
    else if (descr == "<f4") word_size = 4;
    else {
        std::fprintf(stderr, "%s: unsupported descr %s\n", path, descr.c_str());
        std::fclose(f);
        return false;
    }

    size_t pos_shape = header.find("shape");
    if (pos_shape == std::string::npos) { std::fclose(f); return false; }
    size_t lparen = header.find('(', pos_shape);
    size_t rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) {
        std::fclose(f); return false;
    }
    std::string shape_str = header.substr(lparen + 1, rparen - lparen - 1);
    std::vector<size_t> dims;
    size_t idx = 0;
    while (idx < shape_str.size()) {
        while (idx < shape_str.size() && shape_str[idx] == ' ') ++idx;
        size_t next = shape_str.find(',', idx);
        if (next == std::string::npos) next = shape_str.size();
        std::string token = shape_str.substr(idx, next - idx);
        if (!token.empty()) dims.push_back((size_t)std::strtoull(token.c_str(), nullptr, 10));
        idx = next + 1;
    }
    if (dims.empty()) { std::fclose(f); return false; }
    rows = dims[0];
    cols = dims.size() > 1 ? dims[1] : 1;
    size_t count = rows * cols;

    out.resize(count);
    if (descr == "<f8") {
        if (std::fread(out.data(), word_size, count, f) != count) { std::fclose(f); return false; }
    } else {
        std::vector<float> tmp(count);
        if (std::fread(tmp.data(), word_size, count, f) != count) { std::fclose(f); return false; }
        for (size_t i = 0; i < count; ++i) out[i] = tmp[i];
    }
    std::fclose(f);
    return true;
}

static bool save_npy_u32_2col(const char* path, const std::vector<uint32_t>& data) {
    size_t rows = data.size() / 2;
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::perror("fopen"); return false; }
    unsigned char magic[6] = {0x93,'N','U','M','P','Y'};
    unsigned char ver[2] = {1,0};
    std::string header = "{'descr': '<u4', 'fortran_order': False, 'shape': (" +
        std::to_string(rows) + ", 2), }";
    while ((10 + header.size()) % 16 != 0) header.push_back(' ');
    header.back() = '\n';
    uint16_t hlen = (uint16_t)header.size();
    std::fwrite(magic,1,6,f);
    std::fwrite(ver,1,2,f);
    std::fwrite(&hlen,2,1,f);
    std::fwrite(header.data(),1,header.size(),f);
    std::fwrite(data.data(), sizeof(uint32_t), data.size(), f);
    std::fclose(f);
    return true;
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "Usage: %s points.npy weights.npy output.npy\n", argv[0]);
        return 1;
    }
    const char* pts_path = argv[1];
    const char* w_path  = argv[2];
    const char* out_path = argv[3];

    using K   = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb0 = CGAL::Regular_triangulation_vertex_base_3<K>;
    using Vb  = CGAL::Triangulation_vertex_base_with_info_3<uint32_t, K, Vb0>;
    using Cb  = CGAL::Regular_triangulation_cell_base_3<K>;
    #ifdef CGAL_LINKED_WITH_TBB
      using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>;
    #else
      using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    #endif
    using Rt  = CGAL::Regular_triangulation_3<K, Tds>;
    using Bare_point = Rt::Bare_point;
    using Weighted_point = Rt::Weighted_point;

    #ifdef CGAL_LINKED_WITH_TBB
    {
        int threads = (int)std::thread::hardware_concurrency();
        if (const char* env = std::getenv("CGAL_NTHREADS")) {
            int req = std::atoi(env);
            if (req > 0) threads = req;
        }
        static tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
        std::fprintf(stderr, "[info] TBB enabled, using up to %d threads\n", threads);
    }
    #endif

    std::vector<double> coord_data;
    size_t nrows=0, ncols=0;
    if (!load_npy_double(pts_path, coord_data, nrows, ncols) || ncols != 3) {
        std::fprintf(stderr, "Failed to load %s or unexpected shape\n", pts_path);
        return 2;
    }
    std::vector<double> weight_data;
    size_t wrows=0, wcols=0;
    if (!load_npy_double(w_path, weight_data, wrows, wcols) || (wcols != 1 && wcols != 0) || wrows != nrows) {
        std::fprintf(stderr, "Failed to load %s or mismatched shape\n", w_path);
        return 2;
    }

    std::vector<std::pair<Weighted_point, uint32_t>> pts;
    pts.reserve(nrows);
    BBox bb;
    for (size_t i = 0; i < nrows; ++i) {
        double x = coord_data[3*i];
        double y = coord_data[3*i+1];
        double z = coord_data[3*i+2];
        double w = weight_data[i];
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w)) {
            bb.update(x,y,z);
            pts.emplace_back(Weighted_point(Bare_point(x,y,z), K::FT(w)), (uint32_t)pts.size());
        }
    }

    if (pts.size() < 2) {
        std::fprintf(stderr, "Input has < 2 valid points. Nothing to do.\n");
        std::FILE* fout = std::fopen(out_path, "wb");
        if (fout) std::fclose(fout);
        return 0;
    }
    std::fprintf(stderr, "[info] Loaded %zu points\n", pts.size());

    // Triangulation
    #ifdef CGAL_LINKED_WITH_TBB
    std::unique_ptr<Rt::Lock_data_structure> lock_holder;
    Rt::Lock_data_structure* lock_ptr = nullptr;
    if (bb.valid()) {
        CGAL::Bbox_3 box(bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax);
        lock_holder.reset(new Rt::Lock_data_structure(box, 64));
        lock_ptr = lock_holder.get();
    }
    Rt rt(Rt::Geom_traits(), lock_ptr);
    #else
    Rt rt;
    #endif

    rt.insert(pts.begin(), pts.end());
    if (!rt.is_valid(false))
        std::fprintf(stderr, "[warning] CGAL triangulation reports invalid structure.\n");

    // Write edges
    std::vector<uint32_t> edges;
    edges.reserve(rt.number_of_finite_edges()*2);
    for (auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit) {
        auto c = eit->first;
        int i = eit->second, j = eit->third;
        auto vi = c->vertex(i);
        auto vj = c->vertex(j);
        if (rt.is_infinite(vi) || rt.is_infinite(vj)) continue;
        uint32_t a = vi->info(), b = vj->info();
        if (a == b) continue;
        if (a > b) std::swap(a,b);
        edges.push_back(a);
        edges.push_back(b);
    }
    if (!save_npy_u32_2col(out_path, edges)) {
        std::fprintf(stderr, "Failed to write %s\n", out_path);
        return 3;
    }
    std::fprintf(stderr, "[info] Wrote %zu edges to %s\n", edges.size()/2, out_path);
    return 0;
}
