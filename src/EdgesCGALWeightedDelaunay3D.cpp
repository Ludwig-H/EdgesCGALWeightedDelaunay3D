// EdgesCGALWeightedDelaunay3D.cpp
// Fast extraction of the 1-skeleton of the 3D Regular Triangulation (Weighted Delaunay) using CGAL.
// Input: .xyzw with lines "x y z w" (w = CGAL power weight = squared radius).
// Output: lines "i j" with 0-based indices of endpoints of each finite edge.
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
#include <fstream>
#include <iostream>
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

static inline bool parse_xyzw(const char* s, double& x, double& y, double& z, double& w) {
    while (*s == ' ' || *s == '\t') ++s;
    if (*s == '\0' || *s == '\n' || *s == '\r' || *s == '#') return false;
    char* endp;
    x = std::strtod(s, &endp); if (endp == s) return false; s = endp;
    y = std::strtod(s, &endp); if (endp == s) return false; s = endp;
    z = std::strtod(s, &endp); if (endp == s) return false; s = endp;
    w = std::strtod(s, &endp); if (endp == s) return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "Usage: %s input.xyzw output.edges\n"
            "Each input line: x y z w (floats). w is the CGAL power weight (squared radius).\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];

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

    std::FILE* fin = std::fopen(in_path, "rb");
    if (!fin) { std::perror("fopen(input)"); return 2; }

    std::vector<std::pair<Weighted_point, uint32_t>> pts;
    pts.reserve(1<<20);
    BBox bb;

    // bulk read
    {
        const size_t BUFSZ = size_t(1) << 20;
        std::vector<char> buf(BUFSZ);
        std::string acc; acc.reserve(BUFSZ);
        while (true) {
            size_t n = std::fread(buf.data(), 1, BUFSZ, fin);
            if (n == 0) break;
            for (size_t i = 0; i < n; ++i) {
                char c = buf[i];
                if (c == '\r') continue;
                if (c == '\n') {
                    double x,y,z,w;
                    acc.push_back('\0');
                    if (parse_xyzw(acc.c_str(), x,y,z,w)) {
                        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w)) {
                            bb.update(x,y,z);
                            pts.emplace_back(Weighted_point(Bare_point(x,y,z), K::FT(w)),
                                             (uint32_t)pts.size());
                        }
                    }
                    acc.clear();
                } else acc.push_back(c);
            }
        }
        if (!acc.empty()) {
            double x,y,z,w;
            acc.push_back('\0');
            if (parse_xyzw(acc.c_str(), x,y,z,w)) {
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w)) {
                    bb.update(x,y,z);
                    pts.emplace_back(Weighted_point(Bare_point(x,y,z), K::FT(w)),
                                     (uint32_t)pts.size());
                }
            }
        }
    }
    std::fclose(fin);

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
    std::FILE* fout = std::fopen(out_path, "wb");
    if (!fout) { std::perror("fopen(output)"); return 3; }
    const size_t OUTBUFSZ = size_t(1) << 20;
    std::vector<char> filebuf(OUTBUFSZ);
    setvbuf(fout, filebuf.data(), _IOFBF, filebuf.size());

    std::vector<char> outbuf;
    outbuf.reserve(OUTBUFSZ);
    auto flush_buf = [&](){ if (!outbuf.empty()) { std::fwrite(outbuf.data(),1,outbuf.size(),fout); outbuf.clear(); } };

    size_t edge_count = 0;
    for (auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit) {
        auto c = eit->first;
        int i = eit->second, j = eit->third;
        auto vi = c->vertex(i);
        auto vj = c->vertex(j);
        if (rt.is_infinite(vi) || rt.is_infinite(vj)) continue;
        uint32_t a = vi->info(), b = vj->info();
        if (a == b) continue;
        if (a > b) std::swap(a,b);
        char line[64];
        int len = std::snprintf(line, sizeof(line), "%u %u\n", a, b);
        if (outbuf.size() + (size_t)len > OUTBUFSZ) flush_buf();
        outbuf.insert(outbuf.end(), line, line + len);
        ++edge_count;
    }
    flush_buf();
    std::fclose(fout);
    std::fprintf(stderr, "[info] Wrote %zu edges to %s\n", edge_count, out_path);
    return 0;
}
