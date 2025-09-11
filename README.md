# EdgesCGALWeightedDelaunay3D

Extraction rapide du **1‑squelette** de la triangulation régulière 3D (Delaunay pondéré) avec **CGAL**.

- **Entrée**: `.xyzw` où chaque ligne contient `x y z w` (float). `w` est le **poids de puissance** de CGAL (si vous pensez en rayon `r`, utilisez `w = r^2`).
- **Sortie**: texte, une ligne par arête finie: `i j` (indices 0‑based des points).

## Dépendances (Ubuntu 22.04/24.04)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcgal-dev libtbb-dev libtbbmalloc2 libgmp-dev libmpfr-dev
```
> `tbbmalloc` est explicitement linké pour éviter les symboles non résolus `scalable_malloc/free`.

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Utilisation
```bash
./build/EdgesCGALWeightedDelaunay3D data/example.xyzw out.edges
# Parallélisme (si TBB trouvé + tbbmalloc présent) :
CGAL_NTHREADS=$(nproc) ./build/EdgesCGALWeightedDelaunay3D data/example.xyzw out.edges
```

## Données de test
`data/example.xyzw` contient 1000 points uniformes dans `[0,1]^3` avec des poids plausibles (`w=r^2`, `r~U(0,0.02)`).

## Remarques
- Si `tbb` est trouvé **sans** `tbbmalloc`, le build **désactive** la voie parallèle pour éviter les erreurs de linkage (`scalable_*`).  
- Insertion **par lot** des points avec `vertex->info()` renseigné (indice d'origine), puis itération sur `finite_edges_*`.
