# EdgesCGALWeightedDelaunay3D

Extraction rapide du **1‑squelette** de la triangulation régulière 3D (Delaunay pondéré) avec **CGAL**.

- **Entrée**: deux fichiers binaires `.npy` :
  - `points.npy` de taille `(N,3)` contenant les coordonnées 3D.
  - `weights.npy` de taille `(N,)` contenant les poids de puissance (`w = r^2`).
- **Sortie**: `edges.npy`, tableau `(M,2)` d'indices `uint32` triés des arêtes finies.

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
./build/EdgesCGALWeightedDelaunay3D points.npy weights.npy out_edges.npy
# Parallélisme (si TBB trouvé + tbbmalloc présent) :
CGAL_NTHREADS=$(nproc) ./build/EdgesCGALWeightedDelaunay3D points.npy weights.npy out_edges.npy
```

## Données de test
`data/example.xyzw` contient 1000 points uniformes dans `[0,1]^3` avec des poids plausibles (`w=r^2`, `r~U(0,0.02)`).
Pour l'utiliser avec le nouveau format, convertissez-le au préalable en `.npy`.

## Remarques
- Si `tbb` est trouvé **sans** `tbbmalloc`, le build **désactive** la voie parallèle pour éviter les erreurs de linkage (`scalable_*`).  
- Insertion **par lot** des points avec `vertex->info()` renseigné (indice d'origine), puis itération sur `finite_edges_*`.
