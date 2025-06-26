import multiprocessing
import warnings
from typing import Optional

import numpy as np
import rpy2.robjects as robjects
import torch
from rpy2.rinterface import NULL, RRuntimeWarning
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, isinstalled
from torch_geometric.utils import dense_to_sparse

from graphs_transformations.utils import get_radius_for_rec

warnings.filterwarnings("ignore", category=RRuntimeWarning)

numpy2ri.activate()


class Ts2Net:
    def __init__(self):
        robjects.r("options(warn = -1)")
        self.utils = importr("utils")
        self._ensure_ts2net_installed()
        self.r_ts2net = importr("ts2net")
        self._suppress_warnings("library(ts2net)")
        self._ensure_dependencies_installed()
        self._suppress_warnings("library(utils)")
        self._suppress_warnings("library(base)")

    def _ensure_ts2net_installed(self):
        # packages = ("remote", "quantmod", "tseries", "nonlinearTseries", "ts2net")
        # names_to_install = [name for name in packages if not isinstalled(name)]
        # if len(names_to_install) > 0:
        #     self.utils.install_packages(StrVector(names_to_install))
        if not isinstalled("ts2net"):
            print("Installing 'ts2net'")
            if not robjects.r("requireNamespace('remotes', quietly=TRUE)"):
                self._suppress_warnings('install.packages("remotes")')
            self._suppress_warnings(
                "remotes::install_github('lnferreira/ts2net', dependencies = TRUE)"
            )
        else:
            print("'ts2net' already installed")

    def _ensure_dependencies_installed(self):
        if not isinstalled("nonlinearTseries"):
            print("Installing 'nonlinearTseries")
            self.utils.install_packages("nonlinearTseries")

        return importr("nonlinearTseries")

    def ts_dist(
        self,
        ts_list,
        dist_func,
        is_symetric,
        error_values,
        warn_error,
        num_cores,
        **kwargs,
    ) -> np.ndarray:
        if self.r_ts2net is None:
            raise RuntimeError(
                "ts2net was not loaded, ts_dist function is not available"
            )
        dist = self.r_ts2net.ts_dist(
            ts_list,
            dist_func,
            is_symetric,
            error_values,
            warn_error,
            num_cores,
            **kwargs,
        )
        return np.array(dist)

    def tsdist_cor(self, ts1, ts2, cor_type="abs"):
        if self.r_ts2net is None:
            raise RuntimeError(
                "ts2net was not loaded, ts_dist function is not available"
            )
        corr = self.r_ts2net.tsdist_cor(ts1, ts2, cor_type)
        return float(corr[0])

    def tsnet_vg(
        self,
        x: torch.Tensor,
        method: str = "nvg",
        directed: bool = False,
        sparse: bool = True,
        weighted: bool = False,
        limit: Optional[int | object] = None,
        num_cores: Optional[int] = None,
    ):
        if self.r_ts2net is None:
            raise RuntimeError(
                "ts2net was not loaded, tsnet_vg function is not available"
            )
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        x_np = x.detach().numpy().flatten()
        r_data = robjects.FloatVector(x_np)
        limit = limit if limit is not None else robjects.r("Inf")
        net = self.r_ts2net.tsnet_vg(
            r_data, method, directed, limit, num_cores=num_cores
        )
        edge_index, edge_weight = self._get_adjacency_matrix(net, sparse, weighted)
        return edge_index, edge_weight

    def chunked_tsnet_vg(
        self,
        x: torch.Tensor,
        window_size: int = 32,
        stride: int = 1,
        method: str = "nvg",
        directed: bool = False,
        sparse: bool = True,
        weighted: bool = False,
        limit: Optional[int | object] = None,
        num_cores: Optional[int] = None,
    ):
        time_steps = x.size(0)
        all_edges, all_weights = [], []

        for start in range(0, time_steps, stride):
            end = min(start + window_size, time_steps)
            if end - start < 3:
                continue

            ei, ew = self.tsnet_vg(
                x=x[start:end],
                method=method,
                directed=directed,
                limit=limit,
                num_cores=num_cores,
            )

            ei += start  # shift to global indices
            all_edges.append(ei)
            all_weights.append(ew)

        if not all_edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros((0,), dtype=torch.float)
            return edge_index, edge_weight

        # concatenation then deduplication
        edge_index = torch.cat(all_edges, dim=1).T
        edge_weight = torch.cat(all_weights)

        edge_index_sorted, _ = torch.sort(edge_index, dim=1)
        uniq, idx = torch.unique(edge_index_sorted, dim=0, return_inverse=True)

        # aggregate weights (mean) per unique edge
        if edge_weight.numel():
            edge_weight = (
                torch.zeros(uniq.size(0)).index_add(0, idx, edge_weight)
                / torch.bincount(idx).float()
            )
        edge_index = uniq.T

        return edge_index, edge_weight

    def tsnet_rn(
        self,
        x: torch.Tensor,
        radius: float,
        embedding_dim: Optional[int] = None,
        time_lag: int = 1,
        sparse: bool = False,
        weighted: bool = False,
        do_plot: bool = False,
        **kwargs,
    ):
        lib = self._ensure_dependencies_installed()
        x = x.squeeze(-1)
        x_np = x.detach().numpy().flatten()
        r_data = robjects.FloatVector(x_np)
        if embedding_dim is None:
            dim = lib.estimateEmbeddingDim(r_data, time_lag=time_lag, do_plot=do_plot)
            if isinstance(dim, robjects.vectors.BoolVector):
                embedding_dim = min(10, max(3, x.shape[0] // 10))
            else:
                embedding_dim = int(dim.item())
        radius = get_radius_for_rec(
            x=x, alpha=radius, dim=embedding_dim, time_delay=time_lag
        )
        # r_data = robjects.FloatVector(x)
        net = self.r_ts2net.tsnet_rn(
            r_data, radius, embedding_dim, time_lag, do_plot, **kwargs
        )
        edge_index, edge_weight = self._get_adjacency_matrix(net, sparse, weighted)
        return edge_index, edge_weight

    def tsnet_qn(
        self,
        x: torch.Tensor,
        breaks,
        weights_as_prob: bool = True,
        remove_loops: bool = False,
        sparse: bool = False,
        weighted: bool = False,
        **kwargs,
    ):
        x = x.squeeze(-1)
        x_np = x.detach().numpy().flatten()
        r_data = robjects.FloatVector(x_np)
        net = self.r_ts2net.tsnet_qn(
            r_data, breaks, weights_as_prob, remove_loops, **kwargs
        )
        edge_index, edge_weight = self._get_adjacency_matrix(net, sparse, weighted)
        print(f"{edge_index.shape=}")
        return edge_index, edge_weight

    def _suppress_warnings(self, expr: str):
        """Run an R command with warnings suppressed."""
        return robjects.r(f"suppressWarnings({expr})")

    def _get_adjacency_matrix(self, graph, sparse=False, weighted=False):
        """Retrieve the adjacency matrix from an igraph object."""
        # Extract adjacency matrix from R as dense matrix
        robjects.r("""
            get_adj_matrix <- function(graph, sparse, attr) {
                if (sparse) {
                    as.matrix(as_adjacency_matrix(graph, sparse = TRUE, attr = attr))
                } else {
                    as.matrix(as_adjacency_matrix(graph, sparse = FALSE, attr = attr))
                }
            }
        """)
        attr = "weight" if weighted else NULL
        adj_matrix = robjects.r("get_adj_matrix")(graph, sparse, attr)
        adj_matrix_tensor = torch.tensor(np.asarray(adj_matrix), dtype=torch.float32)

        return dense_to_sparse(adj_matrix_tensor)
