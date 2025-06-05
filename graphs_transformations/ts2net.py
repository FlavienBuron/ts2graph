import warnings
from typing import Optional

import numpy as np
import rpy2.robjects as robjects
import torch
from rpy2.rinterface import NULL, RRuntimeWarning
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, isinstalled
from torch_geometric.utils import dense_to_sparse

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
        num_cores: int = 1,
    ):
        if self.r_ts2net is None:
            raise RuntimeError(
                "ts2net was not loaded, tsnet_vg function is not available"
            )
        x_np = x.detach().numpy().flatten()
        print(f"{x.shape=} {x_np.shape=}")
        r_data = robjects.FloatVector(x_np)
        limit = limit if limit is not None else robjects.r("Inf")
        net = self.r_ts2net.tsnet_vg(
            r_data, method, directed, limit, num_cores=num_cores
        )
        edge_index, edge_weight = self._get_adjacency_matrix(net, sparse, weighted)
        return edge_index, edge_weight

    def tsnet_rn(
        self,
        x: np.ndarray,
        radius: float,
        embedding_dim: Optional[int] = None,
        time_lag: int = 1,
        sparse: bool = False,
        do_plot: bool = False,
        **kwargs,
    ):
        lib = self._ensure_dependencies_installed()
        embedding_dim = (
            embedding_dim
            if embedding_dim is not None
            else lib.estimateEmbeddingDim(x, time_lag=time_lag, do_plot=do_plot)
        )
        # r_data = robjects.FloatVector(x)
        net = self.r_ts2net.tsnet_rn(
            x, radius, embedding_dim, time_lag, do_plot, **kwargs
        )
        adj_matrix = self._get_adjacency_matrix(net, sparse)
        return adj_matrix

    def tsnet_qn(
        self,
        x: np.ndarray,
        breaks,
        weights_as_prob: bool = True,
        remove_loops: bool = False,
        sparse: bool = False,
        **kwargs,
    ):
        r_data = robjects.FloatVector(x)
        net = self.r_ts2net.tsnet_qn(
            r_data, breaks, weights_as_prob, remove_loops, **kwargs
        )
        adj_matrix = self._get_adjacency_matrix(net, sparse)
        return adj_matrix

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
        print(f"{adj_matrix.shape=}")
        adj_matrix_tensor = torch.tensor(np.asarray(adj_matrix), dtype=torch.float32)

        return dense_to_sparse(adj_matrix_tensor)


# Example usage
if __name__ == "__main__":
    api = Ts2Net()

    ts1 = np.random.rand(10, 100)

    print(f"Visibility Graph:\n {api.tsnet_vg(ts1, 'nvg').shape}.")
    print(f"Recurrent Graph:\n {api.tsnet_rn(ts1, 3.0).shape}")
    print(f"Quantile Graph:\n {api.tsnet_qn(ts1, ts1.shape[0]).shape}")
