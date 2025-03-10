import warnings

import numpy as np
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, isinstalled

warnings.filterwarnings("ignore", category=RRuntimeWarning)

numpy2ri.activate()


class Ts2Net:
    def __init__(self):
        self._suppress_warnings("library(ts2net)")
        self._suppress_warnings("library(utils)")
        self._suppress_warnings("library(base)")
        robjects.r("options(warn = -1)")
        self.utils = importr("utils")
        self._ensure_ts2net_installed()
        self.r_ts2net = importr("ts2net")

    def _ensure_ts2net_installed(self):
        # packages = ("remote", "quantmod", "tseries", "nonlinearTseries", "ts2net")
        # names_to_install = [name for name in packages if not isinstalled(name)]
        # if len(names_to_install) > 0:
        #     self.utils.install_packages(StrVector(names_to_install))
        if not isinstalled("ts2net"):
            print("Installing 'ts2net'")
            self._suppress_warnings("""
                if (!requireNamespace("remotes", quietly = TRUE)) {
                    install.packages("remotes")
                }
                remotes::install_github("lnferreira/ts2net", dependencies = TRUE)
            """)
        else:
            print("'ts2net' already installed")

        # # Define adjacency matrix function in R
        # robjects.r("""
        # get_adjacency_matrix <- function(graph, sparse = TRUE) {
        #     as.matrix(get_adjacency_matrix(graph, sparse = sparse))
        # }
        # """)

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
        self, x, method, directed=False, sparse=False, limit=np.inf, num_cores=1
    ):
        if self.r_ts2net is None:
            raise RuntimeError(
                "ts2net was not loaded, tsnet_vg function is not available"
            )
        net = self.r_ts2net.tsnet_vg(x, method, num_cores=num_cores)

        adj_matrix = self._get_adjacency_matrix(net, sparse)
        return adj_matrix

    def _suppress_warnings(self, expr: str):
        """Run an R command with warnings suppressed."""
        return robjects.r(f"suppressWarnings({expr})")

    def _get_adjacency_matrix(self, graph, sparse=False):
        """Retrieve the adjacency matrix from an igraph object."""
        # Extract adjacency matrix from R as dense matrix
        robjects.r("""
            get_adj_matrix <- function(graph, sparse) {
                if (sparse) {
                    as.matrix(as_adjacency_matrix(graph, sparse = TRUE))
                } else {
                    as.matrix(as_adjacency_matrix(graph, sparse = FALSE))
                }
            }
        """)
        adj_matrix = robjects.r("get_adj_matrix")(graph, sparse)
        return np.array(adj_matrix)


# Example usage
if __name__ == "__main__":
    api = Ts2Net()

    ts1 = np.random.rand(10, 100)
    ts2 = np.random.rand(10, 100)

    print(f"Correlation Distance: {api.tsdist_cor(ts1, ts2)}")

    print(f"Visibility Graph: {api.tsnet_vg(ts1, 'nvg')}")
