from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task

from typing import List, Union
import copy

import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import lab as B


# For dispatching with TF and PyTorch model types when they have not yet been loaded.
# See https://beartype.github.io/plum/types.html#moduletype


def create_empty_spatiotemporal_xarray(
    X: Union[xr.Dataset, xr.DataArray],
    dates: List,
    resolution_factor: Union[float, int] = 1.0,
    coord_names: dict = {"x1": "x1", "x2": "x2"},
    data_vars: List = ["var"],
    prepend_dims: List[str] = None,
    prepend_coords: dict = None,
):
    if prepend_dims is None:
        prepend_dims = []
    if prepend_coords is None:
        prepend_coords = {}

    # Check for any repeated data_vars
    if len(data_vars) != len(set(data_vars)):
        raise ValueError(
            f"Duplicate data_vars found in data_vars: {data_vars}. "
            "This woudld cause the xarray.Dataset to have fewer variables than expected."
        )

    x1_raw = X.coords[coord_names["x1"]]
    x2_raw = X.coords[coord_names["x2"]]

    # Assert uniform spacing
    if not np.allclose(np.diff(x1_raw), np.diff(x1_raw)[0]):
        raise ValueError(f"Coordinate {coord_names['x1']} must be uniformly spaced.")
    if not np.allclose(np.diff(x2_raw), np.diff(x2_raw)[0]):
        raise ValueError(f"Coordinate {coord_names['x2']} must be uniformly spaced.")

    if resolution_factor == 1.0:
        x1_predict = x1_raw
        x2_predict = x2_raw
    else:
        x1_predict = np.linspace(
            x1_raw[0],
            x1_raw[-1],
            int(x1_raw.size * resolution_factor),
            dtype="float32",
        )
        x2_predict = np.linspace(
            x2_raw[0],
            x2_raw[-1],
            int(x2_raw.size * resolution_factor),
            dtype="float32",
        )

    if len(prepend_dims) != len(set(prepend_dims)):
        # TODO unit test
        raise ValueError(
            f"Length of prepend_dims ({len(prepend_dims)}) must be equal to length of "
            f"prepend_coords ({len(prepend_coords)})."
        )

    dims = [*prepend_dims, "time", coord_names["x1"], coord_names["x2"]]
    coords = {
        **prepend_coords,
        "time": pd.to_datetime(dates),
        coord_names["x1"]: x1_predict,
        coord_names["x2"]: x2_predict,
    }

    pred_ds = xr.Dataset(
        {data_var: xr.DataArray(dims=dims, coords=coords) for data_var in data_vars}
    ).astype("float32")

    # Convert time coord to pandas timestamps
    pred_ds = pred_ds.assign_coords(time=pd.to_datetime(pred_ds.time.values))

    # TODO: Convert init time to forecast time?
    # pred_ds = pred_ds.assign_coords(
    #     time=pred_ds['time'] + pd.Timedelta(days=task_loader.target_delta_t[0]))

    return pred_ds


class ProbabilisticModel:

    """
    Base class for probabilistic model used for DeepSensor.
    Ensures a set of methods required for DeepSensor
    are implemented by specific model classes that inherit from it.
    """

    def mean(self, task: Task, *args, **kwargs):
        """
        Computes the model mean prediction over target points based on given context
        data.
        """
        raise NotImplementedError()

    def variance(self, task: Task, *args, **kwargs):
        """
        Model marginal variance over target points given context points.
        Shape (N,).
        """
        raise NotImplementedError()

    def stddev(self, task: Task):
        """
        Model marginal standard deviation over target points given context points.
        Shape (N,).
        """
        var = self.variance(task)
        return var**0.5

    def covariance(self, task: Task, *args, **kwargs):
        """
        Computes the model covariance matrix over target points based on given context
        data. Shape (N, N).
        """
        raise NotImplementedError()

    def mean_marginal_entropy(self, task: Task, *args, **kwargs):
        """
        Computes the mean marginal entropy over target points based on given context
        data.

        Note: Getting a vector of marginal entropies would be useful too.
        """
        raise NotImplementedError()

    def joint_entropy(self, task: Task, *args, **kwargs):
        """
        Computes the model joint entropy over target points based on given context
        data.
        """
        raise NotImplementedError()

    def logpdf(self, task: Task, *args, **kwargs):
        """
        Computes the joint model logpdf over target points based on given context
        data.
        """
        raise NotImplementedError()

    def loss(self, task: Task, *args, **kwargs):
        """
        Computes the model loss over target points based on given context data.
        """
        raise NotImplementedError()

    def sample(self, task: Task, n_samples=1, *args, **kwargs):
        """
        Draws `n_samples` joint samples over target points based on given context
        data.
        returned shape is (n_samples, n_target).
        """
        raise NotImplementedError()


class DeepSensorModel(ProbabilisticModel):

    """
    Implements DeepSensor prediction functionality of a ProbabilisticModel.
    Allows for outputting an xarray object containing on-grid predictions or a pandas
    object containing off-grid predictions.
    """

    def __init__(
        self, data_processor: DataProcessor = None, task_loader: TaskLoader = None
    ):
        """Initialise DeepSensorModel

        :param task_loader: TaskLoader object, used to determine target variables for unnormalising
        :param data_processor: DataProcessor object, used to unnormalise predictions
        """
        self.task_loader = task_loader
        self.data_processor = data_processor

    def predict(
        self,
        tasks: Union[List[Task], Task],
        X_t: Union[
            xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Index, np.ndarray
        ],
        X_t_normalised: bool = False,
        resolution_factor=1,
        n_samples=0,
        ar_sample=False,
        ar_subsample_factor=1,
        unnormalise=True,
        seed=0,
        append_indexes: dict = None,
        progress_bar=0,
        verbose=False,
    ):
        """Predict on a regular grid or at off-grid locations.

        TODO:
        - Test with multiple targets model

        :param tasks: List of tasks containing context data.
        :param X_t: Target locations to predict at. Can be an xarray object containing
            on-grid locations or a pandas object containing off-grid locations.
        :param X_t_normalised: Whether the `X_t` coords are normalised.
            If False, will normalise the coords before passing to model. Default False.
        :param resolution_factor: Optional factor to increase the resolution of the
            target grid by. E.g. 2 will double the target resolution, 0.5 will halve it.
            Applies to on-grid predictions only. Default 1.
        :param n_samples: Number of joint samples to draw from the model.
            If 0, will not draw samples. Default 0.
        :param ar_sample: Whether to use autoregressive sampling. Default False.
        :param unnormalise: Whether to unnormalise the predictions. Only works if
            `self` has a `data_processor` and `task_loader` attribute. Default True.
        :param seed: Random seed for deterministic sampling. Default 0.
        :param append_indexes: Dictionary of index metadata to append to pandas indexes
            in the off-grid case. Default None.
        :param progress_bar: Whether to display a progress bar over tasks. Default 0.
        :param verbose: Whether to print time taken for prediction. Default False.

        Returns:
            - If X_t is a pandas object, returns pandas objects containing off-grid predictions.
            - If X_t is an xarray object, returns xarray object containing on-grid predictions.
            - If n_samples == 0, returns only mean and std predictions.
            - If n_samples > 0, returns mean, std and samples predictions.
        """
        tic = time.time()

        if not isinstance(X_t, (xr.DataArray, xr.Dataset)):
            if resolution_factor != 1:
                raise ValueError(
                    "resolution_factor can only be used with on-grid predictions."
                )
        if not isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
            if append_indexes is not None:
                raise ValueError(
                    "append_indexes can only be used with off-grid predictions."
                )

        if type(tasks) is Task:
            tasks = [tasks]

        if n_samples >= 1:
            B.set_random_seed(seed)
            np.random.seed(seed)

        dates = [task["time"] for task in tasks]

        # Flatten tuple of tups to single list
        target_var_IDs = [
            var_ID for set in self.task_loader.target_var_IDs for var_ID in set
        ]

        if isinstance(X_t, pd.Index):
            X_t = pd.DataFrame(index=X_t)
        elif isinstance(X_t, np.ndarray):
            # Convert to empty dataframe with normalised or unnormalised coord names
            if X_t_normalised:
                index_names = ["x1", "x2"]
            else:
                index_names = self.data_processor.raw_spatial_coord_names
            X_t = pd.DataFrame(X_t.T, columns=index_names)
            X_t = X_t.set_index(index_names)

        if not X_t_normalised:
            X_t = self.data_processor.map_coords(X_t)  # Normalise

        if isinstance(X_t, (xr.DataArray, xr.Dataset)):
            mode = "on-grid"
        elif isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index)):
            mode = "off-grid"
            if append_indexes is not None:
                # Check append_indexes are all same length as X_t
                if append_indexes is not None:
                    for idx, vals in append_indexes.items():
                        if len(vals) != len(X_t):
                            raise ValueError(
                                f"append_indexes[{idx}] must be same length as X_t, got {len(vals)} and {len(X_t)} respectively."
                            )
                X_t = X_t.reset_index()
                X_t = pd.concat([X_t, pd.DataFrame(append_indexes)], axis=1)
                X_t = X_t.set_index(list(X_t.columns))
        else:
            raise ValueError(
                f"X_t must be an xarray object or a pandas object, not {type(X_t)}"
            )

        if mode == "on-grid":
            mean = create_empty_spatiotemporal_xarray(
                X_t, dates, resolution_factor, data_vars=target_var_IDs
            ).to_array(dim="data_var")
            std = create_empty_spatiotemporal_xarray(
                X_t, dates, resolution_factor, data_vars=target_var_IDs
            ).to_array(dim="data_var")
            if n_samples >= 1:
                samples = create_empty_spatiotemporal_xarray(
                    X_t,
                    dates,
                    resolution_factor,
                    data_vars=target_var_IDs,
                    prepend_dims=["sample"],
                    prepend_coords={"sample": np.arange(n_samples)},
                ).to_array(dim="data_var")

            X_t_arr = (mean["x1"].values, mean["x2"].values)

        elif mode == "off-grid":
            # Repeat target locs for each date to create multiindex
            idxs = [(date, *idxs) for date in dates for idxs in X_t.index]
            index = pd.MultiIndex.from_tuples(idxs, names=["time", *X_t.index.names])
            mean = pd.DataFrame(index=index, columns=target_var_IDs)
            std = pd.DataFrame(index=index, columns=target_var_IDs)
            if n_samples >= 1:
                idxs_samples = [
                    (sample, date, *idxs)
                    for sample in range(n_samples)
                    for date in dates
                    for idxs in X_t.index
                ]
                index_samples = pd.MultiIndex.from_tuples(
                    idxs_samples, names=["sample", "time", *X_t.index.names]
                )
                samples = pd.DataFrame(index=index_samples, columns=target_var_IDs)

            X_t_arr = X_t.reset_index()[["x1", "x2"]].values.T

        # Don't change tasks by reference when overriding target locations
        tasks = copy.deepcopy(tasks)

        for task in tqdm(tasks, position=0, disable=progress_bar < 1, leave=True):
            task["X_t"] = [X_t_arr for _ in range(len(task["X_t"]))]

            # If `DeepSensor` model child has been sub-classed with a `__call__` method,
            # we assume this is a distribution-like object that can be used to compute
            # mean, std and samples. Otherwise, run the model with `Task` for each prediction type.
            if hasattr(self, "__call__"):
                # Run model forwards once to generate output distribution, which we re-use
                dist = self(task, n_samples=n_samples)
                mean_arr = self.mean(dist)
                std_arr = self.stddev(dist)
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    if ar_sample:
                        samples_arr = self.ar_sample(
                            task,
                            n_samples=n_samples,
                            ar_subsample_factor=ar_subsample_factor,
                        )
                        samples_arr = samples_arr.reshape((n_samples, *mean_arr.shape))
                    else:
                        samples_arr = self.sample(dist, n_samples=n_samples)
            else:
                # Re-run model for each prediction type
                mean_arr = self.mean(task)
                std_arr = self.stddev(task)
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    if ar_sample:
                        samples_arr = self.ar_sample(
                            task,
                            n_samples=n_samples,
                            ar_subsample_factor=ar_subsample_factor,
                        )
                        samples_arr = samples_arr.reshape((n_samples, *mean_arr.shape))
                    else:
                        samples_arr = self.sample(task, n_samples=n_samples)

            # Concatenate multi-target predictions
            if isinstance(mean_arr, (list, tuple)):
                mean_arr = np.concatenate(mean_arr, axis=0)
                std_arr = np.concatenate(std_arr, axis=0)
                if n_samples >= 1:
                    samples_arr = np.concatenate(samples_arr, axis=0)

            if mode == "on-grid":
                mean.loc[:, task["time"], :, :] = mean_arr
                std.loc[:, task["time"], :, :] = std_arr
                if n_samples >= 1:
                    for sample_i in range(n_samples):
                        samples.loc[:, sample_i, task["time"], :, :] = samples_arr[
                            sample_i
                        ]
            elif mode == "off-grid":
                # TODO multi-target case
                mean.loc[task["time"]] = mean_arr.T
                std.loc[task["time"]] = std_arr.T
                if n_samples >= 1:
                    for sample_i in range(n_samples):
                        samples.loc[sample_i, task["time"]] = samples_arr[sample_i].T

        if mode == "on-grid":
            mean = mean.to_dataset(dim="data_var")
            std = std.to_dataset(dim="data_var")
            if n_samples >= 1:
                samples = samples.to_dataset(dim="data_var")

        if (
            self.task_loader is not None
            and self.data_processor is not None
            and unnormalise == True
        ):
            mean = self.data_processor.unnormalise(mean)
            std = self.data_processor.unnormalise(std, add_offset=False)
            if n_samples >= 1:
                samples = self.data_processor.unnormalise(samples)

        if verbose:
            dur = time.time() - tic
            print(f"Done in {np.floor(dur / 60)}m:{dur % 60:.0f}s.\n")

        if n_samples >= 1:
            return mean, std, samples
        else:
            return mean, std
