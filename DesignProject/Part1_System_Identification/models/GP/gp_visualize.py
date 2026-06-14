"""

Visualizations: 
1. One-step prediction with uncertainty band
2. Free-run simulation plot
3. Residual/error plot
4. Residual histogram
5. GP predictive standard deviation over time
6. Kernel ARD length-scale bar plot
7. 1D GP posterior slice with sampled functions
8. 2D GP mean heatmap slice

 usage :
    python SystemModelling/gp_visualize.py \
        --model SystemModelling/gp_narx_outputs/gp_narx_bundle_na6_nb6.joblib \
        --project-root . \
        --out-dir SystemModelling/gp_narx_visualizations



"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error



def create_io_data(u, th, na, nb):
    """
    Build NARX input-output data.

    X[k] = [u[k-nb], ..., u[k-1], th[k-na], ..., th[k-1]]
    Y[k] = th[k]
    """
    u = np.asarray(u).reshape(-1)
    th = np.asarray(th).reshape(-1)

    if len(u) != len(th):
        raise ValueError("u and th must have the same length.")

    X = []
    Y = []

    start = max(na, nb)
    for k in range(start, len(th)):
        xk = np.concatenate([u[k - nb:k], th[k - na:k]])
        X.append(xk)
        Y.append(th[k])

    return np.asarray(X), np.asarray(Y)


def time_series_split(u, th, val_fraction=0.2):
    """
    Split time series without shuffling.
    """
    n = len(th)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    return u[:n_train], th[:n_train], u[n_train:], th[n_train:]


def load_training_data(project_root, data_path=None, val_fraction=0.2):
    """
    Load training-val-test-data.npz and return train/validation split.
    """
    project_root = Path(project_root).resolve()

    if data_path is None:
        data_path = (
            project_root
            / "gym-unbalanced-disk"
            / "disc-benchmark-files"
            / "training-val-test-data.npz"
        )
    else:
        data_path = Path(data_path).resolve()

    data = np.load(data_path)
    u_all = np.asarray(data["u"]).reshape(-1)
    th_all = np.asarray(data["th"]).reshape(-1)

    u_all = np.clip(u_all, -3.0, 3.0)

    u_train, th_train, u_val, th_val = time_series_split(
        u_all,
        th_all,
        val_fraction=val_fraction,
    )

    return u_train, th_train, u_val, th_val, data_path




def load_gp_model(model_path):
    """
    Load either a model bundle dictionary or a saved GPNARX-like object.
    """
    model_path = Path(model_path).resolve()
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        required = ["gp", "x_scaler", "y_scaler", "na", "nb"]
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f"Model bundle is missing keys: {missing}")

        model = {
            "gp": obj["gp"],
            "x_scaler": obj["x_scaler"],
            "y_scaler": obj["y_scaler"],
            "na": int(obj["na"]),
            "nb": int(obj["nb"]),
        }
        return model

    # Fallback: saved object with attributes
    required_attrs = ["gp", "x_scaler", "y_scaler", "na", "nb"]
    missing = [a for a in required_attrs if not hasattr(obj, a)]
    if missing:
        raise ValueError(
            f"Loaded object is not a valid GP-NARX model. Missing: {missing}"
        )

    model = {
        "gp": obj.gp,
        "x_scaler": obj.x_scaler,
        "y_scaler": obj.y_scaler,
        "na": int(obj.na),
        "nb": int(obj.nb),
    }
    return model


def predict_from_x(model, X, return_std=False):
    """
    Predict using loaded model dictionary.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X[None, :]

    Xs = model["x_scaler"].transform(X)

    if return_std:
        yp_s, std_s = model["gp"].predict(Xs, return_std=True)

        yp = model["y_scaler"].inverse_transform(
            yp_s.reshape(-1, 1)
        ).reshape(-1)

        std = std_s * model["y_scaler"].scale_[0]

        return yp, std

    yp_s = model["gp"].predict(Xs)
    yp = model["y_scaler"].inverse_transform(
        yp_s.reshape(-1, 1)
    ).reshape(-1)

    return yp


def one_step_prediction(model, u, th, return_std=True):
    """
    One-step prediction using measured past angle values.
    """
    na = model["na"]
    nb = model["nb"]

    X, Y = create_io_data(u, th, na, nb)

    if return_std:
        Yp, std = predict_from_x(model, X, return_std=True)
        return X, Y, Yp, std

    Yp = predict_from_x(model, X, return_std=False)
    return X, Y, Yp


def simulate_free_run(model, u, th_initial, skip):
    """
    Free-run simulation using predicted previous outputs.
    """
    na = model["na"]
    nb = model["nb"]

    u = np.asarray(u).reshape(-1)
    th_initial = np.asarray(th_initial).reshape(-1)

    if skip < max(na, nb):
        raise ValueError("skip must be at least max(na, nb).")

    if len(th_initial) < skip:
        raise ValueError("th_initial must contain at least skip samples.")

    Y = th_initial[:skip].astype(float).tolist()

    upast = u[skip - nb:skip].astype(float).tolist()
    thpast = th_initial[skip - na:skip].astype(float).tolist()

    for uk in u[skip:]:
        x = np.concatenate([upast, thpast])
        ypred = float(predict_from_x(model, x)[0])

        Y.append(ypred)

        upast.append(float(uk))
        upast.pop(0)

        thpast.append(ypred)
        thpast.pop(0)

    return np.asarray(Y)




def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrms = rmse / np.std(y_true) * 100.0

    return {
        "rmse_rad": rmse,
        "rmse_deg": np.rad2deg(rmse),
        "mae_rad": mae,
        "nrms_percent": nrms,
    }


def print_metrics(name, y_true, y_pred):
    m = compute_metrics(y_true, y_pred)

    print(f"\n{name}")
    print(f"RMSE : {m['rmse_rad']:.6f} rad")
    print(f"RMSE : {m['rmse_deg']:.6f} deg")
    print(f"MAE  : {m['mae_rad']:.6f} rad")
    print(f"NRMS : {m['nrms_percent']:.2f} %")

    return m




def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def get_feature_labels(na, nb):
    """
    Feature ordering:
        [u[k-nb], ..., u[k-1], th[k-na], ..., th[k-1]]
    """
    labels = []

    for lag in range(nb, 0, -1):
        labels.append(f"u[k-{lag}]")

    for lag in range(na, 0, -1):
        labels.append(f"th[k-{lag}]")

    return labels




def plot_one_step_with_uncertainty(out_dir, y_true, y_pred, std):
    """
    Plot measured angle, GP mean prediction, and ±2σ uncertainty band.
    """
    x = np.arange(len(y_true))

    plt.figure(figsize=(13, 5))
    plt.plot(x, y_true, label="Measured")
    plt.plot(x, y_pred, label="GP mean prediction")

    plt.fill_between(
        x,
        y_pred - 2.0 * std,
        y_pred + 2.0 * std,
        alpha=0.25,
        label="GP ±2 standard deviations",
    )

    plt.xlabel("Validation sample")
    plt.ylabel("Angle th [rad]")
    plt.title("One-step GP-NARX prediction with uncertainty")
    plt.grid(True)
    plt.legend()

    savefig(Path(out_dir) / "01_one_step_prediction_uncertainty.png")


def plot_free_run_simulation(out_dir, th_val, th_sim, skip):
    """
    Plot measured validation signal and free-run GP simulation.
    """
    x = np.arange(skip, len(th_val))

    plt.figure(figsize=(13, 5))
    plt.plot(x, th_val[skip:], label="Measured")
    plt.plot(x, th_sim[skip:], label="GP free-run simulation")

    plt.xlabel("Validation sample")
    plt.ylabel("Angle th [rad]")
    plt.title("Free-run GP-NARX simulation")
    plt.grid(True)
    plt.legend()

    savefig(Path(out_dir) / "02_free_run_simulation.png")


def plot_residual_time_series(out_dir, y_true, y_pred, name="one_step"):
    """
    Plot prediction residual over time.
    """
    residual = y_true - y_pred

    plt.figure(figsize=(13, 4))
    plt.plot(residual)

    plt.xlabel("Sample")
    plt.ylabel("Residual [rad]")
    plt.title(f"Residual over time: {name}")
    plt.grid(True)

    savefig(Path(out_dir) / f"03_residual_time_series_{name}.png")


def plot_residual_histogram(out_dir, y_true, y_pred, name="one_step"):
    """
    Plot histogram of residuals.
    """
    residual = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=60, alpha=0.85)

    plt.xlabel("Residual [rad]")
    plt.ylabel("Count")
    plt.title(f"Residual histogram: {name}")
    plt.grid(True)

    savefig(Path(out_dir) / f"04_residual_histogram_{name}.png")


def plot_predictive_std(out_dir, std):
    """
    Plot GP predictive uncertainty over validation samples.
    """
    plt.figure(figsize=(13, 4))
    plt.plot(std)

    plt.xlabel("Validation sample")
    plt.ylabel("Predictive standard deviation [rad]")
    plt.title("GP predictive uncertainty over time")
    plt.grid(True)

    savefig(Path(out_dir) / "05_predictive_std_over_time.png")


def plot_kernel_lengthscales(out_dir, model):
    """
    Plot ARD-RBF length-scales for each NARX regressor.
    """
    gp = model["gp"]
    na = model["na"]
    nb = model["nb"]

    labels = get_feature_labels(na, nb)

    try:
        # Kernel structure:
        # ConstantKernel * RBF + WhiteKernel
        length_scales = gp.kernel_.k1.k2.length_scale
    except Exception as e:
        print("Could not extract length-scales from kernel.")
        print(f"Kernel was: {gp.kernel_}")
        print(f"Error: {e}")
        return

    length_scales = np.asarray(length_scales).reshape(-1)

    plt.figure(figsize=(12, 5))
    plt.bar(labels, length_scales)

    plt.xlabel("NARX regressor")
    plt.ylabel("ARD length-scale")
    plt.title("Optimised GP RBF length-scales")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")

    savefig(Path(out_dir) / "06_kernel_lengthscales.png")


def plot_gp_slice_1d(
    out_dir,
    model,
    x_ref,
    dim_to_vary,
    x_min,
    x_max,
    n_points=250,
    n_func_samples=8,
):
    """
    1D posterior slice:
        vary one input dimension,
        keep all others fixed,
        plot GP mean, ±2σ, and sampled posterior functions.
    """
    labels = get_feature_labels(model["na"], model["nb"])
    dim_label = labels[dim_to_vary]

    x_ref = np.asarray(x_ref).reshape(-1)
    X_plot = np.tile(x_ref, (n_points, 1))

    x_values = np.linspace(x_min, x_max, n_points)
    X_plot[:, dim_to_vary] = x_values

    X_plot_s = model["x_scaler"].transform(X_plot)

    mean_s, std_s = model["gp"].predict(X_plot_s, return_std=True)
    mean = model["y_scaler"].inverse_transform(mean_s.reshape(-1, 1)).reshape(-1)
    std = std_s * model["y_scaler"].scale_[0]

    samples_s = model["gp"].sample_y(
        X_plot_s,
        n_samples=n_func_samples,
        random_state=0,
    )

    samples = np.zeros_like(samples_s)
    for i in range(n_func_samples):
        samples[:, i] = model["y_scaler"].inverse_transform(
            samples_s[:, i].reshape(-1, 1)
        ).reshape(-1)

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, mean, label="GP posterior mean")
    plt.fill_between(
        x_values,
        mean - 2.0 * std,
        mean + 2.0 * std,
        alpha=0.25,
        label="±2 standard deviations",
    )

    for i in range(n_func_samples):
        plt.plot(
            x_values,
            samples[:, i],
            linestyle="--",
            linewidth=1,
            alpha=0.75,
        )

    plt.xlabel(dim_label)
    plt.ylabel("Predicted th[k] [rad]")
    plt.title(f"1D GP posterior slice varying {dim_label}")
    plt.grid(True)
    plt.legend()

    savefig(Path(out_dir) / f"07_gp_1d_slice_dim{dim_to_vary}_{dim_label}.png")


def plot_gp_slice_2d(
    out_dir,
    model,
    x_ref,
    dim1,
    dim2,
    x1_min,
    x1_max,
    x2_min,
    x2_max,
    n_points=80,
):
    """
    2D posterior mean slice:
        vary two input dimensions,
        keep all others fixed,
        plot predicted mean as a heatmap.
    """
    labels = get_feature_labels(model["na"], model["nb"])

    label1 = labels[dim1]
    label2 = labels[dim2]

    x_ref = np.asarray(x_ref).reshape(-1)

    x1_values = np.linspace(x1_min, x1_max, n_points)
    x2_values = np.linspace(x2_min, x2_max, n_points)

    XX1, XX2 = np.meshgrid(x1_values, x2_values)

    X_plot = np.tile(x_ref, (n_points * n_points, 1))
    X_plot[:, dim1] = XX1.reshape(-1)
    X_plot[:, dim2] = XX2.reshape(-1)

    y_pred = predict_from_x(model, X_plot).reshape(n_points, n_points)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(XX1, XX2, y_pred, levels=40)
    plt.colorbar(contour, label="Predicted th[k] [rad]")

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(f"2D GP mean slice: {label1} vs {label2}")

    savefig(Path(out_dir) / f"08_gp_2d_slice_dim{dim1}_dim{dim2}.png")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved GP model bundle .joblib file.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to DesignProject root folder.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to training-val-test-data.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction used for plotting.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=1000,
        help="Reference validation NARX sample index for GP slices.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=-1,
        help="Input dimension for 1D slice. Default -1 means th[k-1].",
    )
    parser.add_argument(
        "--dim1",
        type=int,
        default=-2,
        help="First input dimension for 2D slice. Default -2 means th[k-2].",
    )
    parser.add_argument(
        "--dim2",
        type=int,
        default=-1,
        help="Second input dimension for 2D slice. Default -1 means th[k-1].",
    )
    parser.add_argument(
        "--n-points-1d",
        type=int,
        default=250,
        help="Number of points for 1D GP slice.",
    )
    parser.add_argument(
        "--n-points-2d",
        type=int,
        default=80,
        help="Number of points per axis for 2D GP heatmap.",
    )
    parser.add_argument(
        "--n-function-samples",
        type=int,
        default=8,
        help="Number of posterior function samples in 1D slice.",
    )

    args = parser.parse_args()

    model_path = Path(args.model).resolve()

    if args.out_dir is None:
        out_dir = model_path.parent / "visualizations"
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model:")
    print(model_path)

    model = load_gp_model(model_path)

    na = model["na"]
    nb = model["nb"]

    print(f"Loaded GP-NARX model with na={na}, nb={nb}")
    print("Kernel:")
    print(model["gp"].kernel_)

    print("\nLoading data...")
    u_train, th_train, u_val, th_val, data_path = load_training_data(
        project_root=args.project_root,
        data_path=args.data,
        val_fraction=args.val_fraction,
    )

    print(f"Data file: {data_path}")
    print(f"Validation length: {len(th_val)}")


    X_val, y_true, y_pred, std = one_step_prediction(
        model,
        u_val,
        th_val,
        return_std=True,
    )

    print_metrics("One-step prediction", y_true, y_pred)

    plot_one_step_with_uncertainty(out_dir, y_true, y_pred, std)
    plot_residual_time_series(out_dir, y_true, y_pred, name="one_step")
    plot_residual_histogram(out_dir, y_true, y_pred, name="one_step")
    plot_predictive_std(out_dir, std)


    skip = max(na, nb)

    th_sim = simulate_free_run(
        model,
        u_val,
        th_initial=th_val,
        skip=skip,
    )

    print_metrics(
        "Free-run simulation",
        th_val[skip:],
        th_sim[skip:],
    )

    plot_free_run_simulation(out_dir, th_val, th_sim, skip)
    plot_residual_time_series(
        out_dir,
        th_val[skip:],
        th_sim[skip:],
        name="free_run_simulation",
    )
    plot_residual_histogram(
        out_dir,
        th_val[skip:],
        th_sim[skip:],
        name="free_run_simulation",
    )


    plot_kernel_lengthscales(out_dir, model)


    if args.ref_index < 0 or args.ref_index >= len(X_val):
        raise ValueError(
            f"ref-index must be between 0 and {len(X_val) - 1}."
        )

    x_ref = X_val[args.ref_index]

    n_features = len(x_ref)

    dim = args.dim
    dim1 = args.dim1
    dim2 = args.dim2

    if dim < 0:
        dim = n_features + dim

    if dim1 < 0:
        dim1 = n_features + dim1

    if dim2 < 0:
        dim2 = n_features + dim2

    labels = get_feature_labels(na, nb)

    print("\nFeature dimensions:")
    for i, label in enumerate(labels):
        print(f"  dim {i}: {label}")

    print(f"\nReference sample index: {args.ref_index}")
    print(f"1D slice dimension: {dim} ({labels[dim]})")
    print(f"2D slice dimensions: {dim1} ({labels[dim1]}), {dim2} ({labels[dim2]})")

    # Choose plotting ranges from validation feature distribution.
    # This avoids absurd values outside the training region.
    x_dim_values = X_val[:, dim]
    x_min = np.percentile(x_dim_values, 1)
    x_max = np.percentile(x_dim_values, 99)

    x1_values = X_val[:, dim1]
    x2_values = X_val[:, dim2]

    x1_min = np.percentile(x1_values, 1)
    x1_max = np.percentile(x1_values, 99)
    x2_min = np.percentile(x2_values, 1)
    x2_max = np.percentile(x2_values, 99)

    plot_gp_slice_1d(
        out_dir=out_dir,
        model=model,
        x_ref=x_ref,
        dim_to_vary=dim,
        x_min=x_min,
        x_max=x_max,
        n_points=args.n_points_1d,
        n_func_samples=args.n_function_samples,
    )

    plot_gp_slice_2d(
        out_dir=out_dir,
        model=model,
        x_ref=x_ref,
        dim1=dim1,
        dim2=dim2,
        x1_min=x1_min,
        x1_max=x1_max,
        x2_min=x2_min,
        x2_max=x2_max,
        n_points=args.n_points_2d,
    )

    print("\nDone.")
    print(f"All plots saved in: {out_dir}")


if __name__ == "__main__":
    main()