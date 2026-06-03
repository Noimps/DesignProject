import joblib
from pathlib import Path
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_io_data(u, th, na, nb):

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
    Split a time series without shuffling.
    """
    n = len(th)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    return u[:n_train], th[:n_train], u[n_train:], th[n_train:]


def choose_training_subset(X, Y, max_train_samples=2000, seed=42):
    """
     select a random subset for GP hyperparameter training.
    """
    if len(X) <= max_train_samples:
        return X, Y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_train_samples, replace=False)
    idx = np.sort(idx)
    return X[idx], Y[idx]


class GPNARX:
    """
    
    Handle scaling and  NARX feature ordering.
    """

    def __init__(self, na=5, nb=5, max_train_samples=2000, seed=42):
        self.na = na
        self.nb = nb
        self.max_train_samples = max_train_samples
        self.seed = seed

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.gp = None

    def fit(self, u, th):
        X, Y = create_io_data(u, th, self.na, self.nb)
        X_fit, Y_fit = choose_training_subset(
            X, Y,
            max_train_samples=self.max_train_samples,
            seed=self.seed
        )

        X_fit_s = self.x_scaler.fit_transform(X_fit)
        Y_fit_s = self.y_scaler.fit_transform(Y_fit.reshape(-1, 1)).ravel()

        n_features = X_fit_s.shape[1]

        # ARD RBF: one length-scale per regressor.
        # WhiteKernel: measurement noise.
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-2, 1e2)
            )
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=(1e-7, 1e0)
            )
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=3,
            random_state=self.seed
        )

        print(f"\nTraining GP-NARX with na={self.na}, nb={self.nb}")
        print(f"Total NARX samples: {len(X)}")
        print(f"Samples used for GP training: {len(X_fit)}")
        t0 = time.time()
        self.gp.fit(X_fit_s, Y_fit_s)
        print(f"Training time: {time.time() - t0:.1f} s")
        print("Optimised kernel:")
        print(self.gp.kernel_)

        return self

    def predict_from_x(self, x, return_std=False):
        """
        Predict from one or more NARX regressors.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]

        xs = self.x_scaler.transform(x)

        if return_std:
            yp_s, std_s = self.gp.predict(xs, return_std=True)
            yp = self.y_scaler.inverse_transform(yp_s.reshape(-1, 1)).ravel()
            std = std_s * self.y_scaler.scale_[0]
            return yp, std

        yp_s = self.gp.predict(xs)
        yp = self.y_scaler.inverse_transform(yp_s.reshape(-1, 1)).ravel()
        return yp

    def one_step_prediction(self, u, th, return_std=False):
        """
         prediction using measured past th.
        """
        X, Y = create_io_data(u, th, self.na, self.nb)

        if return_std:
            Yp, std = self.predict_from_x(X, return_std=True)
            return Y, Yp, std

        Yp = self.predict_from_x(X, return_std=False)
        return Y, Yp

    def hidden_prediction(self, upast, thpast):
        """
        Prediction for hidden-test-prediction-submission-file.npz.

        Provided arrays:
            upast:   N x 15, columns [u[k-15], ..., u[k-1]]
            thpast:  N x 15, columns [th[k-15], ..., th[k-1]]

         use only the last nb and last na columns.
        """
        X = np.concatenate(
            [upast[:, 15 - self.nb:], thpast[:, 15 - self.na:]],
            axis=1
        )
        return self.predict_from_x(X)

    def simulate(self, u, th_initial, skip=50):

        u = np.asarray(u).reshape(-1)
        th_initial = np.asarray(th_initial).reshape(-1)

        if skip < max(self.na, self.nb):
            raise ValueError("skip must be at least max(na, nb).")

        if len(th_initial) < skip:
            raise ValueError("th_initial must contain at least skip samples.")

        Y = th_initial[:skip].astype(float).tolist()
        upast = u[skip - self.nb:skip].astype(float).tolist()
        thpast = th_initial[skip - self.na:skip].astype(float).tolist()

        for uk in u[skip:]:
            x = np.concatenate([upast, thpast])
            ypred = float(self.predict_from_x(x)[0])

            Y.append(ypred)

            upast.append(float(uk))
            upast.pop(0)

            thpast.append(ypred)
            thpast.pop(0)

        return np.asarray(Y)



def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrms = rmse / np.std(y_true) * 100.0

    print(f"\n{name}")
    print(f"RMSE : {rmse:.6f} rad")
    print(f"RMSE : {np.rad2deg(rmse):.6f} deg")
    print(f"MAE  : {mae:.6f} rad")
    print(f"NRMS : {nrms:.2f} %")

    return {"rmse": rmse, "mae": mae, "nrms": nrms}


def plot_prediction(path, y_true, y_pred, std=None, title=""):
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="Measured")
    plt.plot(y_pred, label="GP")

    if std is not None:
        x = np.arange(len(y_pred))
        plt.fill_between(
            x,
            y_pred - 2.0 * std,
            y_pred + 2.0 * std,
            alpha=0.2,
            label="GP ±2 std"
        )

    plt.xlabel("Sample")
    plt.ylabel("Angle th [rad]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to DesignProject-master folder."
    )
    parser.add_argument("--na", type=int, default=5)
    parser.add_argument("--nb", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    benchmark_dir = project_root / "gym-unbalanced-disk" / "disc-benchmark-files"
    out_dir = project_root / "SystemModelling" / "gp_narx_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = benchmark_dir / "training-val-test-data.npz"
    pred_path = benchmark_dir / "hidden-test-prediction-submission-file.npz"
    sim_path = benchmark_dir / "hidden-test-simulation-submission-file.npz"

    print("Using benchmark directory:")
    print(benchmark_dir)

    train_data = np.load(train_path)
    u_all = train_data["u"]
    th_all = train_data["th"]

    # Assignment voltage range.
    u_all = np.clip(u_all, -3.0, 3.0)

    print("\nLoaded training data")
    print(f"u shape : {u_all.shape}")
    print(f"th shape: {th_all.shape}")

    u_train, th_train, u_val, th_val = time_series_split(
        u_all, th_all,
        val_fraction=args.val_fraction
    )


    model = GPNARX(
        na=args.na,
        nb=args.nb,
        max_train_samples=args.max_train_samples,
        seed=args.seed
    )
    model.fit(u_train, th_train)

    # Save trained GP model 
    model_bundle = {
        "gp": model.gp,
        "x_scaler": model.x_scaler,
        "y_scaler": model.y_scaler,
        "na": model.na,
        "nb": model.nb,
        "max_train_samples": model.max_train_samples,
        "seed": model.seed,
        "kernel": str(model.gp.kernel_),
    }

    model_path = out_dir / f"gp_narx_bundle_na{args.na}_nb{args.nb}.joblib"
    joblib.dump(model_bundle, model_path)

    print("\nSaved GP model bundle:")
    print(model_path)


    model.fit(u_train, th_train)

    y_true_pred, y_pred, y_std = model.one_step_prediction(
        u_val, th_val,
        return_std=True
    )

    print_metrics("Validation one-step prediction", y_true_pred, y_pred)

    plot_prediction(
        out_dir / "validation_one_step_prediction.png",
        y_true_pred,
        y_pred,
        std=y_std,
        title=f"GP-NARX one-step prediction, na={args.na}, nb={args.nb}"
    )


    skip_val = max(args.na, args.nb)
    th_sim_val = model.simulate(
        u_val,
        th_initial=th_val,
        skip=skip_val
    )

    print_metrics(
        "Validation free-run simulation",
        th_val[skip_val:],
        th_sim_val[skip_val:]
    )

    plot_prediction(
        out_dir / "validation_free_run_simulation.png",
        th_val[skip_val:],
        th_sim_val[skip_val:],
        title=f"GP-NARX free-run simulation, na={args.na}, nb={args.nb}"
    )

    pred_data = np.load(pred_path)
    upast_test = pred_data["upast"]
    thpast_test = pred_data["thpast"]

    thnow_pred = model.hidden_prediction(upast_test, thpast_test)

    np.savez(
        out_dir / "gp_hidden_prediction_submission.npz",
        upast=upast_test,
        thpast=thpast_test,
        thnow=thnow_pred
    )

    print("\nSaved hidden prediction submission:")
    print(out_dir / "gp_hidden_prediction_submission.npz")

    sim_data = np.load(sim_path)
    u_test = np.clip(sim_data["u"], -3.0, 3.0)
    th_test_template = sim_data["th"]

    skip_hidden = 50
    th_hidden_sim = model.simulate(
        u_test,
        th_initial=th_test_template,
        skip=skip_hidden
    )

    th_hidden_sim[:skip_hidden] = th_test_template[:skip_hidden]

    np.savez(
        out_dir / "gp_hidden_simulation_submission.npz",
        th=th_hidden_sim,
        u=u_test
    )

    print("\nSaved hidden simulation submission:")
    print(out_dir / "gp_hidden_simulation_submission.npz")

    print("\nYou can check the files with:")
    print(f"python {benchmark_dir / 'submission-file-checker.py'} "
          f"{out_dir / 'gp_hidden_prediction_submission.npz'} {pred_path}")
    print(f"python {benchmark_dir / 'submission-file-checker.py'} "
          f"{out_dir / 'gp_hidden_simulation_submission.npz'} {sim_path}")



if __name__ == "__main__":
    main()
