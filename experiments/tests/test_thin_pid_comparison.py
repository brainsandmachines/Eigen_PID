from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from benchmarks import adapters
from examples.thin_pid_comparison import (
    _common,
    compare_runtime,
    compare_runtime_optimal_covariance,
)
from examples.thin_pid_comparison.compare_gain_sweep import (
    build_gain_covariances,
    plot_gain_sweep,
)
from examples.thin_pid_comparison.compare_runtime import main as runtime_main


def _adapter_values(offset: float = 0.0) -> dict[str, float]:
    return {
        "imx": 2.0 + offset,
        "imy": 3.0 + offset,
        "imxy": 5.0 + offset,
        "union_info": 4.0 + offset,
        "obj": 4.0 + offset,
        "uix": 1.0 + offset,
        "uiy": 2.0 + offset,
        "redundancy": 1.0 + offset,
        "synergy": 1.0 + offset,
    }


def _write_runtime_config(tmp_path, filename="runtime_config.yaml", **overrides):
    config = {
        "dimensions": [1, 3],
        "vary": "X",
        "dm": 2,
        "dx": 4,
        "dy": 5,
        "trials": 2,
        "percentiles": [25, 75],
        "seed_start": 10,
    }
    config.update(overrides)
    config_path = tmp_path / filename
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_default_runtime_yaml_is_valid() -> None:
    config = compare_runtime._load_runtime_config(compare_runtime.CONFIG_PATH)
    assert config["dimensions"].size
    assert config["seed_start"] >= 0
    assert isinstance(config["resume"], bool)


def test_runtime_channel_generation_is_seeded() -> None:
    first_x, first_y = compare_runtime.generate_random_channels(3, 2, 4, seed=7)
    second_x, second_y = compare_runtime.generate_random_channels(3, 2, 4, seed=7)
    different_x, different_y = compare_runtime.generate_random_channels(3, 2, 4, seed=8)

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert not np.array_equal(first_x, different_x)
    assert not np.array_equal(first_y, different_y)


def test_gain_sequence_matches_flow_pid() -> None:
    gains = [gain for _, gain, _ in build_gain_covariances()]
    expected_gains = np.linspace(0, 3, 10)
    expected_gains[3] = 0.99
    np.testing.assert_array_equal(gains, expected_gains)


def test_gain_sweep_plot_has_five_requested_panels(monkeypatch, tmp_path) -> None:
    rows = []
    for method, offset in (
        ("analytical", 3.0),
        ("thin", 2.9998),
        ("tilde", 2.9999),
    ):
        for gain in (0.0, 1.0):
            row = {"method": method, "gain": gain}
            row.update(
                {
                    component: offset + component_index + gain
                    for component_index, (component, _, _) in enumerate(
                        _common.COMPONENTS
                    )
                }
            )
            rows.append(row)

    saved_figures = []
    save_options = []
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, path, *args, **kwargs: (
            saved_figures.append((figure, path)),
            save_options.append(kwargs),
        ),
    )

    plot_gain_sweep(pd.DataFrame(rows), tmp_path)

    assert [path for _, path in saved_figures] == [tmp_path / "gain_sweep.pdf"]
    assert save_options == [{"metadata": {"CreationDate": None}}]
    figure = saved_figures[0][0]
    axes = figure.axes[:5]
    np.testing.assert_allclose(figure.get_size_inches(), [5.6, 3.4])
    assert figure._suptitle is None
    assert [axis.get_title() for axis in axes] == [
        "Eigen-PID",
        "Thin-PID",
        "Tilde-PID",
        "Eigen minus Thin",
        "Eigen minus Tilde",
    ]
    assert [axis.get_title(loc="left") for axis in axes] == [
        "A",
        "",
        "",
        "B",
        "",
    ]
    assert [len(axis.lines) for axis in axes] == [5, 5, 5, 6, 6]
    assert figure.patch.get_alpha() is None
    np.testing.assert_allclose(axes[3].lines[0].get_ydata(), [0.0002, 0.0002])
    np.testing.assert_allclose(axes[4].lines[0].get_ydata(), [0.0001, 0.0001])
    assert axes[1].get_shared_x_axes().joined(axes[1], axes[3])
    assert axes[2].get_shared_x_axes().joined(axes[2], axes[4])
    assert all(
        axes[0].get_shared_y_axes().joined(axes[0], axis)
        for axis in axes[1:3]
    )
    assert [axis.get_ylabel() for axis in axes] == [
        "PID value\n(bits)",
        "",
        "",
        "Difference\n(bits)",
        "",
    ]
    assert all(
        any(label.get_visible() for label in axis.get_yticklabels())
        for axis in axes
    )
    assert axes[0].get_ylim()[0] < 2.9998
    assert axes[0].get_ylim()[1] > 8.0
    assert axes[3].get_ylim()[0] < 0.0
    assert axes[3].get_ylim()[1] > 0.0002
    figure.canvas.draw()
    for axis in axes[3:5]:
        magnitude = axis.yaxis.get_offset_text()
        assert magnitude.get_visible()
        assert magnitude.get_text()
        assert magnitude.get_fontsize() == 8
    for axis_index, axis in enumerate(axes):
        assert axis.title.get_fontsize() == 8
        assert axis.title.get_fontfamily() == ["Arial"]
        panel_label = axis._left_title
        if axis_index in (0, 3):
            assert panel_label.get_fontsize() == 10
            assert panel_label.get_fontfamily() == ["Arial"]
            assert panel_label.get_fontweight() == "bold"
            assert panel_label.get_position()[0] == -0.1
            assert panel_label.get_position()[1] == axis.title.get_position()[1]
        assert axis.xaxis.label.get_fontsize() == 8
        if axis_index in (0, 3):
            assert axis.yaxis.label.get_fontsize() == 8
        assert all(
            label.get_fontsize() == 7
            for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
        )
        assert all(line.get_linewidth() == 1.0 for line in axis.lines)
        assert all(line.get_markersize() == 2.5 for line in axis.lines[:5])
    for axis in axes[3:5]:
        assert axis.lines[-1].get_zorder() < axis.lines[0].get_zorder()
    legend_axis = figure.axes[5]
    assert not legend_axis.axison
    assert legend_axis.get_position().x0 < axes[3].get_position().x0
    assert legend_axis.get_position().y0 < axes[0].get_position().y0
    legend = legend_axis.get_legend()
    assert legend._ncols == 1
    assert legend.get_title().get_fontsize() == 7
    assert legend.get_title().get_fontfamily() == ["Arial"]
    assert all(label.get_fontsize() == 7 for label in legend.get_texts())
    assert all(label.get_fontfamily() == ["Arial"] for label in legend.get_texts())




def test_regular_runner_normalizes_results_and_selects_tilde(
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters.torch.cuda, "is_available", lambda: False)
    calls = []

    def fake_run(method, *args, **kwargs):
        calls.append(method)
        return _adapter_values(), {"warnings": []}, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", lambda method: None)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1)
    venkatesh_rows = adapters.run_pid_methods(
        np.eye(3), 1, 1, 1, regular_tilde="venkatesh"
    )

    assert [row["method"] for row in rows] == ["analytical", "thin", "tilde"]
    assert [row["adapter_method"] for row in rows] == [
        "eigen_math",
        "zhao_thin",
        "venkatesh_tilde",
    ]
    assert venkatesh_rows[-1]["adapter_method"] == "venkatesh_tilde"
    assert calls == [
        "eigen_math",
        "zhao_thin",
        "venkatesh_tilde",
        "eigen_math",
        "zhao_thin",
        "venkatesh_tilde",
    ]
    assert rows[0]["mutual_information_x"] == 2.0
    assert all(row["status"] == "SUCCESS" for row in rows)


def test_eigen_adapter_requests_optimal_covariance_only_when_needed(
    monkeypatch,
) -> None:
    requested = []

    def fake_eigen_pid(
        covariance, dm, dx, dy, *, compute_optimal_covariance, **_
    ):
        requested.append(compute_optimal_covariance)
        result = {
            "mix": torch.tensor(2.0),
            "miy": torch.tensor(3.0),
            "mixy": torch.tensor(5.0),
            "union_info": torch.tensor(4.0),
            "uix": torch.tensor(1.0),
            "uiy": torch.tensor(2.0),
            "red": torch.tensor(1.0),
            "syn": torch.tensor(1.0),
            "gen_ev": torch.ones(1),
            "log2_gen_ev": torch.zeros(1),
        }
        if compute_optimal_covariance:
            result["optimal_cov"] = torch.eye(3)
        return result

    monkeypatch.setattr(adapters, "eigen_pid", fake_eigen_pid)

    _, regular_details, _ = adapters._run_method(
        "eigen_math", np.eye(3), 1, 1, 1
    )
    _, optimal_details, _ = adapters._run_method(
        "eigen_math", np.eye(3), 1, 1, 1, include_optimal_cov=True
    )

    assert requested == [False, True]
    assert "optimal_cov" not in regular_details
    assert isinstance(optimal_details["optimal_cov"], np.ndarray)
    np.testing.assert_array_equal(optimal_details["optimal_cov"], np.eye(3))


def test_optimal_runner_reuses_eigen_covariance(monkeypatch) -> None:
    optimal_cov = np.full((3, 3), 2.0)
    calls = []

    def fake_run(
        method,
        covariance,
        dm,
        dx,
        dy,
        init_cov=None,
        *,
        include_optimal_cov=False,
        add_reg=False,
    ):
        calls.append((method, init_cov, include_optimal_cov))
        details = {"warnings": []}
        if include_optimal_cov:
            details["optimal_cov"] = optimal_cov
        return _adapter_values(), details, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", lambda method: None)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1, mode="optimal")

    assert [row["adapter_method"] for row in rows] == [
        "eigen_math",
        "zhao_thin",
        "venkatesh_tilde",
    ]
    assert [row["initialization"] for row in rows] == [
        "default",
        "optimal_cov",
        "optimal_cov",
    ]
    assert calls[0] == ("eigen_math", None, True)
    for _, initialization, _ in calls[1:]:
        np.testing.assert_array_equal(initialization, optimal_cov)


def test_runner_records_nonfinite_and_independent_failures(monkeypatch) -> None:
    monkeypatch.setattr(adapters.torch.cuda, "is_available", lambda: False)
    calls = []

    def fake_run(method, *args, **kwargs):
        calls.append(method)
        if method == "zhao_thin":
            raise RuntimeError("solver failed")
        values = _adapter_values()
        if method == "venkatesh_tilde":
            values["synergy"] = np.nan
        return values, {"warnings": ["warning"]}, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", lambda method: None)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1)

    assert calls == ["eigen_math", "zhao_thin", "venkatesh_tilde"]
    assert rows[1]["status"] == "EXCEPTION"
    assert rows[1]["error"] == "RuntimeError: solver failed"
    assert rows[2]["status"] == "NONFINITE"
    assert np.isnan(rows[2]["synergy"])
    assert rows[2]["warnings"] == "warning"


def test_runner_records_preparation_failure_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(adapters.torch.cuda, "is_available", lambda: False)
    calls = []

    def prepare(method):
        if method == "zhao_thin":
            raise ImportError("missing solver")

    def fake_run(method, *args, **kwargs):
        calls.append(method)
        return _adapter_values(), {"warnings": []}, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", prepare)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1)

    assert calls == ["eigen_math", "venkatesh_tilde"]
    assert rows[1]["status"] == "EXCEPTION"
    assert rows[1]["error"] == "ImportError: missing solver"
    assert rows[2]["status"] == "SUCCESS"


@pytest.mark.parametrize(
    ("updates", "expected"),
    (
        ({"uix": 0.0}, "IDENTITY_VIOLATION"),
        (
            {
                "redundancy": -0.1,
                "uix": 2.1,
                "uiy": 3.1,
                "synergy": -0.1,
            },
            "NEGATIVE_ATOM",
        ),
    ),
)
def test_runner_records_pid_diagnostic_statuses(
    monkeypatch, updates, expected
) -> None:
    def fake_run(method, *args, **kwargs):
        values = {**_adapter_values(), **updates}
        return values, {"warnings": []}, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", lambda method: None)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1)

    assert {row["status"] for row in rows} == {expected}


def test_optimal_runner_skips_optimizers_without_eigen_covariance(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(method, *args, **kwargs):
        calls.append(method)
        return _adapter_values(), {"warnings": []}, 0.1

    monkeypatch.setattr(adapters, "_prepare_method", lambda method: None)
    monkeypatch.setattr(adapters, "_run_method", fake_run)

    rows = adapters.run_pid_methods(np.eye(3), 1, 1, 1, mode="optimal")

    assert calls == ["eigen_math"]
    assert [row["status"] for row in rows] == ["SUCCESS", "SKIPPED", "SKIPPED"]


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("mode", "both"), ("regular_tilde", "unknown")),
)
def test_runner_rejects_unknown_options(keyword, value) -> None:
    with pytest.raises(ValueError, match=keyword):
        adapters.run_pid_methods(np.eye(3), 1, 1, 1, **{keyword: value})


@pytest.mark.parametrize("dimensions", ([], [[2, 3]], [0, 2], [2.5]))
def test_runtime_dimension_array_validation(dimensions, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, dimensions=dimensions)
    with pytest.raises(ValueError, match="dimensions"):
        runtime_main(config_path)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("dm", 0),
        ("dx", False),
        ("dy", 1.5),
        ("trials", 1),
        ("seed_start", -1),
        ("resume", "yes"),
        ("dimensions", [1, 1]),
    ),
)
def test_runtime_fixed_dimension_validation(keyword, value, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, **{keyword: value})
    with pytest.raises(ValueError, match=keyword):
        runtime_main(config_path)


@pytest.mark.parametrize(
    "percentiles",
    ([50, 75], [25, 50], [-1, 75], [25, 101], [25], ["low", "high"]),
)
def test_runtime_percentile_validation(percentiles, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, percentiles=percentiles)
    with pytest.raises(ValueError, match="percentiles"):
        runtime_main(config_path)


def test_runtime_config_rejects_path_inputs(tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, output_path="/tmp/results")
    with pytest.raises(ValueError, match="unknown inputs: output_path"):
        runtime_main(config_path)


@pytest.mark.parametrize(
    ("vary", "expected"),
    (
        ("M", [(1, 4, 5), (1, 4, 5), (3, 4, 5), (3, 4, 5)]),
        ("X", [(2, 1, 5), (2, 1, 5), (2, 3, 5), (2, 3, 5)]),
        ("Y", [(2, 4, 1), (2, 4, 1), (2, 4, 3), (2, 4, 3)]),
    ),
)
def test_runtime_can_vary_each_random_variable(
    monkeypatch, tmp_path, vary, expected
) -> None:
    generated = []
    plotted_csvs = []
    config_path = _write_runtime_config(tmp_path, vary=vary)

    def generate_random_channel(dm, dx, dy, seed):
        generated.append((dm, dx, dy, seed))
        return np.zeros((dx, dm)), np.zeros((dy, dm))

    monkeypatch.setattr(
        compare_runtime,
        "generate_random_channels",
        generate_random_channel,
    )
    monkeypatch.setattr(compare_runtime, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        compare_runtime,
        "run_pid_methods",
        lambda *args, **kwargs: [{"method": "analytical", "runtime_seconds": 1.0}],
    )
    monkeypatch.setattr(compare_runtime, "plot_runtime_results", plotted_csvs.append)

    runtime_main(config_path)

    assert [values[:3] for values in generated] == expected
    assert [values[3] for values in generated] == [10, 11, 12, 13]
    experiment_name = {"M": "dx_4_dy_5", "X": "dm_2_dy_5", "Y": "dm_2_dx_4"}[vary]
    experiment_directory = tmp_path / "runtime" / experiment_name
    assert plotted_csvs == [experiment_directory / "runtime_results.csv"]
    results = pd.read_csv(experiment_directory / "runtime_results.csv")
    assert results["varied_variable"].eq(vary).all()
    assert results["trial"].tolist() == [0, 1, 0, 1]
    assert results["seed"].tolist() == [10, 11, 12, 13]
    inputs = (experiment_directory / "inputs.yaml").read_text(encoding="utf-8")
    assert inputs == config_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in inputs


def test_runtime_snapshots_config_before_first_trial(monkeypatch, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path)

    def stop_before_trial(*args):
        raise RuntimeError("stop")

    monkeypatch.setattr(compare_runtime, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        compare_runtime,
        "generate_random_channels",
        stop_before_trial,
    )

    with pytest.raises(RuntimeError, match="stop"):
        runtime_main(config_path)

    saved_config = tmp_path / "runtime" / "dm_2_dy_5" / "inputs.yaml"
    assert saved_config.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )


def test_runtime_saves_each_completed_dimension(monkeypatch, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path)
    csv_path = tmp_path / "runtime" / "dm_2_dy_5" / "runtime_results.csv"

    def stop_during_second_dimension(dm, dx, dy, seed):
        if dx == 3:
            saved = pd.read_csv(csv_path)
            assert saved["dx"].tolist() == [1, 1]
            assert saved["seed"].tolist() == [10, 11]
            raise RuntimeError("stop")
        return np.zeros((dx, dm)), np.zeros((dy, dm))

    monkeypatch.setattr(compare_runtime, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        compare_runtime,
        "generate_random_channels",
        stop_during_second_dimension,
    )
    monkeypatch.setattr(
        compare_runtime,
        "run_pid_methods",
        lambda *args, **kwargs: [
            {"method": "analytical", "runtime_seconds": 1.0}
        ],
    )

    with pytest.raises(RuntimeError, match="stop"):
        runtime_main(config_path)


def test_runtime_resume_runs_only_new_dimensions(monkeypatch, tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, dimensions=[1])
    generated = []

    def generate_random_channel(dm, dx, dy, seed):
        generated.append((dm, dx, dy, seed))
        return np.zeros((dx, dm)), np.zeros((dy, dm))

    monkeypatch.setattr(compare_runtime, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(compare_runtime, "METHODS", ["analytical"])
    monkeypatch.setattr(
        compare_runtime,
        "generate_random_channels",
        generate_random_channel,
    )
    monkeypatch.setattr(
        compare_runtime,
        "run_pid_methods",
        lambda *args, **kwargs: [{"method": "analytical", "runtime_seconds": 1.0}],
    )
    monkeypatch.setattr(compare_runtime, "plot_runtime_results", lambda path: None)

    runtime_main(config_path)
    generated.clear()

    config_path = _write_runtime_config(
        tmp_path,
        dimensions=[1, 3],
        resume=True,
    )
    runtime_main(config_path)

    assert generated == [(2, 3, 5, 12), (2, 3, 5, 13)]
    experiment_directory = tmp_path / "runtime" / "dm_2_dy_5"
    results = pd.read_csv(experiment_directory / "runtime_results.csv")
    assert results["dx"].tolist() == [1, 1, 3, 3]
    assert results["seed"].tolist() == [10, 11, 12, 13]
    assert (experiment_directory / "inputs.yaml").read_text(
        encoding="utf-8"
    ) == config_path.read_text(encoding="utf-8")


def test_runtime_resume_fills_missing_seeds_per_method(
    monkeypatch, tmp_path, capsys
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        dimensions=[1],
        trials=5,
        resume=True,
    )
    experiment_directory = tmp_path / "runtime" / "dm_2_dy_5"
    experiment_directory.mkdir(parents=True)
    csv_path = experiment_directory / "runtime_results.csv"
    pd.DataFrame(
        [
            {
                "trial": trial,
                "seed": seed,
                "varied_variable": "X",
                "dm": 2,
                "dx": 1,
                "dy": 5,
                "method": method,
                "runtime_seconds": 1.0,
            }
            for method, trial, seed in (
                ("analytical", 0, 10),
                ("analytical", 1, 11),
                ("thin", 0, 10),
            )
        ]
    ).to_csv(csv_path, index=False)
    generated = []
    runner_calls = []

    def generate(dm, dx, dy, seed):
        generated.append((dm, dx, dy, seed))
        return np.zeros((dx, dm)), np.zeros((dy, dm))

    def run_missing(covariance, dm, dx, dy, *, methods):
        runner_calls.append((generated[-1][3], tuple(methods)))
        return [
            {"method": method, "runtime_seconds": 1.0}
            for method in methods
        ]

    monkeypatch.setattr(compare_runtime, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(compare_runtime, "METHODS", ["analytical", "thin"])
    monkeypatch.setattr(compare_runtime, "generate_random_channels", generate)
    monkeypatch.setattr(compare_runtime, "run_pid_methods", run_missing)
    monkeypatch.setattr(compare_runtime, "plot_runtime_results", lambda path: None)

    runtime_main(config_path)

    output = capsys.readouterr().out
    assert (
        "Seeds (12-14) are missing for method (analytical); "
        "evaluating them now."
    ) in output
    assert (
        "Seeds (11-14) are missing for method (thin); evaluating them now."
    ) in output
    assert generated == [
        (2, 1, 5, 11),
        (2, 1, 5, 12),
        (2, 1, 5, 13),
        (2, 1, 5, 14),
    ]
    assert runner_calls == [
        (11, ("thin",)),
        (12, ("analytical", "thin")),
        (13, ("analytical", "thin")),
        (14, ("analytical", "thin")),
    ]
    results = pd.read_csv(csv_path)
    assert not results.duplicated(["dx", "method", "seed"]).any()
    assert {
        method: sorted(method_rows["seed"].tolist())
        for method, method_rows in results.groupby("method")
    } == {
        "analytical": [10, 11, 12, 13, 14],
        "thin": [10, 11, 12, 13, 14],
    }

    runtime_main(config_path)

    assert len(generated) == 4
    assert len(pd.read_csv(csv_path)) == 10


def test_runtime_optimal_covariance_uses_x_sweep_and_checkpoints(
    monkeypatch, tmp_path
) -> None:
    config_path = _write_runtime_config(tmp_path)
    output_directory = (
        tmp_path / "runtime_optimal_covariance" / "dm_2_dy_5"
    )
    csv_path = output_directory / "optimal_covariance_results.csv"
    generated = []
    runner_calls = []
    plotted = []
    checked_checkpoint = False

    def result(method, adapter_method, initialization, offset):
        return {
            "method": method,
            "adapter_method": adapter_method,
            "initialization": initialization,
            "union_information": offset,
            "objective": offset + 0.25,
            **{
                field: offset
                for field in compare_runtime_optimal_covariance.RESULT_FIELDS
            },
        }

    def generate(dm, dx, dy, seed):
        nonlocal checked_checkpoint
        generated.append((dm, dx, dy, seed))
        if dx == 3 and not checked_checkpoint:
            checkpoint = pd.read_csv(csv_path)
            assert len(checkpoint) == 6
            assert checkpoint["dx"].eq(1).all()
            assert checkpoint["seed"].tolist() == [10, 10, 10, 11, 11, 11]
            checked_checkpoint = True
        return np.zeros((dx, dm)), np.zeros((dy, dm))

    def run_all(covariance, dm, dx, dy, *, mode):
        runner_calls.append((dm, dx, dy, mode))
        return [
            result("analytical", "eigen_math", "default", 3.0),
            result("thin", "zhao_thin", "optimal_cov", 1.0),
            result("tilde", "venkatesh_tilde", "optimal_cov", 2.0),
        ]

    monkeypatch.setattr(compare_runtime_optimal_covariance, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        compare_runtime_optimal_covariance,
        "generate_random_channels",
        generate,
    )
    monkeypatch.setattr(
        compare_runtime_optimal_covariance,
        "run_pid_methods",
        run_all,
    )
    monkeypatch.setattr(
        compare_runtime_optimal_covariance,
        "plot_optimal_covariance_differences",
        plotted.append,
    )

    compare_runtime_optimal_covariance.main(config_path)

    assert checked_checkpoint
    assert generated == [
        (2, 1, 5, 10),
        (2, 1, 5, 11),
        (2, 3, 5, 12),
        (2, 3, 5, 13),
    ]
    assert runner_calls == [
        (2, 1, 5, "optimal"),
        (2, 1, 5, "optimal"),
        (2, 3, 5, "optimal"),
        (2, 3, 5, "optimal"),
    ]
    assert plotted == [csv_path]
    results = pd.read_csv(csv_path)
    assert len(results) == 12
    assert {"union_information", "objective"}.issubset(results.columns)
    assert results["method"].value_counts().to_dict() == {
        "analytical": 4,
        "thin": 4,
        "tilde": 4,
    }
    assert results["initialization"].value_counts().to_dict() == {
        "optimal_cov": 8,
        "default": 4,
    }
    assert (output_directory / "inputs.yaml").read_text(
        encoding="utf-8"
    ) == config_path.read_text(encoding="utf-8")


def test_runtime_optimal_covariance_plot_shows_union_median_iqr(
    monkeypatch, tmp_path
) -> None:
    rows = []
    for dx in (2, 4):
        for trial in (0, 1):
            for method, initialization, union_value in (
                ("eigen_math", "default", 5.0),
                ("zhao_thin", "optimal_cov", 4.0 - 2 * trial),
                (
                    "venkatesh_tilde",
                    "optimal_cov",
                    3.0 - 2 * trial,
                ),
            ):
                rows.append(
                    {
                        "dx": dx,
                        "trial": trial,
                        "seed": dx + trial,
                        "method": method,
                        "initialization": initialization,
                        "union_info": union_value,
                    }
                )
    csv_path = tmp_path / "optimal_covariance_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    saved_figures = []
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, path, *args, **kwargs: saved_figures.append(
            (figure, path, kwargs)
        ),
    )

    compare_runtime_optimal_covariance.plot_optimal_covariance_differences(csv_path)

    expected_figures = (
        (
            "union_information_differences.pdf",
            "Union information difference\n(bits)",
            (2.0, 3.0),
            ((1.5, 2.5), (2.5, 3.5)),
        ),
    )
    assert len(saved_figures) == 1
    for (figure, path, options), (filename, ylabel, medians, bands) in zip(
        saved_figures, expected_figures
    ):
        assert path == tmp_path / filename
        assert options == {"metadata": {"CreationDate": None}}
        assert len(figure.axes) == 2
        axes = figure.axes
        assert [axis.get_title() for axis in axes] == [
            "Eigen-PID minus Thin-PID",
            "Eigen-PID minus Tilde-PID",
        ]
        assert [axis.get_title(loc="left") for axis in axes] == ["A", "B"]
        assert axes[0].get_shared_y_axes().joined(*axes)
        assert [axis.get_ylabel() for axis in axes] == [ylabel, ""]
        for axis in axes:
            assert axis.get_xlabel() == "Source (X) dimension"
        for axis, expected in zip(axes, medians):
            difference_lines = [
                line for line in axis.lines if line.get_label() == "Median"
            ]
            assert len(difference_lines) == 1
            line = difference_lines[0]
            np.testing.assert_array_equal(line.get_xdata(), [0, 1])
            np.testing.assert_allclose(line.get_ydata(), expected)
        for axis, expected in zip(axes, bands):
            assert len(axis.collections) == 1
            band_y = axis.collections[0].get_paths()[0].vertices[:, 1]
            np.testing.assert_allclose([band_y.min(), band_y.max()], expected)


def test_plot_runtime_results_can_rerender_saved_csv(monkeypatch, tmp_path) -> None:
    _write_runtime_config(
        tmp_path,
        "inputs.yaml",
        dimensions=[2],
        dx=2,
        dy=3,
        trials=4,
        percentiles=[10, 90],
    )
    csv_path = tmp_path / "runtime_results.csv"
    pd.DataFrame(
        {
            "trial": [0, 1, 2, 3],
            "varied_variable": ["X"] * 4,
            "dm": [2] * 4,
            "dx": [2] * 4,
            "dy": [3] * 4,
            "method": ["analytical"] * 4,
            "runtime_seconds": [1.0, 2.0, 3.0, 100.0],
        }
    ).to_csv(csv_path, index=False)
    saved_figures = []
    output_path = tmp_path / "runtime_scaling.pdf"
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, path, *args, **kwargs: saved_figures.append((figure, path)),
    )

    compare_runtime.plot_runtime_results(csv_path)
    compare_runtime.plot_runtime_results(csv_path)

    assert [path for _, path in saved_figures] == [output_path, output_path]
    axis = saved_figures[0][0].axes[0]
    assert axis.title.get_fontfamily() == ["Arial"]
    assert axis.get_xlabel() == "Source (X) dimension"
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [2.5])
    band_y = axis.collections[0].get_paths()[0].vertices[:, 1]
    np.testing.assert_allclose([band_y.min(), band_y.max()], [1.3, 70.9])
    assert axis.get_ylabel() == "Runtime (seconds)"


def test_runtime_rejects_unknown_varied_variable(tmp_path) -> None:
    config_path = _write_runtime_config(tmp_path, vary="Z")
    with pytest.raises(ValueError, match="vary"):
        runtime_main(config_path)
